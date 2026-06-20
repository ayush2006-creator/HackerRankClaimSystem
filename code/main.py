"""
main.py
Entry point: reads dataset/claims.csv (+ user_history.csv +
evidence_requirements.csv), runs the pipeline on every row, writes
output.csv.

Usage:
    export GEMINI_API_KEY=...
    export ROBOFLOW_API_KEY=...
    python main.py
"""

import os
import sys
import csv
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd

from schema import ClaimPacket
from pipeline import run_pipeline


# ============================================================================
# CONFIG -- column names verified against actual CSV headers
# ============================================================================

DATASET_DIR = Path(__file__).parent.parent / "dataset"
IMAGES_BASE_DIR = DATASET_DIR / "images"

CLAIMS_CSV       = DATASET_DIR / "claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQ_CSV = DATASET_DIR / "evidence_requirements.csv"

OUTPUT_CSV = DATASET_DIR / "output.csv"

# --- claims.csv columns ---
# "user_id","image_paths","user_claim","claim_object"
# NOTE: claims.csv has NO claim_id column; we generate one from the row index.
USER_ID_COL      = "user_id"
IMAGE_PATHS_COL  = "image_paths"
USER_CLAIM_COL   = "user_claim"
CLAIM_OBJECT_COL = "claim_object"

# --- evidence_requirements.csv columns ---
# "requirement_id","claim_object","applies_to","minimum_image_evidence"
EVIDENCE_KEY_COL = "claim_object"   # key used for lookup dict

# --- user_history.csv columns ---
# "user_id","past_claim_count","accept_claim","manual_review_claim",
# "rejected_claim","last_90_days_claim_count","history_flags","history_summary"

# --- Required output column order (per README) ---
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


# ============================================================================
# Loaders
# ============================================================================

def _load_user_history(path: Path) -> dict:
    """Returns {user_id: {column: value}} for quick lookup."""
    if not path.exists():
        print(f"Warning: {path} not found, proceeding without user history.")
        return {}
    df = pd.read_csv(path)
    if USER_ID_COL not in df.columns:
        print(f"Warning: '{USER_ID_COL}' column not found in {path.name}, skipping history.")
        return {}
    return {str(row[USER_ID_COL]): row.to_dict() for _, row in df.iterrows()}


def _load_evidence_requirements(path: Path) -> dict:
    """Returns {claim_object: [list of requirement dicts]} for lookup.

    Multiple rows can share the same claim_object (one per applies_to family),
    so we group them into a list rather than overwriting on collision.
    """
    if not path.exists():
        print(f"Warning: {path} not found, proceeding without evidence requirements.")
        return {}
    df = pd.read_csv(path)
    if EVIDENCE_KEY_COL not in df.columns:
        print(f"Warning: '{EVIDENCE_KEY_COL}' column not found in {path.name}, skipping requirements.")
        return {}
    result: dict = {}
    for _, row in df.iterrows():
        key = str(row[EVIDENCE_KEY_COL]).strip().lower()
        result.setdefault(key, []).append(row.to_dict())
    return result


def _resolve_image_paths(raw_paths: str, images_dir: Path) -> list:
    """Splits semicolon-separated image paths and resolves them relative
    to the dataset images directory if they aren't already absolute/found."""
    paths = [p.strip() for p in str(raw_paths).split(";") if p.strip()]
    resolved = []
    for p in paths:
        candidate = Path(p)
        if candidate.is_file():
            resolved.append(str(candidate))
            continue
        # try relative to images_dir/test or images_dir/sample or images_dir root
        for sub in ("test", "sample", ""):
            alt = images_dir / sub / p
            if alt.is_file():
                resolved.append(str(alt))
                break
        else:
            # last resort: try relative to DATASET_DIR
            alt2 = DATASET_DIR / p
            if alt2.is_file():
                resolved.append(str(alt2))
            else:
                print(f"  Warning: could not resolve image path '{p}', skipping.")
    return resolved


# ============================================================================
# Packet builder
# ============================================================================

def build_claim_packets(claims_df: pd.DataFrame, user_history: dict, evidence_req: dict) -> list:
    packets = []
    for idx, row in claims_df.iterrows():
        # Generate a stable claim_id from the row index (0-based → 1-based string)
        claim_id = str(idx + 1)

        user_id      = str(row.get(USER_ID_COL, "")).strip()
        claim_object = str(row.get(CLAIM_OBJECT_COL, "")).strip().lower()
        user_claim   = str(row.get(USER_CLAIM_COL, ""))
        raw_paths    = row.get(IMAGE_PATHS_COL, "")

        image_paths = _resolve_image_paths(raw_paths, IMAGES_BASE_DIR)

        # evidence_req values are lists of dicts (one per applies_to family)
        # pass the full list; pipeline can filter by applies_to as needed
        ev_reqs = evidence_req.get(claim_object, evidence_req.get("all", []))

        packets.append(ClaimPacket(
            claim_id=claim_id,
            user_id=user_id,
            claim_object=claim_object,
            user_claim=user_claim,
            image_paths=image_paths,
            # preserve original image_paths string for output passthrough
            raw_image_paths=str(raw_paths),
            user_history=user_history.get(user_id, {}),
            evidence_requirement=ev_reqs,
        ))
    return packets


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run claim verification pipeline.")
    parser.add_argument("--claims-csv", type=Path, default=CLAIMS_CSV,
                        help="Path to claims CSV (default: dataset/claims.csv)")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV,
                        help="Path to write output.csv (default: dataset/output.csv)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N rows (for quick testing)")
    args = parser.parse_args()

    # API key warnings
    for key in ("GEMINI_API_KEY", "ROBOFLOW_API_KEY"):
        if not os.environ.get(key):
            print(f"Warning: {key} is not set. Calls requiring it will degrade "
                  f"gracefully to not_enough_information.")

    if not args.claims_csv.exists():
        print(f"Error: claims CSV not found at {args.claims_csv}")
        sys.exit(1)

    claims_df = pd.read_csv(args.claims_csv)
    if args.limit:
        claims_df = claims_df.head(args.limit)

    # Validate required input columns
    required_cols = [USER_ID_COL, IMAGE_PATHS_COL, USER_CLAIM_COL, CLAIM_OBJECT_COL]
    missing_cols  = [c for c in required_cols if c not in claims_df.columns]
    if missing_cols:
        print(f"Error: expected columns {missing_cols} not found in {args.claims_csv}.")
        print(f"Actual columns: {list(claims_df.columns)}")
        sys.exit(1)

    user_history = _load_user_history(USER_HISTORY_CSV)
    evidence_req = _load_evidence_requirements(EVIDENCE_REQ_CSV)

    packets = build_claim_packets(claims_df, user_history, evidence_req)
    print(f"Loaded {len(packets)} claim(s). Running pipeline...")

    rows = []
    for i, packet in enumerate(packets, 1):
        print(f"[{i}/{len(packets)}] Processing claim {packet.claim_id} "
              f"({packet.claim_object}, {len(packet.image_paths)} image(s))...")
        try:
            decision = run_pipeline(packet)
            row = decision.to_row(packet.claim_id)
        except Exception as e:
            print(f"  Error processing claim {packet.claim_id}: {e}")
            row = {
                "evidence_standard_met":        "false",
                "evidence_standard_met_reason": f"Pipeline error: {e}",
                "risk_flags":                   "none",
                "issue_type":                   "unknown",
                "object_part":                  "unknown",
                "claim_status":                 "not_enough_information",
                "claim_status_justification":   f"Processing failed: {e}",
                "supporting_image_ids":         "none",
                "valid_image":                  "false",
                "severity":                     "unknown",
            }

        # Always copy the original input fields into every output row so the
        # output columns match the required schema exactly.
        row["user_id"]      = packet.user_id
        row["image_paths"]  = packet.raw_image_paths
        row["user_claim"]   = packet.user_claim
        row["claim_object"] = packet.claim_object

        # Print the output details for the case study
        print("  -> Result:")
        print(f"     Evidence Standard Met: {row['evidence_standard_met']} (Reason: {row['evidence_standard_met_reason']})")
        print(f"     Claim Status:          {row['claim_status']} (Justification: {row['claim_status_justification']})")
        print(f"     Risk Flags:            {row['risk_flags']}")
        print(f"     Issue Type:            {row['issue_type']}")
        print(f"     Object Part:           {row['object_part']}")
        print(f"     Severity:              {row['severity']}")
        print(f"     Valid Image:           {row['valid_image']}")
        print(f"     Supporting Image IDs:  {row['supporting_image_ids']}")
        print("-" * 60)

        rows.append(row)

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\nDone. Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()