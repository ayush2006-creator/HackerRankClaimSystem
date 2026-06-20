import os
import sys
import logging
import pandas as pd
from typing import List

# Configure logging to show pipeline debug info
logging.basicConfig(
    level=logging.INFO,
    format="%(name)-12s | %(message)s",
    stream=sys.stdout
)

# Add the parent directory to the python path so we can import from code/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_claim_packets, _load_user_history, _load_evidence_requirements
from pipeline import run_pipeline
import config

def evaluate():
    print("Starting evaluation on sample_claims.csv...")
    
    # Load inputs
    from pathlib import Path
    df = pd.read_csv(config.SAMPLE_CLAIMS_CSV)
    user_history = _load_user_history(Path(config.USER_HISTORY_CSV))
    evidence_req = _load_evidence_requirements(Path(config.EVIDENCE_REQ_CSV))
    
    packets = build_claim_packets(df, user_history, evidence_req)
    
    # Run pipeline and store results
    predictions = []
    for p in packets:
        print(f"Processing claim {p.claim_id} ({p.claim_object})...")
        decision = run_pipeline(p)
        pred_dict = decision.to_row()
        pred_dict['claim_id'] = p.claim_id
        predictions.append(pred_dict)
        
    pred_df = pd.DataFrame(predictions)
    
    # Compare against ground truth in df
    print("\n--- Evaluation Results ---")
    
    # Metrics we care about
    metrics = {
        "evidence_standard_met": 0,
        "claim_status": 0,
        "issue_type": 0,
        "object_part": 0
    }
    
    total = len(df)
    
    for i, row in df.iterrows():
        # df index is 0-based, our claim_ids are 1-based strings "1", "2"
        pred = pred_df[pred_df['claim_id'] == str(i+1)].iloc[0]
        
        for k in metrics.keys():
            if str(row[k]).lower() == str(pred[k]).lower():
                metrics[k] += 1
                
    for k, v in metrics.items():
        print(f"{k} Accuracy: {v}/{total} ({v/total*100:.1f}%)")
        
    # Write report
    report_path = os.path.join(os.path.dirname(__file__), "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("# Evaluation Report\n\n")
        f.write("## Accuracy Metrics (Sample Set)\n")
        for k, v in metrics.items():
            f.write(f"- **{k}**: {v}/{total} ({v/total*100:.1f}%)\n")
            
        f.write("\n## Detailed Mismatch Breakdown\n")
        f.write("| Claim ID | Field | Expected (GT) | Predicted | Justification |\n")
        f.write("|---|---|---|---|---|\n")
        
        for i, row in df.iterrows():
            pred = pred_df[pred_df['claim_id'] == str(i+1)].iloc[0]
            claim_mismatches = []
            for k in ["evidence_standard_met", "claim_status", "issue_type", "object_part", "valid_image", "severity"]:
                val_gt = str(row.get(k, "")).lower().strip()
                val_pred = str(pred.get(k, "")).lower().strip()
                if val_gt != val_pred:
                    claim_mismatches.append((k, val_gt, val_pred))
            
            for field, gt, pr in claim_mismatches:
                just = str(pred.get("claim_status_justification", "")).replace("\n", " ")
                f.write(f"| {i+1} | {field} | {gt} | {pr} | {just} |\n")
            
        f.write("\n## Operational Analysis\n")
        f.write("- **Stage A (Object Verification)**: Local CLIP inference. Cost: Free. Call count: 1 per image.\n")
        f.write("- **Stage B (Damage Detection)**: Roboflow API / Local fallback. Call count: 1 per valid image.\n")
        f.write("- **Stage C (Instance Verification)**: Local DINOv2 inference. Cost: Free. Call count: 1 per valid image if >1 image.\n")
        f.write("- **Stage D (Reasoning)**: Amazon Bedrock (Nova Pro). Call count: Exactly 1 per claim. All images and signals sent in one batch.\n")
        f.write("\nThis architecture significantly reduces API calls compared to the naive approach by moving object verification and instance matching to local, free models, and restricting the LLM to a single reasoning step per claim.\n")
        
    print(f"\nDetailed report written to {report_path}")

if __name__ == "__main__":
    evaluate()
