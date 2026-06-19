"""
schema.py
Shared enums, dataclasses, and constant lists used across the claim
verification pipeline. Every other module imports from here so the
allowed values stay consistent end-to-end.

Allowed values are taken verbatim from problem_statement.md.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


# ============================================================================
# ENUMS / ALLOWED VALUES  (must match problem_statement.md exactly)
# ============================================================================

class ClaimStatus(str, Enum):
    SUPPORTED               = "supported"
    CONTRADICTED            = "contradicted"
    NOT_ENOUGH_INFORMATION  = "not_enough_information"


class Severity(str, Enum):
    NONE    = "none"
    LOW     = "low"
    MEDIUM  = "medium"
    HIGH    = "high"
    UNKNOWN = "unknown"


class BoolStr(str, Enum):
    """evidence_standard_met / valid_image are 'true'/'false' strings in the CSV."""
    TRUE  = "true"
    FALSE = "false"


class ObjectType(str, Enum):
    CAR     = "car"
    LAPTOP  = "laptop"
    PACKAGE = "package"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Issue types  (problem_statement.md § Allowed values)
# ---------------------------------------------------------------------------
ISSUE_TYPES = [
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
]

# Aliases used by per-object damage modules — all reference the same
# allowed issue_type list from the problem statement.
CAR_DAMAGE_TYPES     = ISSUE_TYPES
LAPTOP_DAMAGE_TYPES  = ISSUE_TYPES
PACKAGE_DAMAGE_TYPES = ISSUE_TYPES

# ---------------------------------------------------------------------------
# Object parts  (problem_statement.md § Allowed values)
# ---------------------------------------------------------------------------
CAR_PARTS = [
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "unknown",
]

LAPTOP_PARTS = [
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "body",
    "unknown",
]

PACKAGE_PARTS = [
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "unknown",
]

# Convenience map: claim_object → allowed parts list
PARTS_BY_OBJECT: Dict[str, List[str]] = {
    "car":     CAR_PARTS,
    "laptop":  LAPTOP_PARTS,
    "package": PACKAGE_PARTS,
}

# ---------------------------------------------------------------------------
# Risk flags  (problem_statement.md § Allowed values)
# ---------------------------------------------------------------------------
RISK_FLAGS = [
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ImageQuality:
    """Output of the blur / usability check on a single image."""
    valid_image: bool
    is_blurry:   bool
    blur_score:  float
    reason:      str = ""


@dataclass
class ObjectClassification:
    """Output of object_classifier.py for a single image."""
    image_id:    str
    object_type: str       # one of ObjectType values
    confidence:  float
    reasoning:   str = ""


@dataclass
class PartDamageFinding:
    """A single detected (part, issue_type) pair on one image."""
    part:        str
    damage_type: str       # must be a value from ISSUE_TYPES
    confidence:  float
    severity:    str       = Severity.UNKNOWN.value
    bbox:        Optional[tuple] = None   # (x1, y1, x2, y2) or None


@dataclass
class ImageFindings:
    """Aggregated findings for a single image, regardless of object type."""
    image_id:          str
    image_path:        str
    object_type:       str
    object_confidence: float
    quality:           ImageQuality
    findings:          List[PartDamageFinding] = field(default_factory=list)
    raw_notes:         str = ""


@dataclass
class TranscriptClaim:
    """What the LLM extracted from the user's chat transcript."""
    claimed_object:      str
    claimed_parts:       List[str] = field(default_factory=list)
    claimed_issue_types: List[str] = field(default_factory=list)   # values from ISSUE_TYPES
    claimed_severity:    str       = Severity.UNKNOWN.value
    summary:             str       = ""


@dataclass
class MatchResult:
    """Output of transcript_matcher.py."""
    claim_status:         str        # ClaimStatus value
    justification:        str
    supporting_image_ids: List[str]  = field(default_factory=list)
    issue_type:           str        = "unknown"   # value from ISSUE_TYPES
    object_part:          str        = "unknown"   # value from the relevant *_PARTS list
    severity:             str        = Severity.UNKNOWN.value


@dataclass
class ClaimPacket:
    """Normalized claim assembled from the input CSVs.

    raw_image_paths  – the original semicolon-separated string from claims.csv,
                       preserved so main.py can write it verbatim to output.csv.
    image_paths      – resolved absolute/relative paths used by the pipeline.
    evidence_requirement – list of requirement dicts for this claim_object
                           (one dict per applies_to family); empty list if none found.
    """
    claim_id:           str
    user_id:            str
    claim_object:       str
    user_claim:         str
    image_paths:        List[str]
    raw_image_paths:    str                    = ""
    user_history:       Dict[str, Any]         = field(default_factory=dict)
    evidence_requirement: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ClaimDecision:
    """Final output row — decision fields only.

    main.py is responsible for writing the four input passthrough columns
    (user_id, image_paths, user_claim, claim_object) into the output row,
    so to_row() returns only the decision fields.
    """
    evidence_standard_met:        str   # BoolStr value
    evidence_standard_met_reason: str
    risk_flags:                   str   # semicolon-joined RISK_FLAGS values, or "none"
    issue_type:                   str   # value from ISSUE_TYPES
    object_part:                  str   # value from the relevant *_PARTS list
    claim_status:                 str   # ClaimStatus value
    claim_status_justification:   str
    supporting_image_ids:         str   # semicolon-joined image IDs, or "none"
    valid_image:                  str   # BoolStr value
    severity:                     str   # Severity value

    def to_row(self, claim_id: str = "") -> Dict[str, str]:   # claim_id kept for backward compat
        """Return the decision fields as a plain dict.

        The four input passthrough fields (user_id, image_paths, user_claim,
        claim_object) are NOT included here — main.py adds them after this call.
        """
        return {
            "evidence_standard_met":        self.evidence_standard_met,
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags":                   self.risk_flags,
            "issue_type":                   self.issue_type,
            "object_part":                  self.object_part,
            "claim_status":                 self.claim_status,
            "claim_status_justification":   self.claim_status_justification,
            "supporting_image_ids":         self.supporting_image_ids,
            "valid_image":                  self.valid_image,
            "severity":                     self.severity,
        }