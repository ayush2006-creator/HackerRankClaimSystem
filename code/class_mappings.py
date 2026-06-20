"""
class_mappings.py
Maps raw class strings from Roboflow or HuggingFace models to our standardized
schema.py vocabulary (ISSUE_TYPES and *_PARTS).
"""

from typing import Dict
from schema import ObjectType

# ============================================================================
# CAR MAPPINGS
# ============================================================================
# Map Roboflow car-damage bounding boxes to CAR_PARTS
CAR_ROBOFLOW_TO_PART: Dict[str, str] = {
    "Bonnet": "hood",
    "Bumper": "front_bumper",
    "Dickey": "rear_bumper",
    "Door": "door",
    "Fender": "fender",
    "Light": "headlight",
    "Windshield": "windshield",
    # Add any other observed roboflow labels here
}

# Map HuggingFace classification to ISSUE_TYPES
# (Fixing the bug where tire_flat/lamp_broken leaked into output)
CAR_HF_TO_ISSUE: Dict[str, str] = {
    "Crack": "crack",
    "Scratch": "scratch",
    "Dent": "dent",
    "Glass Shatter": "glass_shatter",
    "Tire Flat": "broken_part",  # tire_flat is not in ISSUE_TYPES
    "Lamp Broken": "broken_part" # lamp_broken is not in ISSUE_TYPES
}

# ============================================================================
# LAPTOP MAPPINGS
# ============================================================================
LAPTOP_ROBOFLOW_TO_PART: Dict[str, str] = {
    "screen": "screen",
    "display": "screen",
    "keyboard": "keyboard",
    "keys": "keyboard",
    "trackpad": "trackpad",
    "touchpad": "trackpad",
    "hinge": "hinge",
    "lid": "lid",
    "corner": "corner",
    "port": "port",
    "base": "base",
    "body": "body",
    "chassis": "body",
}

LAPTOP_ROBOFLOW_TO_ISSUE: Dict[str, str] = {
    "crack": "crack",
    "broken": "broken_part",
    "scratch": "scratch",
    "dent": "dent",
    "missing_key": "missing_part",
    "missing": "missing_part",
    "stain": "stain",
    "water": "water_damage",
}

# ============================================================================
# PACKAGE MAPPINGS
# ============================================================================
PACKAGE_ROBOFLOW_TO_PART: Dict[str, str] = {
    "box": "box",
    "corner": "package_corner",
    "side": "package_side",
    "seal": "seal",
    "tape": "seal",
    "label": "label",
    "barcode": "label",
    "contents": "contents",
    "item": "item",
}

PACKAGE_ROBOFLOW_TO_ISSUE: Dict[str, str] = {
    "crushed": "crushed_packaging",
    "torn": "torn_packaging",
    "open": "torn_packaging",
    "hole": "torn_packaging",
    "water": "water_damage",
    "wet": "water_damage",
    "stain": "stain",
    "dent": "dent",
}

# ============================================================================
# CURACEL AI & ZERO-SHOT WORKFLOW MAPPINGS (for CAR object)
# ============================================================================
CURACEL_CLASS_MAP: Dict[str, tuple[str, str]] = {
    # Dents
    "bonnet-dent": ("hood", "dent"),
    "doorouter-dent": ("door", "dent"),
    "fender-dent": ("fender", "dent"),
    "running-board-dent": ("body", "dent"),
    "quarter-panel-dent": ("quarter_panel", "dent"),
    "quarter panel dent": ("quarter_panel", "dent"),
    "pillar-dent": ("body", "dent"),
    "body-panel-dent": ("body", "dent"),
    "front-bumper-dent": ("front_bumper", "dent"),
    "major-rear-bumper-dent": ("rear_bumper", "dent"),
    
    # Scratches / Paint
    "doorouter-scratch": ("door", "scratch"),
    "doorouter-paint-trace": ("door", "scratch"),
    "paint-chip": ("body", "scratch"),
    "paint-trace": ("body", "scratch"),
    "front-bumper-scratch": ("front_bumper", "scratch"),
    
    # Windscreens
    "front-windscreen damage": ("windshield", "crack"),
    "rear-windscreen damage": ("windshield", "crack"),
    
    # Parts
    "headlight damage": ("headlight", "broken_part"),
    "taillight damage": ("taillight", "broken_part"),
    "side-mirror damage": ("side_mirror", "broken_part"),
    "sign-light damage": ("headlight", "broken_part"),
    
    # Simple parts (to be classified for damage by HF model)
    "bonnet": ("hood", "unknown"),
    "bumper": ("front_bumper", "unknown"),
    "door": ("door", "unknown"),
    "fender": ("fender", "unknown"),
    "headlight": ("headlight", "unknown"),
    "taillight": ("taillight", "unknown"),
    "side-mirror": ("side_mirror", "unknown"),
    "windshield": ("windshield", "unknown"),
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_part(raw_label: str, claim_object: str) -> str:
    raw_lower = raw_label.lower().strip()
    
    if claim_object == ObjectType.CAR.value:
        if raw_lower in CURACEL_CLASS_MAP:
            return CURACEL_CLASS_MAP[raw_lower][0]
        # Roboflow car model usually has capitalized labels
        for k, v in CAR_ROBOFLOW_TO_PART.items():
            if k.lower() == raw_lower:
                return v
                
    elif claim_object == ObjectType.LAPTOP.value:
        for k, v in LAPTOP_ROBOFLOW_TO_PART.items():
            if k in raw_lower:
                return v
                
    elif claim_object == ObjectType.PACKAGE.value:
        for k, v in PACKAGE_ROBOFLOW_TO_PART.items():
            if k in raw_lower:
                return v
                
    return "unknown"

def normalize_issue(raw_label: str, claim_object: str) -> str:
    raw_lower = raw_label.lower().strip()
    
    if claim_object == ObjectType.CAR.value:
        if raw_lower in CURACEL_CLASS_MAP:
            val = CURACEL_CLASS_MAP[raw_lower][1]
            if val is not None:
                return val
        for k, v in CAR_HF_TO_ISSUE.items():
            if k.lower() == raw_lower:
                return v
                
    elif claim_object == ObjectType.LAPTOP.value:
        for k, v in LAPTOP_ROBOFLOW_TO_ISSUE.items():
            if k in raw_lower:
                return v
                
    elif claim_object == ObjectType.PACKAGE.value:
        for k, v in PACKAGE_ROBOFLOW_TO_ISSUE.items():
            if k in raw_lower:
                return v
                
    return "unknown"
