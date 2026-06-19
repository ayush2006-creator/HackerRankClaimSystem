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
# HELPER FUNCTIONS
# ============================================================================

def normalize_part(raw_label: str, claim_object: str) -> str:
    raw_lower = raw_label.lower().strip()
    
    if claim_object == ObjectType.CAR.value:
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
