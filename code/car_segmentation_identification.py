# ~/code/main.py
"""
HackerRank Orchestrate - Car/Laptop/Package Claim Verification Pipeline
Option B: General car detector + region-based part classifier + damage detector
"""

import os
import sys
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
from PIL import Image

# Model imports (will be installed separately)
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO


# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================

class ClaimStatus(Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"


class Severity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class EvidenceMet(Enum):
    TRUE = "true"
    FALSE = "false"


# Car parts (common insurance claim parts)
CAR_PARTS = [
    "front_bumper", "rear_bumper", "left_front_door", "right_front_door",
    "left_rear_door", "right_rear_door", "hood", "trunk", "left_fender",
    "right_fender", "windshield", "rear_window", "left_headlight", 
    "right_headlight", "left_taillight", "right_taillight", "side_mirror_left",
    "side_mirror_right", "door_mirror_left", "door_mirror_right", "roof",
    "side_panel_left", "side_panel_right", "wheel_left_front", "wheel_right_front",
    "wheel_left_rear", "wheel_right_rear", "tire_left_front", "tire_right_front",
    "tire_left_rear", "tire_right_rear", "undercarriage", "other"
]

# Damage types
DAMAGE_TYPES = [
    "scratch", "dent", "crack", "shatter", "dislocation", "rust",
    "broken_light", "torn", "puncture", "leakage", "crushed", "none"
]

# Risk flags
RISK_FLAGS = [
    "repeat_claimant", "object_mismatch", "low_image_quality", "duplicate_images",
    "insufficient_views", "issue_not_visible", "suspicious_source", "severity_mismatch"
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ImageEvidence:
    """Structured evidence from a single image"""
    image_id: str
    valid_image: str  # "true" or "false"
    object_detected: str  # "true" or "false"
    object_type: str  # "car", "laptop", "package", "unknown"
    object_confidence: float
    visible_parts: List[str] = field(default_factory=list)
    visible_issues: List[str] = field(default_factory=list)
    severity: Severity = Severity.UNKNOWN
    supports_claim: bool = False
    contradicts_claim: bool = False
    usability_reason: str = ""
    damage_boxes: List[tuple] = field(default_factory=list)  # (x1, y1, x2, y2, class_id, conf)
    part_boxes: List[tuple] = field(default_factory=list)


@dataclass
class ClaimPacket:
    """Normalized claim from CSV row"""
    claim_id: str
    user_id: str
    claim_object: str  # car, laptop, package
    user_claim: str  # chat transcript
    image_paths: List[str]
    user_history: Dict[str, Any]
    evidence_requirement: Dict[str, Any]


@dataclass
class ClaimDecision:
    """Final output for a claim"""
    evidence_standard_met: str
    evidence_standard_met_reason: str
    risk_flags: List[str]
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: List[str]
    valid_image: str
    severity: str


# ============================================================================
# MODEL PIPELINE
# ============================================================================

class CarAnalysisPipeline:
    """
    Option B pipeline:
    1. Detect car with YOLOv8
    2. Crop car region
    3. Classify parts within crop (simple CNN or VLM)
    4. Detect damage types
    """
    
    def __init__(self):
        self.car_model = YOLO("yolov8m.pt")  # General object detector with car class
        self.damage_model = YOLO("yolov8m.pt")  # Reuse for damage detection
        self._load_models()
    
    def _load_models(self):
        """Initialize models - can be customized for production"""
        # Models are lazily downloaded by ultralytics when first used
        pass

if __name__ == "__main__":
    import os, glob
    from PIL import Image
    # Resolve sample image directory relative to this file
    sample_dir ="dataset/images/sample/case_001"
    image_paths = [p for p in glob.glob(os.path.join(sample_dir, "**", "*.*"), recursive=True)
                   if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]
    print(f"Found {len(image_paths)} sample images.")
    if not image_paths:
        print("No images to process.")
    else:
        pipeline = CarAnalysisPipeline()
        for img_path in image_paths[:5]:  # demo first few images
            try:
                img = Image.open(img_path)
                img.verify()
                print(f"Loaded {img_path} ({img.width}x{img.height})")
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
        print("Sample run complete.")
