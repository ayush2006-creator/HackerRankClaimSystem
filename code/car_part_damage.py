"""
car_part_damage.py
Two-stage car damage detection, both stages free / pretrained:

  Stage 1 -- Part localization:
    Roboflow Universe "Car Damage Detection" YOLOv8 model
    (bonnet, bumper, dickey, door, fender, light, windshield)
    https://universe.roboflow.com/capstone-nh0nc/car-damage-detection-t0g92
    Requires env var ROBOFLOW_API_KEY (free account).
    Weights are downloaded once and cached locally on first call.

  Stage 2 -- Damage type classification:
    beingamit99/car_damage_detection (Hugging Face, ViT/BEiT, MIT license)
    Runs fully locally via `transformers`, no API key needed.
    Classes: crack, scratch, tire_flat, dent, glass_shatter, lamp_broken
    https://huggingface.co/beingamit99/car_damage_detection

For each part box detected in Stage 1, we crop that region and run it
through Stage 2 to get a damage type -> gives us (part, damage) pairs.
If Stage 1 finds no part boxes (e.g. damage model didn't trigger), we
fall back to running Stage 2 on the whole image once, with part="unknown".
"""

import os
from typing import List, Optional

from PIL import Image

from schema import PartDamageFinding, CAR_PARTS, CAR_DAMAGE_TYPES, Severity

ROBOFLOW_MODEL_ID = "car-damage-detection-t0g92/1"
HF_DAMAGE_MODEL = "beingamit99/car_damage_detection"

# Roboflow's raw class names -> our normalized schema names
ROBOFLOW_LABEL_MAP = {
    "Bonnet": "hood",
    "Bumper": "front_bumper",
    "Dickey": "rear_bumper",
    "Door": "door",
    "Fender": "fender",
    "Light": "headlight",
    "Windshield": "windshield",
}

# HF model's id2label -> our normalized schema names
HF_LABEL_MAP = {
    "Crack": "crack",
    "Scratch": "scratch",
    "Tire Flat": "tire_flat",
    "Dent": "dent",
    "Glass Shatter": "glass_shatter",
    "Lamp Broken": "lamp_broken",
}

_roboflow_client = None
_hf_processor = None
_hf_model = None


def _get_roboflow_client():
    global _roboflow_client
    if _roboflow_client is None:
        from inference_sdk import InferenceHTTPClient
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY environment variable not set. "
                "Get a free key at https://app.roboflow.com/settings/api"
            )
        _roboflow_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
    return _roboflow_client


def _get_hf_damage_model():
    global _hf_processor, _hf_model
    if _hf_model is None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        _hf_processor = AutoImageProcessor.from_pretrained(HF_DAMAGE_MODEL)
        _hf_model = AutoModelForImageClassification.from_pretrained(HF_DAMAGE_MODEL)
    return _hf_processor, _hf_model


def _classify_damage_crop(crop: Image.Image) -> tuple[str, float]:
    """Runs the HF damage classifier on a single crop (or full image)."""
    import torch
    import numpy as np

    processor, model = _get_hf_damage_model()
    inputs = processor(images=crop, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.detach().cpu().numpy()
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()[0]
    predicted_id = int(np.argmax(logits))
    confidence = float(np.max(probs))

    raw_label = model.config.id2label[predicted_id]
    damage_type = HF_LABEL_MAP.get(raw_label, "none")

    return damage_type, confidence


def _severity_from_confidence(confidence: float) -> str:
    """Rough heuristic: higher classifier confidence on visible damage
    tends to correlate with more obvious/severe damage. This is a
    placeholder heuristic, not a calibrated severity model -- swap in
    something better if you have labeled severity data."""
    if confidence >= 0.85:
        return Severity.HIGH.value
    elif confidence >= 0.6:
        return Severity.MEDIUM.value
    elif confidence >= 0.4:
        return Severity.LOW.value
    return Severity.UNKNOWN.value


def detect_car_parts_and_damage(
    image_path: str,
    damage_confidence_threshold: float = 0.35,
) -> List[PartDamageFinding]:
    """
    Runs the full two-stage pipeline on one car image and returns a
    list of (part, damage_type) findings.
    """
    import config
    from class_mappings import CURACEL_CLASS_MAP, normalize_part, normalize_issue

    direct_findings: List[PartDamageFinding] = []
    part_boxes = []

    full_image = Image.open(image_path).convert("RGB")

    # ---- Stage 1: part localization via Roboflow (YOLO or Workflow) ----
    model_id = config.ROBOFLOW_CAR_MODEL or ROBOFLOW_MODEL_ID
    use_workflow = "general-segmentation-api" in model_id

    try:
        if use_workflow:
            from inference_sdk import InferenceHTTPClient
            api_key = os.environ.get("ROBOFLOW_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ROBOFLOW_API_KEY environment variable not set. "
                    "Get a free key at https://app.roboflow.com/settings/api"
                )
            wf_client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=api_key,
            )
            classes_param = "bonnet-dent, doorouter-dent, doorouter-paint-trace, doorouter-scratch, fender-dent, front-bumper-dent, major-rear-bumper-dent, quarter-panel-dent, pillar-dent, body-panel-dent, front-bumper-scratch, paint-chip, paint-trace, front-windscreen damage, rear-windscreen damage, headlight damage, taillight damage, side-mirror damage, bonnet, bumper, door, fender, headlight, taillight, side-mirror, windshield, scratch, dent, crack"
            
            result = wf_client.run_workflow(
                workspace_name="ayushs-workspace-zfnzn",
                workflow_id=model_id,
                images={"image": image_path},
                parameters={"classes": classes_param},
                use_cache=True
            )
            if isinstance(result, list) and len(result) > 0:
                predictions = result[0].get('predictions', {}).get('predictions', [])
            elif isinstance(result, dict):
                predictions = result.get('predictions', {}).get('predictions', [])
            else:
                predictions = []
        else:
            client = _get_roboflow_client()
            result = client.infer(image_path, model_id=model_id)
            predictions = result.get("predictions", []) if isinstance(result, dict) else []

        for pred in predictions:
            raw_label = pred.get("class", "").strip()
            raw_label_lower = raw_label.lower()
            
            cx, cy = pred.get("x", 0), pred.get("y", 0)
            w, h = pred.get("width", 0), pred.get("height", 0)
            x1, y1 = max(0, cx - w / 2), max(0, cy - h / 2)
            x2, y2 = cx + w / 2, cy + h / 2
            bbox = (x1, y1, x2, y2)
            conf = pred.get("confidence", 0.0)

            # Map part and damage type
            part_name = normalize_part(raw_label, "car")
            damage_type = normalize_issue(raw_label, "car")

            # Check if this raw label is in CURACEL_CLASS_MAP and is combined
            is_combined = False
            if raw_label_lower in CURACEL_CLASS_MAP:
                mapped_damage = CURACEL_CLASS_MAP[raw_label_lower][1]
                if mapped_damage != "unknown":
                    is_combined = True
                    damage_type = mapped_damage

            if is_combined:
                direct_findings.append(PartDamageFinding(
                    part=part_name,
                    damage_type=damage_type,
                    confidence=round(conf, 3),
                    severity=_severity_from_confidence(conf),
                    bbox=bbox,
                ))
            else:
                if part_name != "unknown":
                    part_boxes.append({
                        "part": part_name,
                        "bbox": bbox,
                        "part_confidence": conf,
                    })
    except Exception as e:
        part_boxes = []
        _stage1_error = str(e)
    else:
        _stage1_error = None

    # ---- Stage 2: damage classification per part crop ----
    findings = list(direct_findings)

    if part_boxes:
        for pb in part_boxes:
            x1, y1, x2, y2 = pb["bbox"]
            try:
                crop = full_image.crop((int(x1), int(y1), int(x2), int(y2)))
                if crop.width < 10 or crop.height < 10:
                    continue
                damage_type, conf = _classify_damage_crop(crop)
            except Exception:
                continue

            if conf < damage_confidence_threshold:
                continue  # not confident enough to claim damage here

            findings.append(PartDamageFinding(
                part=pb["part"],
                damage_type=damage_type,
                confidence=round(conf, 3),
                severity=_severity_from_confidence(conf),
                bbox=pb["bbox"],
            ))
    else:
        # Fallback: only if we got no direct findings AND no part boxes
        if not findings:
            try:
                damage_type, conf = _classify_damage_crop(full_image)
                if conf >= damage_confidence_threshold:
                    findings.append(PartDamageFinding(
                        part="unknown",
                        damage_type=damage_type,
                        confidence=round(conf, 3),
                        severity=_severity_from_confidence(conf),
                        bbox=None,
                    ))
            except Exception:
                pass

    return findings