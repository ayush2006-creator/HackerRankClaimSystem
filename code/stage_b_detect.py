import logging

logger = logging.getLogger("stage_b")
from dataclasses import dataclass
from typing import List, Optional
from inference_sdk import InferenceHTTPClient

import config
from class_mappings import normalize_part, normalize_issue
from schema import ObjectType
from car_part_damage import detect_car_parts_and_damage
from laptop_damage import detect_laptop_damage
from package_damage import detect_package_damage

@dataclass
class Detection:
    bbox: Optional[tuple]  # (x, y, w, h) or (x1, y1, x2, y2)
    class_name: str
    issue_type: str
    object_part: str
    confidence: float
    severity: str = "unknown"

# Initialize Roboflow client
try:
    if config.ROBOFLOW_API_KEY:
        rf_client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=config.ROBOFLOW_API_KEY
        )
    else:
        rf_client = None
except Exception as e:
    logging.warning(f"Failed to initialize Roboflow client: {e}")
    rf_client = None

def _run_roboflow_detection(image_path: str, model_id: str, claim_object: str) -> List[Detection]:
    """Runs a roboflow model and normalizes the results."""
    if not rf_client or not model_id:
        return []
        
    try:
        result = rf_client.infer(image_path, model_id=model_id)
        predictions = result.get('predictions', [])
        
        detections = []
        for p in predictions:
            conf = p.get('confidence', 0)
            if conf < config.DAMAGE_CONFIDENCE_FLOOR:
                continue
                
            raw_class = p.get('class', 'unknown')
            
            # Use class mappings to normalize
            part = normalize_part(raw_class, claim_object)
            issue = normalize_issue(raw_class, claim_object)
            
            # Estimate severity (simple heuristic)
            sev = "low"
            if conf > 0.8: sev = "high"
            elif conf > 0.6: sev = "medium"
            
            x, y, w, h = p.get('x'), p.get('y'), p.get('width'), p.get('height')
            bbox = (x, y, w, h) if all(v is not None for v in [x,y,w,h]) else None
            
            detections.append(Detection(
                bbox=bbox,
                class_name=raw_class,
                issue_type=issue,
                object_part=part,
                confidence=conf,
                severity=sev
            ))
        return detections
    except Exception as e:
        logging.error(f"Roboflow detection failed for {model_id}: {e}")
        return []

def _run_roboflow_workflow(image_path: str, classes: str) -> List[Detection]:
    """Runs a Roboflow serverless workflow for zero-shot object detection."""
    if not config.ROBOFLOW_API_KEY:
        logger.warning("  [Stage B] No ROBOFLOW_API_KEY set, skipping workflow")
        return []
        
    try:
        wf_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=config.ROBOFLOW_API_KEY
        )
        
        result = wf_client.run_workflow(
            workspace_name="ayushs-workspace-zfnzn",
            workflow_id="<YOUR_WORKFLOW_ID>",
            images={"image": image_path},
            parameters={"classes": classes},
            use_cache=True
        )
        
        detections = []
        # Parse zero-shot workflow results
        # Depending on the workflow, results might be nested. 
        # Usually it's in result[0]['predictions'] or similar for zero-shot
        if isinstance(result, list) and len(result) > 0:
            predictions = result[0].get('predictions', [])
        elif isinstance(result, dict):
            # Try some common keys returned by custom workflows
            predictions = result.get('predictions', [])
            if not predictions:
                for v in result.values():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and 'class' in v[0]:
                        predictions = v
                        break
        else:
            predictions = []
            
        for p in predictions:
            conf = p.get('confidence', 0)
            if conf < config.DAMAGE_CONFIDENCE_FLOOR:
                continue
                
            raw_class = p.get('class', 'unknown')
            
            x, y, w, h = p.get('x'), p.get('y'), p.get('width'), p.get('height')
            bbox = (x, y, w, h) if all(v is not None for v in [x,y,w,h]) else None
            
            sev = "low"
            if conf > 0.8: sev = "high"
            elif conf > 0.6: sev = "medium"
            
            detections.append(Detection(
                bbox=bbox,
                class_name=raw_class,
                issue_type=raw_class, # using raw class since zero-shot is dynamic
                object_part="unknown",
                confidence=conf,
                severity=sev
            ))
        return detections
    except Exception as e:
        logger.warning(f"  [Stage B] Roboflow workflow failed (will rely on Stage D): {e}")
        return []

def detect(image_path: str, claim_object: str) -> List[Detection]:
    """
    Unified detection entry point. Routes to correct model based on claim_object.
    Returns normalized list of Detections.
    """
    
    if claim_object == ObjectType.CAR.value:
        logger.info(f"  [Stage B] CAR -> Roboflow({config.ROBOFLOW_CAR_MODEL or 'car-damage-detection-t0g92/1'}) + HuggingFace(local) | NO Gemini call")
        raw_findings = detect_car_parts_and_damage(image_path, config.DAMAGE_CONFIDENCE_FLOOR)
        return [Detection(
            bbox=f.bbox,
            class_name="car_model_raw",
            issue_type=f.damage_type,
            object_part=f.part,
            confidence=f.confidence,
            severity=f.severity
        ) for f in raw_findings]
        
    elif claim_object == ObjectType.LAPTOP.value:
        if config.ROBOFLOW_LAPTOP_MODEL:
            logger.info(f"  [Stage B] LAPTOP -> Roboflow infer({config.ROBOFLOW_LAPTOP_MODEL}) | NO Gemini call")
            detections = _run_roboflow_detection(image_path, config.ROBOFLOW_LAPTOP_MODEL, claim_object)
        else:
            logger.info(f"  [Stage B] LAPTOP -> No model configured, relying on Stage D")
            detections = []
        return detections
        
    elif claim_object == ObjectType.PACKAGE.value:
        if config.ROBOFLOW_PACKAGE_MODEL:
            logger.info(f"  [Stage B] PACKAGE -> Roboflow infer({config.ROBOFLOW_PACKAGE_MODEL}) | NO Gemini call")
            detections = _run_roboflow_detection(image_path, config.ROBOFLOW_PACKAGE_MODEL, claim_object)
        else:
            logger.info(f"  [Stage B] PACKAGE -> No model configured, relying on Stage D")
            detections = []
        return detections
        
    return []
