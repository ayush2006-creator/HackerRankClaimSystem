import logging
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

import config
from schema import ObjectType

# Singleton model loading
_clip_model = None
_clip_processor = None

def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        logging.info(f"Loading CLIP model: {config.CLIP_MODEL_NAME}")
        _clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
        _clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
    return _clip_model, _clip_processor

def verify_object(image_path: str, claim_object: str) -> dict:
    """
    Zero-shot classifies an image to verify it matches the claim_object.
    Returns: {predicted_label: str, confidence: float, match: bool, risk_flags: list}
    """
    model, processor = _load_clip()
    
    labels = [
        "a photo of a car",
        "a photo of a laptop computer",
        "a photo of a shipping package or cardboard box"
    ]
    
    # Map label index to ObjectType string
    label_to_obj = {
        0: ObjectType.CAR.value,
        1: ObjectType.LAPTOP.value,
        2: ObjectType.PACKAGE.value
    }
    
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze().tolist()
        
        best_idx = probs.index(max(probs))
        predicted_obj = label_to_obj[best_idx]
        confidence = probs[best_idx]
        
        match = (predicted_obj == claim_object)
        risk_flags = []
        
        # If mismatch is highly confident, flag it
        if not match and confidence > config.OBJECT_CONFIDENCE_FLOOR:
            risk_flags.append("wrong_object")
            
        return {
            "predicted_label": predicted_obj,
            "confidence": confidence,
            "match": match,
            "risk_flags": risk_flags
        }
        
    except Exception as e:
        logging.error(f"CLIP verification failed for {image_path}: {e}")
        return {
            "predicted_label": "unknown",
            "confidence": 0.0,
            "match": False,
            "risk_flags": []
        }

def verify_all_images(image_paths: list, claim_object: str) -> dict:
    """
    Runs verify_object on all images.
    Returns a dict mapping image_id to its verification result.
    """
    results = {}
    for path in image_paths:
        import os
        img_id = os.path.splitext(os.path.basename(path))[0]
        results[img_id] = verify_object(path, claim_object)
    return results
