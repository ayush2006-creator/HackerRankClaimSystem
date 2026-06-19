import logging
from dataclasses import dataclass, field
from typing import List, Optional
import os
import cv2
import numpy as np
from PIL import Image, ExifTags
import torch
from transformers import AutoImageProcessor, AutoModel

import config
from schema import ObjectType
from stage_b_detect import Detection

# Optional easyocr, fail gracefully if not installed
try:
    import easyocr
    _reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available() or torch.backends.mps.is_available())
except ImportError:
    logging.warning("easyocr not installed, OCR identity checks disabled.")
    _reader = None

@dataclass
class InstanceVerification:
    min_similarity: float
    identity_match: Optional[bool]
    exif_flag: bool
    risk_flags: List[str] = field(default_factory=list)

# DINOv2 singleton
_dino_processor = None
_dino_model = None

def _load_dino():
    global _dino_processor, _dino_model
    if _dino_model is None:
        logging.info(f"Loading DINOv2 model: {config.DINO_MODEL_NAME}")
        _dino_processor = AutoImageProcessor.from_pretrained(config.DINO_MODEL_NAME)
        _dino_model = AutoModel.from_pretrained(config.DINO_MODEL_NAME)
    return _dino_model, _dino_processor

def _extract_exif_time(image_path: str) -> Optional[str]:
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None
            
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == 'DateTimeOriginal':
                return str(value)
    except Exception:
        pass
    return None

def _compute_dino_embedding(image_path: str, bbox: Optional[tuple] = None) -> Optional[torch.Tensor]:
    """Returns normalized embedding vector for the image/crop."""
    try:
        img = Image.open(image_path).convert("RGB")
        
        # If we have a bbox, crop to it to focus on the object
        if bbox:
            x, y, w, h = bbox
            img = img.crop((x, y, x + w, y + h))
            
        model, processor = _load_dino()
        inputs = processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Use pooler output for image representation
        embedding = outputs.pooler_output
        return torch.nn.functional.normalize(embedding, p=2, dim=1)
        
    except Exception as e:
        logging.error(f"DINO embedding failed for {image_path}: {e}")
        return None

def verify_instance(image_paths: List[str], detections_by_path: dict, claim_object: str) -> InstanceVerification:
    """
    Checks if multiple images depict the same instance of the object.
    Requires len(image_paths) > 1.
    """
    risk_flags = set()
    
    # 1. EXIF Check
    times = []
    for p in image_paths:
        t = _extract_exif_time(p)
        if t: times.append(t)
        
    exif_flag = False
    # If we have multiple times and they are far apart, flag it.
    if len(set(times)) > 1:
        exif_flag = True
        risk_flags.add("possible_manipulation")
        
    # 2. DINOv2 Similarity Check
    embeddings = []
    for p in image_paths:
        # Get the largest bounding box for this image to crop to
        dets = detections_by_path.get(p, [])
        best_bbox = None
        if dets:
            best_det = max(dets, key=lambda d: d.bbox[2]*d.bbox[3] if d.bbox else 0)
            best_bbox = best_det.bbox
            
        emb = _compute_dino_embedding(p, best_bbox)
        if emb is not None:
            embeddings.append(emb)
            
    min_sim = 1.0
    if len(embeddings) > 1:
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = torch.nn.functional.cosine_similarity(embeddings[i], embeddings[j]).item()
                if sim < min_sim:
                    min_sim = sim
                    
        if min_sim < config.DINO_SIMILARITY_THRESHOLD:
            # They don't look like the same object
            risk_flags.add("wrong_object")

    # 3. OCR Identity (Optional)
    identity_match = None
    if _reader:
        text_sets = []
        for p in image_paths:
            try:
                results = _reader.readtext(p, detail=0)
                words = set(w.upper() for w in results if len(w) > 4 and any(c.isalnum() for c in w))
                if words:
                    text_sets.append(words)
            except Exception as e:
                logging.warning(f"OCR failed for {p}: {e}")
                
        if len(text_sets) > 1:
            has_intersection = False
            for i in range(len(text_sets)):
                for j in range(i+1, len(text_sets)):
                    if text_sets[i].intersection(text_sets[j]):
                        has_intersection = True
                        break
            
            identity_match = has_intersection

    return InstanceVerification(
        min_similarity=min_sim,
        identity_match=identity_match,
        exif_flag=exif_flag,
        risk_flags=list(risk_flags)
    )
