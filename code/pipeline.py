"""
pipeline.py
Orchestrates the full per-claim pipeline across 5 stages.
"""

import os
from typing import List, Dict
import logging

from schema import ClaimPacket, ClaimDecision, ObjectType
from image_quality import check_image_quality
from stage_a_object_verify import verify_all_images
from stage_b_detect import detect, Detection
import stage_c_instance_verify
from stage_d_gemini_reason import reason_claim
from stage_e_assemble import assemble_decision

logger = logging.getLogger("pipeline")

def run_pipeline(claim: ClaimPacket) -> ClaimDecision:
    """Runs the full 5-stage pipeline for a single claim."""
    
    logger.info(f"{'='*60}")
    logger.info(f"CLAIM {claim.claim_id} | object={claim.claim_object} | images={len(claim.image_paths)}")
    logger.info(f"{'='*60}")
    
    # --- STAGE A: Object Verification & Basic Quality ---
    logger.info(f"[Stage A] Running CLIP object verification (LOCAL, no API call)")
    stage_a_verdicts = verify_all_images(claim.image_paths, claim.claim_object)
    
    # Inject basic quality checks into Stage A verdicts
    for img_id, path in zip([os.path.splitext(os.path.basename(p))[0] for p in claim.image_paths], claim.image_paths):
        quality = check_image_quality(path)
        logger.info(f"  [DEBUG] path={path} score={quality.blur_score} valid={quality.valid_image} reason={quality.reason}")
        if img_id in stage_a_verdicts:
            stage_a_verdicts[img_id]["valid_image"] = quality.valid_image
            if not quality.valid_image:
                 stage_a_verdicts[img_id]["risk_flags"].append("blurry_image")

    for img_id, v in stage_a_verdicts.items():
        logger.info(f"  [Stage A] {img_id}: match={v.get('match')}, valid={v.get('valid_image', True)}, flags={v.get('risk_flags')}")

    # Filter paths for Stage B/C: only those that matched object AND passed quality
    usable_paths = []
    for path in claim.image_paths:
        img_id = os.path.splitext(os.path.basename(path))[0]
        v = stage_a_verdicts.get(img_id, {})
        if v.get("match", False) and v.get("valid_image", True):
            usable_paths.append(path)

    logger.info(f"[Stage A] {len(usable_paths)}/{len(claim.image_paths)} images passed object+quality filter")

    # --- STAGE B: Damage Detection ---
    logger.info(f"[Stage B] Running damage detection for {claim.claim_object}")
    stage_b_detections: Dict[str, List[Detection]] = {}
    for path in usable_paths:
        img_id = os.path.splitext(os.path.basename(path))[0]
        stage_b_detections[img_id] = detect(path, claim.claim_object)
        logger.info(f"  [Stage B] {img_id}: {len(stage_b_detections[img_id])} detections found")
        for d in stage_b_detections[img_id]:
            logger.info(f"    -> part={d.object_part}, issue={d.issue_type}, conf={d.confidence:.2f}, src={d.class_name}")

    # --- STAGE C: Instance Verification ---
    stage_c_verdict = None
    if len(usable_paths) > 1:
        logger.info(f"[Stage C] Running DINOv2 instance verification (LOCAL, no API call)")
        dets_by_path = {}
        for path in usable_paths:
            img_id = os.path.splitext(os.path.basename(path))[0]
            dets_by_path[path] = stage_b_detections.get(img_id, [])
            
        cv_result = stage_c_instance_verify.verify_instance(
            usable_paths, dets_by_path, claim.claim_object
        )
        stage_c_verdict = {
            "min_similarity": cv_result.min_similarity,
            "identity_match": cv_result.identity_match,
            "exif_flag": cv_result.exif_flag,
            "risk_flags": cv_result.risk_flags
        }
        logger.info(f"  [Stage C] similarity={cv_result.min_similarity:.2f}, identity_match={cv_result.identity_match}, flags={cv_result.risk_flags}")
    else:
        logger.info(f"[Stage C] Skipped (single image claim)")

    # --- STAGE D: Gemini Reasoning Layer ---
    logger.info(f"[Stage D] *** GEMINI API CALL *** (1 call for this claim)")
    stage_d_reasoning = reason_claim(
        claim=claim,
        stage_a_verdicts=stage_a_verdicts,
        stage_b_detections=stage_b_detections,
        stage_c_verdict=stage_c_verdict
    )
    logger.info(f"  [Stage D] Result: status={stage_d_reasoning.get('claim_status')}, issue={stage_d_reasoning.get('issue_type')}, severity={stage_d_reasoning.get('severity')}")

    # --- STAGE E: Assembly ---
    logger.info(f"[Stage E] Assembling final decision (LOCAL, no API call)")
    decision = assemble_decision(
        claim=claim,
        stage_a_verdicts=stage_a_verdicts,
        stage_b_detections=list(stage_b_detections.values()),
        stage_c_verdict=stage_c_verdict,
        stage_d_reasoning=stage_d_reasoning
    )
    logger.info(f"  [Stage E] Final: status={decision.claim_status}, issue={decision.issue_type}, severity={decision.severity}")
    logger.info(f"{'='*60}\n")

    return decision