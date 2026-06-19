"""
pipeline.py
Orchestrates the full per-claim pipeline:

  for each image:
    1. image_quality.check_image_quality()      -- blur/corrupt check (local, free)
    2. object_classifier.classify_image()        -- car/laptop/package/unknown (Gemini)
    3. route to car_part_damage / laptop_damage / package_damage based on (2)

  then, across all images for the claim:
    4. transcript_matcher.match_claim_to_evidence() -- final decision (Gemini)
    5. apply risk flags + evidence_standard_met logic
    6. build ClaimDecision
"""

import os
from typing import List

from schema import (
    ClaimPacket, ClaimDecision, ImageFindings, ObjectType,
    ClaimStatus, Severity, RISK_FLAGS,
)
from image_quality import check_image_quality
from object_classifier import classify_image
from car_part_damage import detect_car_parts_and_damage
from laptop_damage import detect_laptop_damage
from package_damage import detect_package_damage
from transcript_matcher import match_claim_to_evidence


# Minimum number of usable images we'd like to see before treating
# evidence as fully "standard met" -- below this we still proceed,
# but flag damage_not_visible as a risk flag.
MIN_USABLE_IMAGES_FOR_FULL_EVIDENCE = 2


def _analyze_single_image(image_path: str, image_id: str) -> ImageFindings:
    quality = check_image_quality(image_path)

    if not quality.valid_image:
        return ImageFindings(
            image_id=image_id,
            image_path=image_path,
            object_type=ObjectType.UNKNOWN.value,
            object_confidence=0.0,
            quality=quality,
            findings=[],
            raw_notes="Skipped damage detection -- image failed quality check.",
        )

    obj_result = classify_image(image_path, image_id)

    findings = []
    if obj_result.object_type == ObjectType.CAR.value:
        findings = detect_car_parts_and_damage(image_path)
    elif obj_result.object_type == ObjectType.LAPTOP.value:
        findings = detect_laptop_damage(image_path)
    elif obj_result.object_type == ObjectType.PACKAGE.value:
        findings = detect_package_damage(image_path)
    # else: unknown object type -> no detector to run, findings stays empty

    return ImageFindings(
        image_id=image_id,
        image_path=image_path,
        object_type=obj_result.object_type,
        object_confidence=obj_result.confidence,
        quality=quality,
        findings=findings,
        raw_notes=obj_result.reasoning,
    )


def _compute_risk_flags(
    claim: ClaimPacket,
    image_findings: List[ImageFindings],
) -> List[str]:
    flags = []

    usable = [imf for imf in image_findings if imf.quality.valid_image]

    if len(usable) < len(image_findings):
        flags.append("blurry_image")

    if len(usable) < MIN_USABLE_IMAGES_FOR_FULL_EVIDENCE:
        flags.append("damage_not_visible")

    # wrong_object: claim_object says one thing, majority of usable
    # images detect a different object type
    if usable:
        mismatched = [
            imf for imf in usable
            if imf.object_type not in (claim.claim_object, ObjectType.UNKNOWN.value)
        ]
        if mismatched and len(mismatched) >= len(usable):
            flags.append("wrong_object")

    # possible_manipulation: cheap check via file size as a weak proxy
    # (a real implementation would hash pixel content)
    sizes = {}
    for imf in image_findings:
        try:
            sz = os.path.getsize(imf.image_path)
            sizes[sz] = sizes.get(sz, 0) + 1
        except OSError:
            continue
    if any(count > 1 for count in sizes.values()):
        flags.append("possible_manipulation")

    # user_history_risk: from user_history.csv lookup, passed in via claim.user_history
    prior_claims = claim.user_history.get("past_claim_count")
    try:
        if prior_claims is not None and int(prior_claims) > 2:
            flags.append("user_history_risk")
    except (ValueError, TypeError):
        pass

    return flags


def run_pipeline(claim: ClaimPacket) -> ClaimDecision:
    """Runs the full pipeline for a single claim and returns the decision."""

    # ---- Step 1-3: per-image analysis ----
    image_findings: List[ImageFindings] = []
    for idx, path in enumerate(claim.image_paths):
        image_id = os.path.splitext(os.path.basename(path))[0]
        image_findings.append(_analyze_single_image(path, image_id))

    # ---- Step 4: transcript vs. evidence matching ----
    match = match_claim_to_evidence(claim.user_claim, image_findings)

    # ---- Step 5: risk flags + evidence_standard_met ----
    risk_flags = _compute_risk_flags(claim, image_findings)
    if match.claim_status == ClaimStatus.NOT_ENOUGH_INFORMATION.value and "damage_not_visible" not in risk_flags:
        # only add this flag if it wasn't already an obvious quality/mismatch issue
        if "blurry_image" not in risk_flags and "wrong_object" not in risk_flags:
            risk_flags.append("damage_not_visible")

    usable_count = sum(1 for imf in image_findings if imf.quality.valid_image)
    evidence_standard_met = usable_count >= 1 and match.claim_status != ClaimStatus.NOT_ENOUGH_INFORMATION.value

    if evidence_standard_met:
        evidence_reason = f"{usable_count} usable image(s) provided clear evidence to evaluate the claim."
    elif usable_count == 0:
        evidence_reason = "No usable images were available to evaluate the claim."
    else:
        evidence_reason = "Available images were insufficient to confidently evaluate the claim against the transcript."

    all_valid = all(imf.quality.valid_image for imf in image_findings) if image_findings else False

    # ---- Step 6: assemble ClaimDecision ----
    return ClaimDecision(
        evidence_standard_met="true" if evidence_standard_met else "false",
        evidence_standard_met_reason=evidence_reason,
        risk_flags=";".join(risk_flags) if risk_flags else "none",
        issue_type=match.issue_type,
        object_part=match.object_part,
        claim_status=match.claim_status,
        claim_status_justification=match.justification,
        supporting_image_ids=";".join(match.supporting_image_ids) if match.supporting_image_ids else "none",
        valid_image="true" if all_valid else "false",
        severity=match.severity,
    )