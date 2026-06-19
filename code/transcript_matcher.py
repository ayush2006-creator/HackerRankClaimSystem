"""
transcript_matcher.py
Takes the user's chat transcript (free text) + the aggregated image
findings (from car_part_damage / laptop_damage / package_damage) and
asks Gemini to decide: supported / contradicted / not_enough_information.

This is the core decision step. We hand the LLM both sides (what the
user says vs. what the images show) rather than doing brittle keyword
matching ourselves, since transcripts are free-form chat text.

Requires env var: GEMINI_API_KEY
"""

import os
import json
from typing import List

from google.genai import types

from schema import MatchResult, ImageFindings, ClaimStatus, Severity
from gemini_client import generate_content




_PROMPT_TEMPLATE = """You are the final decision step in an automated insurance
claim verification pipeline. You are given:

1. The user's chat transcript describing their claim.
2. Structured findings extracted from the photos they submitted (by
   computer vision models -- already run, not something you need to redo).

Your job is to decide whether the photo evidence SUPPORTS, CONTRADICTS,
or gives NOT ENOUGH INFORMATION to evaluate the user's claim.

Rules:
- "supported": the image findings clearly show the same type of issue
  (same general part/area and same general damage type) that the user
  describes in the transcript.
- "contradicted": the images are usable and show the relevant object
  clearly, but show NO damage/issue, or show a clearly different issue
  than what the user describes (e.g. user claims a cracked windshield,
  images show an undamaged windshield, or damage to an unrelated part only).
- "not_enough_information": the images are blurry/unusable, don't show
  the relevant part/area, don't show the claimed object at all, or the
  transcript is too vague to compare against. ALSO use this if there is
  a mismatch between what the transcript describes and what little the
  images do show, but you cannot tell if that's because the damage truly
  isn't there or because the images simply don't cover the right area/angle.

User claim transcript:
---
{transcript}
---

Image findings (one block per submitted image):
---
{findings_block}
---

Respond with ONLY a JSON object, no markdown fences, no extra text:
{{
  "claim_status": "supported|contradicted|not_enough_information",
  "justification": "2-3 sentences grounded in the specific findings above",
  "supporting_image_ids": ["..."],
  "issue_type": "the single most relevant damage_type from the findings, or 'none'",
  "object_part": "the single most relevant part from the findings, or 'unknown'",
  "severity": "none|low|medium|high|unknown"
}}
"""


def _format_findings_block(image_findings: List[ImageFindings]) -> str:
    lines = []
    for imf in image_findings:
        lines.append(f"Image ID: {imf.image_id}")
        lines.append(f"  Object type detected: {imf.object_type} (confidence {imf.object_confidence:.2f})")
        lines.append(f"  Image usable: {imf.quality.valid_image} ({imf.quality.reason})")
        if imf.findings:
            for fnd in imf.findings:
                lines.append(
                    f"  - part={fnd.part}, damage_type={fnd.damage_type}, "
                    f"confidence={fnd.confidence:.2f}, severity={fnd.severity}"
                )
        else:
            lines.append("  - no damage findings detected in this image")
        lines.append("")
    return "\n".join(lines)


def match_claim_to_evidence(
    user_claim: str,
    image_findings: List[ImageFindings],
) -> MatchResult:
    """
    Core decision call. If image_findings is empty, or all images are
    invalid/unusable, we short-circuit to not_enough_information
    without even calling the LLM (saves a call, and is the correct
    answer regardless of what the LLM might guess).
    """
    usable_images = [imf for imf in image_findings if imf.quality.valid_image]

    if not image_findings:
        return MatchResult(
            claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION.value,
            justification="No images were available to evaluate against the claim.",
            supporting_image_ids=[],
            issue_type="none",
            object_part="unknown",
            severity=Severity.UNKNOWN.value,
        )

    if not usable_images:
        return MatchResult(
            claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION.value,
            justification="All submitted images failed quality checks (blurry, too small, or corrupt) and could not be reliably assessed.",
            supporting_image_ids=[],
            issue_type="none",
            object_part="unknown",
            severity=Severity.UNKNOWN.value,
        )

    findings_block = _format_findings_block(image_findings)
    prompt = _PROMPT_TEMPLATE.format(transcript=user_claim, findings_block=findings_block)

    try:
        raw = generate_content(
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        parsed = json.loads(raw)

        status = parsed.get("claim_status", "not_enough_information")
        if status not in (s.value for s in ClaimStatus):
            status = ClaimStatus.NOT_ENOUGH_INFORMATION.value

        severity = parsed.get("severity", Severity.UNKNOWN.value)
        if severity not in (s.value for s in Severity):
            severity = Severity.UNKNOWN.value

        return MatchResult(
            claim_status=status,
            justification=parsed.get("justification", ""),
            supporting_image_ids=parsed.get("supporting_image_ids", []) or [],
            issue_type=parsed.get("issue_type", "none"),
            object_part=parsed.get("object_part", "unknown"),
            severity=severity,
        )

    except Exception as e:
        # LLM call failed -- fail safe to not_enough_information rather
        # than guessing supported/contradicted.
        return MatchResult(
            claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION.value,
            justification=f"Automated comparison failed and could not be completed: {e}",
            supporting_image_ids=[],
            issue_type="none",
            object_part="unknown",
            severity=Severity.UNKNOWN.value,
        )