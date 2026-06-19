"""
laptop_damage.py
No good free pretrained part+damage detector exists for laptops, so
this uses the Gemini Flash VLM via structured-output prompting --
same model as object_classifier.py, different prompt.

Requires env var: GEMINI_API_KEY
"""

import os
import json
from typing import List

from google.genai import types

from schema import PartDamageFinding, LAPTOP_PARTS, LAPTOP_DAMAGE_TYPES, Severity
from gemini_client import generate_content




_PROMPT = f"""You are inspecting a photo of a laptop for an insurance/claims
verification system. Identify ANY visible damage or defects.

Allowed part values: {", ".join(LAPTOP_PARTS)}
Allowed damage_type values: {", ".join(LAPTOP_DAMAGE_TYPES)}
Allowed severity values: none, low, medium, high, unknown

If there is no visible damage, return an empty findings list.
Do not guess at damage that isn't clearly visible in the image -- only
report what you can actually see.

Respond with ONLY a JSON object, no markdown fences, no extra text:
{{
  "findings": [
    {{"part": "...", "damage_type": "...", "confidence": 0.0-1.0, "severity": "..."}}
  ]
}}
"""


def _guess_mime(path: str) -> str:
    ext = path.lower().rsplit(".", 1)[-1]
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "bmp": "image/bmp", "webp": "image/webp",
    }.get(ext, "image/jpeg")


def detect_laptop_damage(image_path: str) -> List[PartDamageFinding]:

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        raw = generate_content(
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=_guess_mime(image_path)),
                _PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

        parsed = json.loads(raw)
        raw_findings = parsed.get("findings", [])

        findings = []
        for f in raw_findings:
            part = f.get("part", "unknown")
            if part not in LAPTOP_PARTS:
                part = "unknown"
            damage_type = f.get("damage_type", "none")
            if damage_type not in LAPTOP_DAMAGE_TYPES:
                damage_type = "none"
            severity = f.get("severity", Severity.UNKNOWN.value)
            if severity not in (s.value for s in Severity):
                severity = Severity.UNKNOWN.value

            findings.append(PartDamageFinding(
                part=part,
                damage_type=damage_type,
                confidence=float(f.get("confidence", 0.0)),
                severity=severity,
                bbox=None,
            ))

        return findings

    except Exception:
        # On failure, return no findings -- pipeline.py treats an
        # image with no extractable findings + a claim of damage as
        # a not_enough_information signal, not a crash.
        return []