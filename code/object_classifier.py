"""
object_classifier.py
Classifies each submitted image as car / laptop / package / unknown
using the Gemini Flash free-tier API (vision capable).

Requires env var: GEMINI_API_KEY

Free tier reference (check current limits before relying on this at
scale -- Google updates these): as of writing, Gemini 2.0/2.5 Flash
free tier allows a limited number of requests per minute and per day.
Keep this in mind for batching/retry logic in pipeline.py.
"""

import os
import json
import base64
from typing import Optional

from google.genai import types

from schema import ObjectClassification
from gemini_client import generate_content




_PROMPT = """You are an image classifier for an insurance/claims verification system.

Look at this image and classify the primary object shown into exactly ONE of:
- "car" (any motor vehicle: car, SUV, truck, van)
- "laptop" (any laptop/notebook computer)
- "package" (any shipped parcel, box, or package)
- "unknown" (none of the above, or the object is not clearly identifiable)

Respond with ONLY a JSON object, no markdown fences, no extra text, in this exact format:
{"object_type": "car|laptop|package|unknown", "confidence": 0.0-1.0, "reasoning": "one short sentence"}
"""


def classify_image(image_path: str, image_id: str) -> ObjectClassification:
    """
    Sends a single image to Gemini Flash and returns its object type
    classification. On any API failure, returns "unknown" with
    confidence 0.0 rather than raising -- callers should treat that
    as a not-enough-information signal, not crash the whole run.
    """
    client = None  # no longer needed

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        mime_type = _guess_mime(image_path)

        raw = generate_content(
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                _PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

        parsed = json.loads(raw)

        obj_type = parsed.get("object_type", "unknown").lower()
        if obj_type not in ("car", "laptop", "package", "unknown"):
            obj_type = "unknown"

        return ObjectClassification(
            image_id=image_id,
            object_type=obj_type,
            confidence=float(parsed.get("confidence", 0.0)),
            reasoning=parsed.get("reasoning", ""),
        )

    except Exception as e:
        return ObjectClassification(
            image_id=image_id,
            object_type="unknown",
            confidence=0.0,
            reasoning=f"Classification failed: {e}",
        )


def _guess_mime(path: str) -> str:
    ext = path.lower().rsplit(".", 1)[-1]
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "bmp": "image/bmp",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")