"""
gemini_client.py
Shared, rate-limited, retry-enabled Gemini client used by all modules
that call the Gemini API (object_classifier, laptop_damage, package_damage,
transcript_matcher).

Handles:
  - Global rate limiting (respects free-tier 5 RPM for gemini-2.5-flash)
  - Automatic retry with exponential backoff for 429 and 503 errors
  - Shared singleton client (one Client instance, one set of credentials)
"""

import os
import re
import time
import logging
import threading
from typing import Optional

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"

# Model rotation: each model has its own 20 RPD free-tier quota.
# Rotating between them effectively doubles our daily limit to 40 RPD.
_MODEL_POOL = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

# 30 second cooldown between calls to stay well within rate limits
MAX_REQUESTS_PER_MINUTE = 2
_MIN_INTERVAL = 30.0

# Retry config
MAX_RETRIES = 4           # total attempts = MAX_RETRIES + 1
INITIAL_BACKOFF_S = 3.0   # first retry wait
MAX_BACKOFF_S = 15.0      # cap on exponential backoff

# ---------------------------------------------------------------------------
# Singleton client + rate limiter
# ---------------------------------------------------------------------------
_client: Optional[genai.Client] = None
_lock = threading.Lock()
_last_call_time: float = 0.0
_call_count: int = 0  # Global counter for total Gemini API calls


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _rate_limit():
    """Block until enough time has passed since the last call."""
    global _last_call_time
    with _lock:
        now = time.monotonic()
        elapsed = now - _last_call_time
        if elapsed < _MIN_INTERVAL:
            wait = _MIN_INTERVAL - elapsed
            time.sleep(wait)
        _last_call_time = time.monotonic()


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is a retryable Gemini API error (429/503)."""
    err_str = str(exc)
    return ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or
            "503" in err_str or "UNAVAILABLE" in err_str or
            "500" in err_str or "INTERNAL" in err_str)


def _extract_retry_delay(exc: Exception) -> Optional[float]:
    """Try to parse the retry delay from a Gemini 429 error message."""
    err_str = str(exc)
    match = re.search(r"retry in (\d+\.?\d*)s", err_str)
    if match:
        return float(match.group(1)) + 1.0  # add 1s margin
    return None


def generate_content(
    contents: list,
    config: Optional[types.GenerateContentConfig] = None,
    model: str = GEMINI_MODEL,
) -> str:
    """Rate-limited, retry-enabled wrapper around Gemini generate_content.

    Uses model rotation to spread calls across multiple models,
    effectively multiplying the daily quota.
    """
    client = _get_client()
    last_exc = None
    global _call_count
    logger = logging.getLogger("gemini")

    # Pick model from rotation pool based on call count
    actual_model = _MODEL_POOL[_call_count % len(_MODEL_POOL)]
    logger.info(f"  [Gemini] Attempting call #{_call_count+1} with model={actual_model}")

    for attempt in range(MAX_RETRIES + 1):
        _rate_limit()

        try:
            response = client.models.generate_content(
                model=actual_model,
                contents=contents,
                config=config,
            )
            _call_count += 1
            logger.info(f"  [Gemini] ✅ Call #{_call_count} succeeded (model={actual_model})")
            return response.text.strip()

        except Exception as e:
            last_exc = e
            err_str = str(e)

            # If this model's quota is exhausted, try the next model in pool
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                old_model = actual_model
                # Try next model in pool
                pool_idx = _MODEL_POOL.index(actual_model) if actual_model in _MODEL_POOL else 0
                next_idx = (pool_idx + 1) % len(_MODEL_POOL)
                if _MODEL_POOL[next_idx] != old_model:
                    actual_model = _MODEL_POOL[next_idx]
                    logger.warning(f"  [Gemini] ⚠️  {old_model} quota exhausted, switching to {actual_model}")
                    continue  # retry immediately with new model

            if not _is_retryable(e) or attempt == MAX_RETRIES:
                raise

            # Calculate backoff
            server_delay = _extract_retry_delay(e)
            if server_delay:
                backoff = min(server_delay, 15.0)
            else:
                backoff = min(
                    INITIAL_BACKOFF_S * (2 ** attempt),
                    MAX_BACKOFF_S,
                )

            logger.warning(f"  [Gemini] ⏳ Error (attempt {attempt+1}/{MAX_RETRIES+1}): "
                  f"{type(e).__name__}. Retrying in {backoff:.1f}s...")
            time.sleep(backoff)

    raise RuntimeError(f"All {MAX_RETRIES+1} attempts failed. Last error: {last_exc}")

