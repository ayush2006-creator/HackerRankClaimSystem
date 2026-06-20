"""
image_quality.py
Local, free, no-API-key blur / usability check applied to every image
before any model call. Uses OpenCV's variance-of-Laplacian method --
a standard, fast blur metric. Cheap pre-filter so we don't waste
Gemini calls on unusable images.
"""

import cv2
import numpy as np
from PIL import Image

from schema import ImageQuality

# Tunable thresholds -- variance of Laplacian below this = blurry.
# 100 is a common starting point for "reasonably sharp photo" in
# the OpenCV blur-detection literature; tune against your sample set.
BLUR_VARIANCE_THRESHOLD = 35.0
MIN_DIMENSION_PX = 200  # reject tiny/thumbnail images outright


def _variance_of_laplacian(gray_image: np.ndarray) -> float:
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


def check_image_quality(image_path: str) -> ImageQuality:
    """
    Loads an image and returns whether it is usable for automated review.
    Catches: corrupt files, too-small images, and blur.
    """
    try:
        # Verify the file opens and isn't corrupt
        with Image.open(image_path) as pil_img:
            pil_img.verify()

        # Re-open after verify() (verify() leaves the file unusable for further ops)
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            return ImageQuality(
                valid_image=False,
                is_blurry=False,
                blur_score=0.0,
                reason="Could not decode image file.",
            )

        h, w = cv_img.shape[:2]
        if h < MIN_DIMENSION_PX or w < MIN_DIMENSION_PX:
            return ImageQuality(
                valid_image=False,
                is_blurry=False,
                blur_score=0.0,
                reason=f"Image too small ({w}x{h}) to assess reliably.",
            )

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        score = _variance_of_laplacian(gray)
        is_blurry = score < BLUR_VARIANCE_THRESHOLD

        return ImageQuality(
            valid_image=not is_blurry,
            is_blurry=is_blurry,
            blur_score=round(float(score), 2),
            reason="Image too blurry for reliable assessment." if is_blurry else "OK",
        )

    except Exception as e:
        return ImageQuality(
            valid_image=False,
            is_blurry=False,
            blur_score=0.0,
            reason=f"Image failed to load: {e}",
        )