"""Per-frame image quality: brightness, contrast, sharpness → frame_quality + quality_flags.

All measurements are computed on a grey copy resized to a fixed width, so thresholds do not
depend on the camera resolution. Framing (`framing_flags`) and person-count (`person_flags`)
checks use pose output and are applied by the pose estimator.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import cv2
import numpy as np

from kinevra.config import QualityCfg
from kinevra.schemas import Landmark
from kinevra.vision.preprocess import resize_to_width, to_gray
from kinevra.vision.types import Image

LOW_LIGHT = "low_light"
OVEREXPOSED = "overexposed"
LOW_CONTRAST = "low_contrast"
BLURRY = "blurry"
NO_PERSON = "no_person"
MULTIPLE_PEOPLE = "multiple_people"
OUT_OF_FRAME = "out_of_frame"


@dataclass(frozen=True, slots=True)
class FrameQuality:
    brightness: float  # mean grey level, 0..255
    contrast: float  # grey-level standard deviation
    sharpness: float  # variance of the Laplacian (higher = sharper)
    score: float  # 0..1, weakest sub-score
    flags: list[str] = field(default_factory=list)


def _ramp(value: float, lo: float, hi: float) -> float:
    """0 at/below lo, 1 at/above hi, linear in between."""
    if hi <= lo:
        return 1.0 if value >= hi else 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def measure(image: Image, analysis_width: int) -> tuple[float, float, float]:
    """(brightness, contrast, sharpness) of an image."""
    gray = to_gray(image)
    if gray.shape[1] != analysis_width:
        gray = resize_to_width(gray, analysis_width)
    mean, std = cv2.meanStdDev(gray)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(mean[0][0]), float(std[0][0]), sharpness


def assess_frame(image: Image, cfg: QualityCfg) -> FrameQuality:
    brightness, contrast, sharpness = measure(image, cfg.analysis_width)

    flags: list[str] = []
    if brightness < cfg.low_light_brightness:
        flags.append(LOW_LIGHT)
    if brightness > cfg.overexposed_brightness:
        flags.append(OVEREXPOSED)
    if contrast < cfg.min_contrast:
        flags.append(LOW_CONTRAST)
    if sharpness < cfg.blur_laplacian_var:
        flags.append(BLURRY)

    dark, bright = cfg.low_light_brightness, 255.0 - cfg.overexposed_brightness
    light_score = min(
        _ramp(brightness, 0.5 * dark, 1.5 * dark),
        _ramp(255.0 - brightness, 0.5 * bright, 1.5 * bright),
    )
    contrast_score = _ramp(contrast, 0.5 * cfg.min_contrast, 2.0 * cfg.min_contrast)
    sharp_score = _ramp(sharpness, 0.5 * cfg.blur_laplacian_var, 2.0 * cfg.blur_laplacian_var)
    score = min(light_score, contrast_score, sharp_score)
    return FrameQuality(brightness, contrast, sharpness, round(score, 4), flags)


def framing_flags(
    landmarks: Mapping[str, Landmark], required: Iterable[str], visibility_threshold: float
) -> list[str]:
    """`out_of_frame` if any required landmark is missing, invisible or outside the image."""
    for name in required:
        lm = landmarks.get(name)
        if lm is None or lm.visibility < visibility_threshold:
            return [OUT_OF_FRAME]
        if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
            return [OUT_OF_FRAME]
    return []


def person_flags(person_count: int) -> list[str]:
    if person_count <= 0:
        return [NO_PERSON]
    if person_count > 1:
        return [MULTIPLE_PEOPLE]
    return []
