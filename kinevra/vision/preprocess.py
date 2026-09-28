"""Image preprocessing: resize, colour conversion, CLAHE, ROI crop + upsample.

Mirroring is for display only (`mirror_for_display`); analysis always uses the unmirrored
frame so that left/right landmark semantics stay correct.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from kinevra.vision.types import Image


def resize_to_width(image: Image, width: int) -> Image:
    h, w = image.shape[:2]
    if width <= 0:
        raise ValueError("width must be > 0")
    if w == width:
        return image.copy()
    height = max(1, round(h * width / w))
    interp = cv2.INTER_AREA if width < w else cv2.INTER_LINEAR
    return np.asarray(cv2.resize(image, (width, height), interpolation=interp), dtype=np.uint8)


def to_rgb(image: Image) -> Image:
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def to_gray(image: Image) -> Image:
    if image.ndim == 2:
        return image
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8)


def apply_clahe(image: Image, clip_limit: float = 2.0, tile_grid: int = 8) -> Image:
    """Contrast-limited histogram equalisation on the L channel (keeps colours stable)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    if image.ndim == 2:
        return np.asarray(clahe.apply(image), dtype=np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lab = cv2.merge((clahe.apply(lightness), a, b))
    return np.asarray(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), dtype=np.uint8)


def mirror_for_display(image: Image) -> Image:
    return np.asarray(cv2.flip(image, 1), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class Roi:
    """Pixel rectangle in the full frame, plus the full-frame size for coordinate mapping."""

    x: int
    y: int
    w: int
    h: int
    frame_w: int
    frame_h: int

    def to_full_norm(self, x_norm: float, y_norm: float) -> tuple[float, float]:
        """Map normalised coords inside the ROI crop to normalised full-frame coords."""
        return (
            (self.x + x_norm * self.w) / self.frame_w,
            (self.y + y_norm * self.h) / self.frame_h,
        )

    def from_full_norm(self, x_norm: float, y_norm: float) -> tuple[float, float]:
        return (
            (x_norm * self.frame_w - self.x) / self.w,
            (y_norm * self.frame_h - self.y) / self.h,
        )


def roi_from_points(
    points: Iterable[tuple[float, float]],
    frame_w: int,
    frame_h: int,
    padding: float = 0.15,
    min_size_px: int = 32,
) -> Roi:
    """Bounding box around normalised points, padded by `padding` x box size, clamped."""
    pts = np.asarray(list(points), dtype=np.float64)
    if pts.size == 0:
        raise ValueError("need at least one point")
    xs, ys = pts[:, 0] * frame_w, pts[:, 1] * frame_h
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = padding * max(x1 - x0, y1 - y0, min_size_px)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w = max((x1 - x0) / 2 + pad, min_size_px / 2)
    half_h = max((y1 - y0) / 2 + pad, min_size_px / 2)
    left = int(np.clip(np.floor(cx - half_w), 0, frame_w - 1))
    top = int(np.clip(np.floor(cy - half_h), 0, frame_h - 1))
    right = int(np.clip(np.ceil(cx + half_w), left + 1, frame_w))
    bottom = int(np.clip(np.ceil(cy + half_h), top + 1, frame_h))
    return Roi(left, top, right - left, bottom - top, frame_w, frame_h)


def crop_roi(image: Image, roi: Roi, scale: float = 1.0) -> Image:
    """Crop `roi` and upsample by `scale` (bicubic) for higher-resolution re-analysis."""
    if scale <= 0:
        raise ValueError("scale must be > 0")
    h, w = image.shape[:2]
    if (w, h) != (roi.frame_w, roi.frame_h):
        raise ValueError(f"ROI is for {roi.frame_w}x{roi.frame_h}, image is {w}x{h}")
    crop = image[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    if scale == 1.0:
        return crop.copy()
    size = (max(1, round(roi.w * scale)), max(1, round(roi.h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return np.asarray(cv2.resize(crop, size, interpolation=interp), dtype=np.uint8)
