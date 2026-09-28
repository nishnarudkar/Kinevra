"""Pose estimator interface and the shared estimate() logic (ROI mode, PoseFrame building).

Concrete estimators only implement `infer(image) -> PoseResult` in pixel coordinates of the
image they were given. This class handles:
- ROI mode: crop + upsample (+ optional CLAHE) a region of the full frame, run the model on
  it, and map the landmarks back to normalised full-frame coordinates. The agent's
  `reanalyze_segment_roi` tool (Phase 5) uses this to "look again" at an arm.
- visibility: model score clipped to 0..1, and 0 for points outside the frame.
- quality flags: image-quality flags + person-count flags + framing flags.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from kinevra.schemas import Landmark, PoseFrame
from kinevra.vision.preprocess import Roi, apply_clahe, crop_roi
from kinevra.vision.quality import FrameQuality, framing_flags, person_flags
from kinevra.vision.types import Image

COCO17: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO17_EDGES: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
)

Keypoints = NDArray[np.float32]  # (17, 3): x_px, y_px, score — COCO17 order


@dataclass(frozen=True, slots=True)
class PoseResult:
    keypoints: Keypoints | None  # None when no person was found
    person_count: int
    person_box: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1 in px


class PoseEstimator(Protocol):
    name: str
    last_latency_ms: float

    def estimate(
        self,
        image: Image,
        t: float,
        frame_idx: int,
        *,
        session_id: str = "",
        roi: Roi | None = None,
        roi_scale: float | None = None,
        quality: FrameQuality | None = None,
    ) -> PoseFrame: ...


class DnnPoseEstimator(ABC):
    name = "base"

    def __init__(
        self,
        *,
        roi_scale: float = 2.0,
        roi_clahe: bool = True,
        required_landmarks: Sequence[str] = (),
        visibility_threshold: float = 0.5,
    ) -> None:
        self.roi_scale = roi_scale
        self.roi_clahe = roi_clahe
        self.required_landmarks = tuple(required_landmarks)
        self.visibility_threshold = visibility_threshold
        self.last_latency_ms = 0.0

    @abstractmethod
    def infer(self, image: Image) -> PoseResult:
        """Run the model on a full image (detect person → landmarks), pixel coordinates."""

    def infer_roi(self, crop: Image) -> PoseResult:
        """Run the model on an ROI crop. Override when the model can skip detection."""
        return self.infer(crop)

    def estimate(
        self,
        image: Image,
        t: float,
        frame_idx: int,
        *,
        session_id: str = "",
        roi: Roi | None = None,
        roi_scale: float | None = None,
        quality: FrameQuality | None = None,
    ) -> PoseFrame:
        start = time.perf_counter()
        h, w = image.shape[:2]
        if roi is None:
            result = self.infer(image)
            src_w, src_h = w, h
        else:
            crop = crop_roi(image, roi, roi_scale or self.roi_scale)
            if self.roi_clahe:
                crop = apply_clahe(crop)
            result = self.infer_roi(crop)
            src_h, src_w = crop.shape[:2]
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0

        landmarks: dict[str, Landmark] = {}
        if result.keypoints is not None:
            for name, (x, y, score) in zip(COCO17, result.keypoints.tolist(), strict=True):
                xn, yn = x / src_w, y / src_h
                if roi is not None:
                    xn, yn = roi.to_full_norm(xn, yn)
                inside = 0.0 <= xn <= 1.0 and 0.0 <= yn <= 1.0
                vis = float(np.clip(score, 0.0, 1.0)) if inside else 0.0
                landmarks[name] = Landmark(name=name, x=xn, y=yn, visibility=vis)

        flags: list[str] = list(quality.flags) if quality is not None else []
        if roi is None:  # person count is only meaningful on the full frame
            flags += person_flags(result.person_count)
            person_count = result.person_count
        else:
            person_count = 1 if landmarks else 0
        if landmarks:
            flags += framing_flags(landmarks, self.required_landmarks, self.visibility_threshold)

        return PoseFrame(
            session_id=session_id,
            frame_idx=frame_idx,
            t=t,
            image_width=w,
            image_height=h,
            landmarks=landmarks,
            person_count=person_count,
            frame_quality=quality.score if quality is not None else 1.0,
            quality_flags=list(dict.fromkeys(flags)),  # de-duplicate, keep order
        )


def required_for_side(side: str, joints: Sequence[str]) -> list[str]:
    """E.g. ("right", ["hip", "shoulder", "elbow", "wrist"]) → ["right_hip", ...]."""
    return [f"{side}_{j}" for j in joints]
