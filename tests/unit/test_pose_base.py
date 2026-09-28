"""DnnPoseEstimator.estimate(): normalisation, ROI mapping, visibility and flags (fake model)."""

import numpy as np
import pytest

from kinevra.pose.base import COCO17, DnnPoseEstimator, PoseResult, required_for_side
from kinevra.vision.preprocess import Roi
from kinevra.vision.quality import FrameQuality
from kinevra.vision.types import Image

REQUIRED = required_for_side("right", ["hip", "shoulder", "elbow", "wrist"])


class FakeEstimator(DnnPoseEstimator):
    """Puts every keypoint at a fixed fraction of whatever image it is given."""

    name = "fake"

    def __init__(
        self,
        fx: float = 0.5,
        fy: float = 0.5,
        score: float = 0.9,
        persons: int = 1,
        overrides: dict[str, tuple[float, float, float]] | None = None,
    ) -> None:
        super().__init__(required_landmarks=REQUIRED, visibility_threshold=0.5, roi_clahe=False)
        self.fx, self.fy, self.score, self.persons = fx, fy, score, persons
        self.overrides = overrides or {}
        self.seen_shapes: list[tuple[int, ...]] = []

    def infer(self, image: Image) -> PoseResult:
        self.seen_shapes.append(image.shape)
        if self.persons == 0:
            return PoseResult(None, 0)
        h, w = image.shape[:2]
        kps = np.tile(np.array([self.fx * w, self.fy * h, self.score], np.float32), (17, 1))
        for name, (x, y, s) in self.overrides.items():
            kps[COCO17.index(name)] = [x * w, y * h, s]
        return PoseResult(kps, self.persons)


IMG = np.zeros((480, 640, 3), dtype=np.uint8)


def test_landmarks_normalised_and_named() -> None:
    pf = FakeEstimator(0.25, 0.75).estimate(IMG, 1.5, 7, session_id="s")
    assert set(pf.landmarks) == set(COCO17)
    lm = pf.landmarks["right_elbow"]
    assert (lm.x, lm.y, lm.visibility) == pytest.approx((0.25, 0.75, 0.9))
    assert (pf.session_id, pf.frame_idx, pf.t, pf.person_count) == ("s", 7, 1.5, 1)
    assert pf.quality_flags == [] and pf.frame_quality == 1.0


def test_scores_clipped_and_outside_points_invisible() -> None:
    est = FakeEstimator(score=1.7, overrides={"right_wrist": (1.2, 0.5, 0.95)})
    pf = est.estimate(IMG, 0, 0)
    assert pf.landmarks["nose"].visibility == 1.0
    assert pf.landmarks["right_wrist"].visibility == 0.0
    assert "out_of_frame" in pf.quality_flags


def test_low_visibility_required_landmark_flags_out_of_frame() -> None:
    est = FakeEstimator(overrides={"right_hip": (0.5, 0.9, 0.2)})
    assert est.estimate(IMG, 0, 0).quality_flags == ["out_of_frame"]


def test_person_flags_and_quality_merge() -> None:
    q = FrameQuality(20, 5, 3, 0.1, ["low_light", "blurry"])
    pf = FakeEstimator(persons=0).estimate(IMG, 0, 0, quality=q)
    assert pf.landmarks == {} and pf.person_count == 0
    assert pf.quality_flags == ["low_light", "blurry", "no_person"]
    assert pf.frame_quality == 0.1
    pf = FakeEstimator(persons=2).estimate(IMG, 0, 0, quality=q)
    assert pf.quality_flags == ["low_light", "blurry", "multiple_people"]


def test_roi_mode_crops_upsamples_and_maps_back() -> None:
    est = FakeEstimator(0.5, 0.5)
    roi = Roi(x=100, y=40, w=200, h=100, frame_w=640, frame_h=480)
    pf = est.estimate(IMG, 0, 0, roi=roi, roi_scale=2.0)
    assert est.seen_shapes[-1][:2] == (200, 400)  # crop upsampled 2x
    lm = pf.landmarks["right_shoulder"]
    assert (lm.x, lm.y) == pytest.approx((200 / 640, 90 / 480))  # centre of the ROI
    assert pf.person_count == 1
    assert "multiple_people" not in pf.quality_flags  # person count not judged in ROI mode


def test_roi_mode_uses_default_scale_and_records_latency() -> None:
    est = FakeEstimator()
    est.roi_scale = 3.0
    est.estimate(IMG, 0, 0, roi=Roi(0, 0, 10, 20, 640, 480))
    assert est.seen_shapes[-1][:2] == (60, 30)
    assert est.last_latency_ms >= 0.0


def test_required_for_side() -> None:
    assert required_for_side("left", ["hip", "wrist"]) == ["left_hip", "left_wrist"]
