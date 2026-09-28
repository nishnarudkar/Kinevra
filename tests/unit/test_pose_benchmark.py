import numpy as np
import pytest

from kinevra.pose.base import COCO17, DnnPoseEstimator, PoseResult, required_for_side
from kinevra.pose.benchmark import BenchmarkResult, run_benchmark, to_markdown
from kinevra.vision.types import Frame, Image

REQUIRED = required_for_side("right", ["hip", "shoulder", "elbow", "wrist"])


class ScriptedEstimator(DnnPoseEstimator):
    """Full frame: score 0.6 and a small drift; ROI crops: score 0.9."""

    name = "scripted"

    def __init__(self) -> None:
        super().__init__(required_landmarks=REQUIRED, roi_clahe=False)
        self.calls = 0

    def infer(self, image: Image) -> PoseResult:
        self.calls += 1
        h, w = image.shape[:2]
        kps = np.zeros((17, 3), np.float32)
        kps[:, 0], kps[:, 1], kps[:, 2] = 0.5 * w, 0.5 * h, 0.6
        kps[COCO17.index("right_shoulder"), 1] = 0.3 * h
        kps[COCO17.index("right_hip"), 1] = 0.7 * h
        kps[:, 0] += self.calls  # 1 px drift per call
        return PoseResult(kps, 1)

    def infer_roi(self, crop: Image) -> PoseResult:
        res = self.infer(crop)
        assert res.keypoints is not None
        res.keypoints[:, 2] = 0.9
        return res


def _frames(n: int) -> list[Frame]:
    return [Frame(i, i / 10, np.zeros((100, 200, 3), np.uint8)) for i in range(n)]


def test_benchmark_metrics() -> None:
    r = run_benchmark(
        ScriptedEstimator(),
        _frames(10),
        required=REQUIRED,
        visibility_threshold=0.5,
        source="clip.mp4",
    )
    assert r.frames == 10 and r.usable_pct == 100.0
    assert r.required_conf_mean == pytest.approx(0.6)
    assert r.jitter_px_median == pytest.approx(1.0)
    assert r.jitter_torso_pct == pytest.approx(100 * 1 / 40)  # torso = 0.4 * 100 px
    assert r.latency_ms_p95 >= r.latency_ms_p50 >= 0 and r.roi_frames == 0


def test_benchmark_roi_gain() -> None:
    r = run_benchmark(
        ScriptedEstimator(), _frames(5), required=REQUIRED, visibility_threshold=0.5, roi=True
    )
    assert r.roi_frames == 5 and r.roi_usable_pct == 100.0
    assert r.roi_conf_mean == pytest.approx(0.9)
    assert r.roi_conf_gain == pytest.approx(0.3)
    assert r.roi_recovered_pct is None  # full frame never failed


def test_no_roi_without_a_good_seed_frame() -> None:
    r = run_benchmark(
        ScriptedEstimator(), _frames(5), required=REQUIRED, visibility_threshold=0.7, roi=True
    )
    assert r.usable_pct == 0.0 and r.jitter_px_median is None
    assert r.roi_frames == 0


def test_markdown_table() -> None:
    r = BenchmarkResult("m", "a/b.mp4", 1, 1, 1, 1, 1000, 100, 0.9, None, None)
    md = to_markdown([r])
    assert md.splitlines()[0].startswith("| model |") and "| m | b.mp4 |" in md


class LosesPersonEstimator(ScriptedEstimator):
    """Full frame finds the person only on the first frame; ROI always works."""

    def infer(self, image: Image) -> PoseResult:
        if self.calls >= 1:
            self.calls += 1
            return PoseResult(None, 0)
        return super().infer(image)

    def infer_roi(self, crop: Image) -> PoseResult:
        res = ScriptedEstimator.infer(self, crop)
        assert res.keypoints is not None
        res.keypoints[:, 2] = 0.9
        return res


def test_roi_seeded_from_last_good_frame_recovers_lost_person() -> None:
    r = run_benchmark(
        LosesPersonEstimator(), _frames(4), required=REQUIRED, visibility_threshold=0.5, roi=True
    )
    assert r.usable_pct == 25.0
    assert r.roi_frames == 4 and r.roi_usable_pct == 100.0
    assert r.roi_recovered_pct == 100.0
