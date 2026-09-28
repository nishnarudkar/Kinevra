"""Agent tools on fixture frames (OpenCV tools run for real; pose model faked where needed)."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from kinevra.agent.context import BufferFrameProvider, PoseHistory, ToolContext
from kinevra.agent.tools import TOOLS, execute_tool, tool_schemas
from kinevra.config import AppConfig, load_config
from kinevra.movement.session import MovementSession
from kinevra.schemas import Landmark, PoseFrame, RepMetrics, RepQuality
from kinevra.storage.local import LocalStore
from kinevra.vision.buffer import FrameBuffer
from kinevra.vision.preprocess import Roi
from kinevra.vision.types import Frame
from tests.conftest import make_pose

FPS = 15.0
W, H = 640, 480


def _rep(**kw: Any) -> RepMetrics:
    base: dict[str, Any] = dict(
        rep_index=1,
        side="right",
        t_start=1.0,
        t_peak=2.2,
        t_end=3.4,
        min_angle=15,
        max_angle=115,
        rom=100,
        duration_s=2.4,
        mean_velocity_dps=90,
        peak_velocity_dps=150,
        smoothness=0.95,
        max_elbow_flexion=5,
        max_trunk_lean=1,
        confidence=0.55,
        rule_quality=RepQuality.UNCERTAIN,
        rule_reasons=["low_tracking_confidence: 0.55 < 0.6", "possible_reduced_rom: ..."],
    )
    base.update(kw)
    return RepMetrics(**base)


def _ctx(
    tmp_path: Path,
    frames: list[Frame],
    poses: list[PoseFrame],
    rep: RepMetrics,
    estimator: Any = None,
    baseline: float | None = 135.0,
) -> ToolContext:
    cfg: AppConfig = load_config(env={})
    session = MovementSession(cfg.exercise, "sess1", "clip")
    session.state.reps.append(rep)
    session.state.baseline_rom = baseline
    buf = FrameBuffer(seconds=60.0)
    for f in frames:
        buf.push(f)
    history = PoseHistory()
    history.extend(poses)
    return ToolContext(
        cfg=cfg,
        session=session,
        frames=BufferFrameProvider(buf),
        poses=history,
        store=LocalStore(tmp_path),
        estimator=estimator,
    )


def _blank_frames(n: int, value: int = 120) -> list[Frame]:
    return [Frame(i, i / FPS, np.full((H, W, 3), value, np.uint8)) for i in range(n)]


def _truth_angles(n: int, peak: float) -> list[float]:
    start, end = 15, 51  # rep occupies frames 15..51 (t 1.0 .. 3.4)
    out = []
    for i in range(n):
        u = min(max((i - start) / (end - start), 0.0), 1.0)
        out.append(15 + (peak - 15) * np.sin(np.pi * u) ** 2)
    return out


class RoiOnlyEstimator:
    """Fake pose model: poor confidence on the full frame, confident + exact in ROI mode."""

    name = "fake"
    last_latency_ms = 1.0

    def __init__(self, truth: list[float]) -> None:
        self.truth = truth
        self.rois: list[Roi | None] = []

    def estimate(
        self,
        image: np.ndarray,
        t: float,
        frame_idx: int,
        *,
        session_id: str = "",
        roi: Roi | None = None,
        roi_scale: float | None = None,
        quality: Any = None,
    ) -> PoseFrame:
        self.rois.append(roi)
        h, w = image.shape[:2]
        vis = 0.9 if roi is not None else 0.5
        return make_pose(
            self.truth[frame_idx], t=t, frame_idx=frame_idx, width=w, height=h, visibility=vis
        )


# --- reanalyze_segment_roi ----------------------------------------------------------------


@pytest.mark.parametrize(("peak", "quality"), [(150.0, "GOOD"), (100.0, "DEVIATION")])
def test_reanalyze_rejudges_rep_from_roi_measurement(
    tmp_path: Path, peak: float, quality: str
) -> None:
    truth = _truth_angles(66, peak)
    frames = _blank_frames(66)
    poses = [
        make_pose(a, t=i / FPS, frame_idx=i, width=W, height=H, visibility=0.55)
        for i, a in enumerate(truth)
    ]
    est = RoiOnlyEstimator(truth)
    ctx = _ctx(tmp_path, frames, poses, _rep(), estimator=est)
    out, ms = execute_tool(ctx, "reanalyze_segment_roi", {"rep_index": 1})
    assert out["ok"], out
    assert all(r is not None for r in est.rois)  # every call used ROI mode
    assert out["quality_before"] == "UNCERTAIN" and out["quality_after"] == quality
    assert out["assessment"] == quality
    assert out["rom_after"] == pytest.approx(peak - 15, abs=3)
    assert out["confidence_after"] == pytest.approx(0.9)
    assert out["frames_analyzed"] <= ctx.cfg.exercise.tools.reanalyze_max_frames
    if quality == "DEVIATION":
        assert out["reasons_after"][0].startswith("reduced_rom")
    assert ms >= 0


def test_reanalyze_without_frames_or_estimator(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, [], [], _rep(), estimator=RoiOnlyEstimator([0.0]))
    out, _ = execute_tool(ctx, "reanalyze_segment_roi", {"rep_index": 1})
    assert not out["ok"] and "no longer available" in out["error"]
    ctx.estimator = None
    assert (
        "no pose estimator"
        in execute_tool(ctx, "reanalyze_segment_roi", {"rep_index": 1})[0]["error"]
    )
    assert "not found" in execute_tool(ctx, "reanalyze_segment_roi", {"rep_index": 9})[0]["error"]


# --- verify_motion_optical_flow -----------------------------------------------------------


def _arm_pose(i: int, elbow: tuple[float, float]) -> PoseFrame:
    ex, ey = elbow
    lms = {
        "right_shoulder": (ex, ey - 80),
        "right_elbow": (ex, ey),
        "right_wrist": (ex, ey + 70),
        "right_hip": (ex + 40, ey + 150),
        "left_hip": (ex + 160, ey + 150),
        "left_shoulder": (ex + 160, ey - 80),
    }
    return PoseFrame(
        session_id="s",
        frame_idx=i,
        t=i / FPS,
        image_width=W,
        image_height=H,
        landmarks={
            n: Landmark(name=n, x=x / W, y=y / H, visibility=0.9) for n, (x, y) in lms.items()
        },
        person_count=1,
        frame_quality=1.0,
        quality_flags=[],
    )


def _textured_frames(n: int, block_y: list[int] | list[float]) -> list[Frame]:
    rng = np.random.default_rng(0)
    noise = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 0.8)
    gradient = np.tile(np.linspace(0, 255, W, dtype=np.float32), (H, 1))[..., None]
    background = (0.5 * noise + 0.5 * gradient).astype(np.uint8)  # room-like contrast
    block = cv2.GaussianBlur(rng.integers(0, 255, (60, 60, 3), dtype=np.uint8), (0, 0), 1.5)
    frames = []
    for i in range(n):
        img = background.copy()
        y = int(block_y[i])
        img[y - 30 : y + 30, 290:350] = block
        frames.append(Frame(i, i / FPS, img))
    return frames


def test_flow_confirms_real_motion(tmp_path: Path) -> None:
    ys = [200 + 3 * i for i in range(36)]  # block (the "arm") moves 3 px/frame
    frames = _textured_frames(36, ys)
    poses = [_arm_pose(i, (320, y)) for i, y in enumerate(ys)]
    ctx = _ctx(tmp_path, frames, poses, _rep(t_start=0.0, t_end=35 / FPS))
    out, _ = execute_tool(ctx, "verify_motion_optical_flow", {"rep_index": 1})
    assert out["ok"], out
    assert out["verdict"] == "real_motion" and out["assessment"] is None
    assert out["jitter_ratio"] == pytest.approx(1.0, abs=0.35)
    assert out["direction_agreement"] > 0.9


def test_flow_detects_landmark_jitter(tmp_path: Path) -> None:
    frames = _textured_frames(36, [260] * 36)  # nothing moves in the image
    rng = np.random.default_rng(1)
    poses = [_arm_pose(i, (320 + rng.normal(0, 6), 260 + rng.normal(0, 6))) for i in range(36)]
    ctx = _ctx(tmp_path, frames, poses, _rep(t_start=0.0, t_end=35 / FPS))
    out, _ = execute_tool(ctx, "verify_motion_optical_flow", {"rep_index": 1})
    assert out["ok"], out
    assert out["verdict"] == "landmark_jitter" and out["assessment"] == "tracking_noise"
    assert out["landmark_path_px"] > 5 * out["flow_path_px"]


# --- check_camera_setup -------------------------------------------------------------------


def test_camera_dark_frames_give_light_instruction(tmp_path: Path) -> None:
    frames = _blank_frames(45, value=10)
    poses = [make_pose(20, t=i / FPS, frame_idx=i, width=W, height=H) for i in range(45)]
    out, _ = execute_tool(_ctx(tmp_path, frames, poses, _rep()), "check_camera_setup", {})
    assert out["ok"] and "low_light" in out["issues"]
    assert out["assessment"] == "camera_issue" and "light" in out["instruction"]


def test_camera_no_person_takes_priority_and_good_setup_passes(tmp_path: Path) -> None:
    frames = _textured_frames(45, [260] * 45)
    empty = [
        p.model_copy(update={"landmarks": {}, "person_count": 0, "quality_flags": ["no_person"]})
        for p in (make_pose(20, t=i / FPS, frame_idx=i) for i in range(45))
    ]
    out, _ = execute_tool(
        _ctx(tmp_path, frames, empty, _rep()), "check_camera_setup", {"window_s": 2}
    )
    assert out["issues"][0] == "no_person" and out["instruction"].startswith("Step into")

    good = [make_pose(20, t=i / FPS, frame_idx=i) for i in range(45)]  # torso 36% of height
    out, _ = execute_tool(_ctx(tmp_path, frames, good, _rep()), "check_camera_setup", {})
    assert out["issues"] == [] and out["assessment"] == "camera_ok"


def test_camera_subject_too_far(tmp_path: Path) -> None:
    frames = _textured_frames(20, [260] * 20)
    # 1280x720 builder geometry shown in a 4x taller frame → tiny torso fraction
    far = [make_pose(20, t=i / FPS, frame_idx=i, width=1280, height=2880) for i in range(20)]
    out, _ = execute_tool(_ctx(tmp_path, frames, far, _rep()), "check_camera_setup", {})
    assert "too_far" in out["issues"]


# --- render_evidence_snapshot -------------------------------------------------------------


def test_snapshot_writes_three_annotated_jpegs(tmp_path: Path) -> None:
    truth = _truth_angles(66, 150.0)
    frames = _textured_frames(66, [260] * 66)
    poses = [make_pose(a, t=i / FPS, frame_idx=i, width=W, height=H) for i, a in enumerate(truth)]
    ctx = _ctx(tmp_path, frames, poses, _rep())
    out, _ = execute_tool(ctx, "render_evidence_snapshot", {"rep_index": 1})
    assert out["ok"] and out["snapshot_keys"] == [
        "sess1/rep0001_start.jpg",
        "sess1/rep0001_peak.jpg",
        "sess1/rep0001_end.jpg",
    ]
    for key in out["snapshot_keys"]:
        img = cv2.imdecode(np.frombuffer(ctx.store.get_bytes(key), np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape == (H, W, 3)
    assert ctx.snapshots[1] == out["snapshot_keys"]


# --- analysis tools, registry -------------------------------------------------------------


def test_analysis_tools_and_review(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        [],
        [],
        _rep(rule_quality=RepQuality.DEVIATION, rom=100, rule_reasons=["reduced_rom: x"]),
    )
    assert execute_tool(ctx, "analyze_recent_repetitions", {"k": 2})[0]["reps"][0]["rom"] == 100
    base = execute_tool(ctx, "compare_with_session_baseline", {})[0]
    assert base["latest_to_baseline"] == pytest.approx(100 / 135, abs=1e-3)
    assert (
        execute_tool(ctx, "calculate_movement_metrics", {"rep_index": 1})[0]["quality"]
        == "DEVIATION"
    )
    assert "consistency" in execute_tool(ctx, "check_movement_consistency", {})[0]
    fb = execute_tool(ctx, "generate_feedback", {"category": "trunk_lean"})[0]
    assert fb["text"].startswith("Keep your body upright")
    assert not execute_tool(ctx, "generate_feedback", {"category": "diagnosis"})[0]["ok"]
    execute_tool(ctx, "log_event", {"type": "uncertain_rep", "payload": {"rep_index": 1}})
    assert ctx.logged_events[0]["type"] == "uncertain_rep"
    ctx.snapshots[1] = ["sess1/rep0001_peak.jpg"]
    review = execute_tool(
        ctx,
        "request_human_review",
        {"reason": "3 reps below 80% of baseline", "evidence": {"rep_index": 1}},
    )
    assert review[0]["snapshot_keys"] == ["sess1/rep0001_peak.jpg"]
    assert ctx.review_events[0].status == "PENDING"


def test_registry_errors_and_stubs(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, [], [], _rep())
    assert "unknown tool" in execute_tool(ctx, "diagnose", {})[0]["error"]
    assert (
        "invalid arguments"
        in execute_tool(ctx, "calculate_movement_metrics", {"rep_index": "x"})[0]["error"]
    )
    out, _ = execute_tool(
        ctx,
        "reanalyze_segment_roi",
        {"rep_index": 1},
        stubs={"reanalyze_segment_roi": {"quality_after": "GOOD", "assessment": "GOOD"}},
    )
    assert out == {"quality_after": "GOOD", "assessment": "GOOD", "ok": True}
    schemas = tool_schemas()
    assert len(schemas) == len(TOOLS) == 11
    assert schemas[0]["input_schema"]["properties"]["rep_index"]["type"] == "integer"
