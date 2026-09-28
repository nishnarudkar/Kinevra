"""Feature extraction tests (written before the implementation)."""

import math

import pytest

from kinevra.config import ExerciseConfig, load_config
from kinevra.movement.features import FeatureExtractor, extract_features, raw_features
from kinevra.schemas import PoseFrame
from tests.conftest import make_pose

FPS = 15.0


@pytest.fixture
def cfg() -> ExerciseConfig:
    return load_config(env={}).exercise


# --- per-frame measurements -------------------------------------------------------------------


@pytest.mark.parametrize("angle", [0.0, 20.0, 45.0, 90.0, 135.0, 170.0])
@pytest.mark.parametrize(("w", "h"), [(640, 480), (1280, 720), (480, 640)])
def test_abduction_exact_on_any_aspect_ratio(angle: float, w: int, h: int) -> None:
    raw = raw_features(make_pose(angle, width=w, height=h), "right", 0.5)
    assert raw.abduction == pytest.approx(angle, abs=1e-6)


def test_elbow_flexion_and_lean(cfg: ExerciseConfig) -> None:
    raw = raw_features(make_pose(80, elbow_flexion_deg=35, lean_deg=12), "right", 0.5)
    assert raw.abduction == pytest.approx(80, abs=1e-6)
    assert raw.elbow_flexion == pytest.approx(35, abs=1e-6)
    assert raw.trunk_lean == pytest.approx(12, abs=1e-6)
    straight = raw_features(make_pose(80), "right", 0.5)
    assert straight.elbow_flexion == pytest.approx(0, abs=1e-6)
    assert straight.trunk_lean == pytest.approx(0, abs=1e-6)


def test_left_side_uses_left_landmarks() -> None:
    raw = raw_features(make_pose(60, side="left", lean_deg=-8), "left", 0.5)
    assert raw.abduction == pytest.approx(60, abs=1e-6)
    assert raw.trunk_lean == pytest.approx(-8, abs=1e-6)  # leaning toward the left arm
    # measuring the wrong side finds no left/right arm landmarks for "right"
    assert raw_features(make_pose(60, side="left"), "right", 0.5).abduction is None


def test_low_visibility_gives_none_and_low_confidence() -> None:
    raw = raw_features(make_pose(90, hidden=("right_elbow",)), "right", 0.5)
    assert raw.abduction is None and raw.elbow_flexion is None
    assert raw.confidence == pytest.approx(0.1)
    raw = raw_features(make_pose(90, hidden=("left_hip",)), "right", 0.5)
    assert raw.abduction == pytest.approx(90, abs=1e-6)  # primary angle unaffected
    assert raw.trunk_lean is None  # needs both hips
    assert raw.confidence == pytest.approx(0.95)


def test_empty_pose() -> None:
    p = make_pose(0).model_copy(update={"landmarks": {}})
    raw = raw_features(p, "right", 0.5)
    assert raw.abduction is None and raw.confidence == 0.0


# --- helpers for sequences --------------------------------------------------------------------


def _raise_sequence(n: int, *, dps: float = 60.0, start: float = 10.0) -> list[PoseFrame]:
    """Constant-speed raise at `dps` degrees per second, sampled at FPS."""
    return [make_pose(start + dps * i / FPS, t=i / FPS, frame_idx=i) for i in range(n)]


# --- live extractor ---------------------------------------------------------------------------


def test_live_extractor_tracks_angle_and_velocity(cfg: ExerciseConfig) -> None:
    fx = FeatureExtractor(cfg)
    feats = [fx.update(p) for p in _raise_sequence(40)]
    last = feats[-1]
    assert last.side == "right" and last.frame_idx == 39
    true_last = 10 + 60 * 39 / FPS
    assert last.shoulder_abduction_deg == pytest.approx(true_last, abs=6)  # small filter lag
    assert last.angular_velocity_dps == pytest.approx(60, rel=0.15)
    assert last.confidence == pytest.approx(0.95)
    assert feats[0].angular_velocity_dps == 0.0


def test_live_extractor_none_frames_and_gap_reset(cfg: ExerciseConfig) -> None:
    fx = FeatureExtractor(cfg)
    for p in _raise_sequence(10):
        fx.update(p)
    hidden = make_pose(90, t=11 / FPS, frame_idx=11, hidden=("right_elbow",))
    f = fx.update(hidden)
    assert f.shoulder_abduction_deg is None and f.angular_velocity_dps is None
    # after a gap longer than max_gap_s the filter restarts from the new value (no lag)
    after = fx.update(make_pose(150, t=3.0, frame_idx=45))
    assert after.shoulder_abduction_deg == pytest.approx(150, abs=1e-6)


def test_shoulder_elevation_needs_rest_baseline(cfg: ExerciseConfig) -> None:
    fx = FeatureExtractor(cfg)
    rest = [fx.update(make_pose(5, t=i / FPS, frame_idx=i)) for i in range(20)]  # 1.33 s rest
    assert rest[0].shoulder_elevation is None  # baseline not yet established
    assert rest[-1].shoulder_elevation == pytest.approx(0.0, abs=1e-6)
    shrug = fx.update(make_pose(90, t=2.0, frame_idx=30, shoulder_raise_px=26))
    height = 260.0  # builder torso length; shoulder height above hip at rest
    assert shrug.shoulder_elevation == pytest.approx(26 / height, rel=0.1)


# --- offline (clip mode) ----------------------------------------------------------------------


def test_offline_extract_matches_truth_and_velocity(cfg: ExerciseConfig) -> None:
    poses = _raise_sequence(30)
    feats = extract_features(poses, cfg)
    assert len(feats) == 30
    for i, f in enumerate(feats):
        assert f.shoulder_abduction_deg == pytest.approx(10 + 60 * i / FPS, abs=1e-6)
        assert f.angular_velocity_dps == pytest.approx(60, abs=1e-6)


def test_offline_keeps_missing_frames_missing(cfg: ExerciseConfig) -> None:
    poses = _raise_sequence(30)
    poses[12] = make_pose(0, t=12 / FPS, frame_idx=12, hidden=("right_elbow",))
    feats = extract_features(poses, cfg)
    assert feats[12].shoulder_abduction_deg is None
    assert feats[12].confidence == pytest.approx(0.1)
    assert feats[13].shoulder_abduction_deg == pytest.approx(10 + 60 * 13 / FPS, abs=1e-6)


def test_offline_smooths_noise(cfg: ExerciseConfig) -> None:
    import random

    rng = random.Random(0)
    poses, truth = [], []
    for i in range(60):
        a = 80 + 60 * math.sin(2 * math.pi * 0.25 * i / FPS)
        truth.append(a)
        poses.append(make_pose(a + rng.gauss(0, 3), t=i / FPS, frame_idx=i))
    feats = extract_features(poses, cfg)
    raw = [raw_features(p, "right", 0.5).abduction for p in poses]
    err_raw = sum(abs(r - t) for r, t in zip(raw, truth, strict=True) if r is not None)
    err_smooth = sum(
        abs(f.shoulder_abduction_deg - t)
        for f, t in zip(feats, truth, strict=True)
        if f.shoulder_abduction_deg is not None
    )
    assert err_smooth < 0.8 * err_raw


def test_smooth_live_matches_extractor(cfg: ExerciseConfig) -> None:
    from kinevra.movement.features import smooth_live

    poses = _raise_sequence(20)
    live = [FeatureExtractor(cfg)]
    feats = [live[0].update(p) for p in poses]
    raw = [raw_features(p, "right", 0.5).abduction for p in poses]
    smoothed, vel = smooth_live([p.t for p in poses], raw, cfg)
    assert smoothed == [f.shoulder_abduction_deg for f in feats]
    assert vel == [f.angular_velocity_dps for f in feats]
    assert smooth_live([0.0, 0.1], [None, 5.0], cfg) == ([None, 5.0], [None, 0.0])
