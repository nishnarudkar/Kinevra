"""Session state tests (written before the implementation)."""

import pytest

from kinevra.config import ExerciseConfig, load_config
from kinevra.movement.session import MovementSession, SessionState
from kinevra.schemas import FrameFeatures, RepMetrics, RepQuality
from tests.fixtures.sequences import RepSpec, build_session


@pytest.fixture
def cfg() -> ExerciseConfig:
    return load_config(env={}).exercise


def rep(i: int, rom: float, quality: RepQuality = RepQuality.GOOD) -> RepMetrics:
    return RepMetrics(
        rep_index=i,
        side="right",
        t_start=i * 3.0,
        t_peak=i * 3.0 + 1,
        t_end=i * 3.0 + 2.4,
        min_angle=15,
        max_angle=15 + rom,
        rom=rom,
        duration_s=2.4,
        mean_velocity_dps=100,
        peak_velocity_dps=180,
        smoothness=0.97,
        max_elbow_flexion=5,
        max_trunk_lean=2,
        confidence=0.9,
        rule_quality=quality,
        rule_reasons=[],
    )


def state_with(cfg: ExerciseConfig, reps: list[RepMetrics]) -> SessionState:
    s = SessionState(cfg, "s1", "clip")
    for r in reps:
        s.add_rep(r)
    return s


def test_baseline_is_median_of_first_good_reps_only(cfg: ExerciseConfig) -> None:
    s = state_with(cfg, [rep(1, 90, RepQuality.DEVIATION), rep(2, 130), rep(3, 140)])
    assert s.baseline_rom is None  # only 2 GOOD reps so far
    s.add_rep(rep(4, 120, RepQuality.UNCERTAIN))
    s.add_rep(rep(5, 150))
    assert s.baseline_rom == 140  # median of 130, 140, 150
    s.add_rep(rep(6, 60))
    assert s.baseline_rom == 140  # fixed once established


@pytest.mark.parametrize(
    ("roms", "trend"),
    [
        ([140, 138], "insufficient_data"),
        ([140, 139, 141, 140, 139], "stable"),
        ([140, 132, 125, 118, 110], "decreasing"),
        ([100, 110, 118, 126, 135], "increasing"),
    ],
)
def test_rom_trend(cfg: ExerciseConfig, roms: list[float], trend: str) -> None:
    s = state_with(cfg, [rep(i + 1, r) for i, r in enumerate(roms)])
    assert s.evidence().rom_trend == trend


def test_recent_roms_skip_uncertain_and_keep_last_k(cfg: ExerciseConfig) -> None:
    roms = [150, 149, 148, 147, 146, 145]
    reps = [rep(i + 1, r) for i, r in enumerate(roms)] + [rep(7, 50, RepQuality.UNCERTAIN)]
    ev = state_with(cfg, reps).evidence()
    assert ev.recent_roms == [149, 148, 147, 146, 145]  # recent_k = 5
    assert ev.reps_completed == 7


def test_consistency(cfg: ExerciseConfig) -> None:
    same = state_with(cfg, [rep(i, 140) for i in range(1, 6)]).evidence()
    varied = state_with(cfg, [rep(i, r) for i, r in enumerate([140, 100, 150, 90, 130], 1)])
    assert same.movement_consistency == pytest.approx(1.0)
    assert 0.0 <= varied.evidence().movement_consistency < 0.9


def test_consecutive_deviations(cfg: ExerciseConfig) -> None:
    d, g, u = RepQuality.DEVIATION, RepQuality.GOOD, RepQuality.UNCERTAIN
    s = state_with(cfg, [rep(1, 140, d), rep(2, 140, g), rep(3, 140, d), rep(4, 140, u)])
    assert s.evidence().consecutive_deviations == 1  # UNCERTAIN neither resets nor counts
    s.add_rep(rep(5, 140, d))
    assert s.evidence().consecutive_deviations == 2
    s.add_rep(rep(6, 140, g))
    assert s.evidence().consecutive_deviations == 0


def test_rolling_confidence_and_flags(cfg: ExerciseConfig) -> None:
    s = SessionState(cfg, "s1", "live")
    for i in range(150):  # 10 s at 15 FPS; last 5 s are dim and poorly tracked
        dim = i >= 75
        feat = FrameFeatures(
            frame_idx=i,
            t=i / 15,
            side="right",
            shoulder_abduction_deg=20,
            elbow_flexion_deg=5,
            trunk_lean_deg=0,
            angular_velocity_dps=0,
            confidence=0.4 if dim else 0.95,
        )
        s.observe_frame(feat, ["low_light"] if dim else [])
    ev = s.evidence()
    assert ev.tracking_confidence == pytest.approx(0.4, abs=0.02)
    assert ev.data_quality_flags == ["low_light"]
    assert ev.latest_rep is None and ev.baseline_rom is None


def test_evidence_fields(cfg: ExerciseConfig) -> None:
    s = SessionState(cfg, "abc", "clip", reps_target=10)
    s.add_rep(rep(1, 140))
    ev = s.evidence()
    assert (ev.session_id, ev.mode, ev.exercise, ev.side) == (
        "abc",
        "clip",
        "shoulder_abduction",
        "right",
    )
    assert ev.reps_target == 10 and ev.latest_rep is not None and ev.latest_rep.rep_index == 1
    assert ev.model_validate_json(ev.model_dump_json()) == ev


# --- end to end on synthetic features -----------------------------------------------------


def run(cfg: ExerciseConfig, specs: list[RepSpec]) -> tuple[MovementSession, list[RepMetrics]]:
    session = build_session(specs)
    ms = MovementSession(cfg, "s1", "clip")
    reps = []
    for feat, flags in zip(session.features, session.flags, strict=True):
        out = ms.update(feat, flags)
        if out is not None:
            reps.append(out)
    ms.finish()
    return ms, reps


def test_fatigue_decline_session(cfg: ExerciseConfig) -> None:
    peaks = [150, 151, 149, 150, 130, 115, 110, 105]
    ms, reps = run(cfg, [RepSpec(peak=p) for p in peaks])
    assert len(reps) == 8
    qualities = [r.rule_quality for r in reps]
    assert qualities[:5] == [RepQuality.GOOD] * 5  # 115/135 = 85% is still within 80%
    assert qualities[5:] == [RepQuality.DEVIATION] * 3
    assert all(r.rule_reasons[0].startswith("reduced_rom") for r in reps[5:])
    ev = ms.evidence()
    assert ev.baseline_rom == pytest.approx(135, abs=1)
    assert ev.consecutive_deviations == 3
    assert ev.rom_trend == "decreasing"
    assert ev.reps_completed == 8 and ev.latest_rep == reps[-1]


def test_posture_deviations_and_partials(cfg: ExerciseConfig) -> None:
    specs = [RepSpec(), RepSpec(lean=13), RepSpec(peak=45), RepSpec(elbow=40), RepSpec(peak=80)]
    ms, reps = run(cfg, specs)
    assert len(reps) == 4  # the 45° raise is a partial
    assert [r.rule_reasons[0].split(":")[0] if r.rule_reasons else "ok" for r in reps] == [
        "ok",
        "trunk_lean",
        "elbow_flexion",
        "ok",  # no baseline yet → 80° half range not judged
    ]
    assert [e.kind for e in ms.events] == ["partial_rep"]
