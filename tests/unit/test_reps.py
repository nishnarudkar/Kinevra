"""Repetition state machine tests (written before the implementation)."""

import pytest

from kinevra.config import ExerciseConfig, load_config
from kinevra.movement.reps import RepCounter, RepEvent, RepMeasurement, RepPhase
from tests.fixtures.sequences import RepSpec, Session, build_session


@pytest.fixture
def cfg() -> ExerciseConfig:
    return load_config(env={}).exercise


def run(session: Session, cfg: ExerciseConfig) -> tuple[list[RepMeasurement], RepCounter]:
    counter = RepCounter(cfg.reps, cfg.side, cfg.smoothing.max_gap_s)
    reps = []
    for feat, flags in zip(session.features, session.flags, strict=True):
        out = counter.update(feat, flags)
        if isinstance(out, RepMeasurement):
            reps.append(out)
    return reps, counter


def test_clean_reps_counted_with_metrics(cfg: ExerciseConfig) -> None:
    reps, counter = run(build_session([RepSpec()] * 5), cfg)
    assert len(reps) == 5 and counter.count == 5
    assert [r.rep_index for r in reps] == [1, 2, 3, 4, 5]
    for r in reps:
        assert r.min_angle == pytest.approx(15, abs=0.5)
        assert r.max_angle == pytest.approx(150, abs=0.5)
        assert r.rom == pytest.approx(135, abs=1)
        assert 2.0 <= r.duration_s <= 2.6  # 2.4 s movement, edges cut by the rest hysteresis
        assert r.t_start < r.t_peak < r.t_end
        assert r.peak_velocity_dps > r.mean_velocity_dps > 0
        assert r.smoothness > 0.95
        assert r.confidence == pytest.approx(0.95)
        assert r.side == "right"
    assert counter.events == []
    assert counter.phase is RepPhase.REST


def test_noisy_reps_no_double_counting(cfg: ExerciseConfig) -> None:
    reps, _ = run(build_session([RepSpec()] * 6, noise=2.0, seed=4), cfg)
    assert len(reps) == 6
    assert all(r.rom == pytest.approx(135, abs=6) for r in reps)
    noisy_smoothness = min(r.smoothness for r in reps)
    clean, _ = run(build_session([RepSpec()] * 6), cfg)
    assert noisy_smoothness < min(r.smoothness for r in clean)


def test_partial_raises_are_logged_not_counted(cfg: ExerciseConfig) -> None:
    spec = [RepSpec(), RepSpec(peak=45), RepSpec(), RepSpec(peak=50), RepSpec()]
    reps, counter = run(build_session(spec), cfg)
    assert len(reps) == 3 and counter.count == 3
    partial = [e for e in counter.events if e.kind == "partial_rep"]
    assert len(partial) == 2
    assert partial[0].max_angle == pytest.approx(45, abs=0.5)
    assert partial[0].rom < cfg.reps.min_rom


def test_tiny_wobble_is_not_even_a_partial(cfg: ExerciseConfig) -> None:
    reps, counter = run(build_session([RepSpec(peak=25)]), cfg)  # below raise_margin
    assert reps == [] and counter.events == []


def test_pause_at_top_counts_once(cfg: ExerciseConfig) -> None:
    reps, _ = run(build_session([RepSpec(hold_s=2.5)] * 3, noise=1.5, seed=1), cfg)
    assert len(reps) == 3
    assert all(r.duration_s > 4.0 for r in reps)


def test_too_fast_movement_not_counted(cfg: ExerciseConfig) -> None:
    reps, counter = run(build_session([RepSpec(rise_s=0.25, lower_s=0.25)]), cfg)
    assert reps == []
    assert [e.kind for e in counter.events] == ["too_short"]


def test_incomplete_return_then_next_raise_counts_both(cfg: ExerciseConfig) -> None:
    spec = [RepSpec(end=60.0), RepSpec(start=60.0)]
    reps, counter = run(build_session(spec), cfg)
    assert len(reps) == 2
    assert reps[0].t_end <= reps[1].t_start
    assert any(e.kind == "incomplete_return" for e in counter.events)


def test_slow_rep_is_counted_rules_judge_duration(cfg: ExerciseConfig) -> None:
    reps, _ = run(build_session([RepSpec(rise_s=6, lower_s=6)]), cfg)
    assert len(reps) == 1 and reps[0].duration_s > cfg.reps.max_duration_s


def test_dropout_mid_rep_lowers_confidence(cfg: ExerciseConfig) -> None:
    reps, _ = run(build_session([RepSpec(dropout=(0.4, 0.6))]), cfg)
    assert len(reps) == 1
    assert reps[0].confidence < 0.95 * 0.9


def test_posture_metrics_and_flags_collected(cfg: ExerciseConfig) -> None:
    spec = [RepSpec(lean=-14, elbow=38, elevation=0.12, flags=("low_light",))]
    reps, _ = run(build_session(spec), cfg)
    r = reps[0]
    assert r.max_trunk_lean == pytest.approx(-14, abs=0.5)  # signed, largest magnitude
    assert r.max_elbow_flexion == pytest.approx(38, abs=0.5)
    assert r.max_shoulder_elevation == pytest.approx(0.12, abs=0.01)
    assert r.flag_fractions["low_light"] == pytest.approx(1.0)


def test_phases_progress(cfg: ExerciseConfig) -> None:
    session = build_session([RepSpec(hold_s=1.0)])
    counter = RepCounter(cfg.reps, cfg.side, cfg.smoothing.max_gap_s)
    seen: list[RepPhase] = []
    for feat, flags in zip(session.features, session.flags, strict=True):
        counter.update(feat, flags)
        if not seen or seen[-1] is not counter.phase:
            seen.append(counter.phase)
    assert seen == [
        RepPhase.REST,
        RepPhase.RAISING,
        RepPhase.PEAK,
        RepPhase.LOWERING,
        RepPhase.REST,
    ]


def test_events_are_rep_events(cfg: ExerciseConfig) -> None:
    _, counter = run(build_session([RepSpec(peak=45)]), cfg)
    assert all(isinstance(e, RepEvent) for e in counter.events)
