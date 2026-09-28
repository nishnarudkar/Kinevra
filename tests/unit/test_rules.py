"""Rule baseline tests (written before the implementation)."""

from dataclasses import replace
from typing import Any

import pytest

from kinevra.config import ExerciseConfig, load_config
from kinevra.evaluation.rules import evaluate_rep
from kinevra.movement.reps import RepMeasurement
from kinevra.schemas import RepQuality


@pytest.fixture
def cfg() -> ExerciseConfig:
    return load_config(env={}).exercise


GOOD = RepMeasurement(
    rep_index=4,
    side="right",
    t_start=10.0,
    t_peak=11.2,
    t_end=12.6,
    min_angle=15.0,
    max_angle=150.0,
    rom=135.0,
    duration_s=2.6,
    mean_velocity_dps=100.0,
    peak_velocity_dps=180.0,
    smoothness=0.97,
    max_elbow_flexion=8.0,
    max_trunk_lean=3.0,
    max_shoulder_elevation=0.02,
    confidence=0.9,
    frames=39,
)


def codes(reasons: list[str]) -> list[str]:
    return [r.split(":")[0] for r in reasons]


def test_good_rep(cfg: ExerciseConfig) -> None:
    rep = evaluate_rep(GOOD, baseline_rom=140.0, cfg=cfg)
    assert rep.rule_quality is RepQuality.GOOD and rep.rule_reasons == []
    assert rep.rep_index == 4 and rep.rom == 135.0 and rep.max_shoulder_elevation == 0.02


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"max_trunk_lean": -14.0}, "trunk_lean"),
        ({"max_trunk_lean": 12.0}, "trunk_lean"),
        ({"max_elbow_flexion": 38.0}, "elbow_flexion"),
        ({"rom": 100.0, "max_angle": 115.0}, "reduced_rom"),
        ({"duration_s": 11.5}, "abnormal_duration"),
        ({"max_shoulder_elevation": 0.15}, "shoulder_elevation"),
    ],
)
def test_single_deviation_has_the_right_reason(
    cfg: ExerciseConfig, change: dict[str, Any], code: str
) -> None:
    rep = evaluate_rep(replace(GOOD, **change), baseline_rom=140.0, cfg=cfg)
    assert rep.rule_quality is RepQuality.DEVIATION
    assert codes(rep.rule_reasons) == [code]


def test_reasons_cite_numbers(cfg: ExerciseConfig) -> None:
    rep = evaluate_rep(replace(GOOD, rom=100.0), baseline_rom=140.0, cfg=cfg)
    assert rep.rule_reasons == ["reduced_rom: 100.0° < 80% of baseline 140.0° (71%)"]
    rep = evaluate_rep(replace(GOOD, max_trunk_lean=-14.0), baseline_rom=None, cfg=cfg)
    assert rep.rule_reasons == ["trunk_lean: 14.0° toward the arm > 10°"]


def test_multiple_deviations_all_reported(cfg: ExerciseConfig) -> None:
    bad = replace(GOOD, rom=90.0, max_trunk_lean=15.0, max_elbow_flexion=40.0)
    rep = evaluate_rep(bad, baseline_rom=140.0, cfg=cfg)
    assert rep.rule_quality is RepQuality.DEVIATION
    assert codes(rep.rule_reasons) == ["reduced_rom", "trunk_lean", "elbow_flexion"]


def test_no_rom_check_before_baseline(cfg: ExerciseConfig) -> None:
    rep = evaluate_rep(replace(GOOD, rom=60.0), baseline_rom=None, cfg=cfg)
    assert rep.rule_quality is RepQuality.GOOD


def test_low_confidence_is_uncertain_and_keeps_possible_reasons(cfg: ExerciseConfig) -> None:
    rep = evaluate_rep(replace(GOOD, confidence=0.5, rom=100.0), baseline_rom=140.0, cfg=cfg)
    assert rep.rule_quality is RepQuality.UNCERTAIN
    assert codes(rep.rule_reasons) == ["low_tracking_confidence", "possible_reduced_rom"]
    assert rep.rule_reasons[0] == "low_tracking_confidence: 0.50 < 0.6"


def test_quality_flags_make_rep_uncertain(cfg: ExerciseConfig) -> None:
    flagged = replace(GOOD, flag_fractions={"low_light": 0.5, "blurry": 0.1})
    rep = evaluate_rep(flagged, baseline_rom=140.0, cfg=cfg)
    assert rep.rule_quality is RepQuality.UNCERTAIN
    assert rep.rule_reasons == ["quality_flag: low_light on 50% of frames"]
    # below the fraction threshold the flag is ignored
    ok = evaluate_rep(replace(GOOD, flag_fractions={"low_light": 0.2}), 140.0, cfg)
    assert ok.rule_quality is RepQuality.GOOD


def test_elevation_missing_is_not_a_deviation(cfg: ExerciseConfig) -> None:
    rep = evaluate_rep(replace(GOOD, max_shoulder_elevation=None), baseline_rom=140.0, cfg=cfg)
    assert rep.rule_quality is RepQuality.GOOD
