"""Deterministic rule baseline: RepMeasurement → RepMetrics with GOOD / DEVIATION / UNCERTAIN.

Thresholds come from `rules:` in the exercise YAML and are PROTOTYPE DEFAULTS, not clinical
values. Every non-GOOD result carries machine-readable reasons ("code: detail with numbers").

Order of judgement:
1. UNCERTAIN if the data cannot be trusted (low tracking confidence, or a quality flag on too
   many of the rep's frames). Deviations seen on that data are still listed as
   `possible_<code>`, so the agent knows what to re-check with its OpenCV tools.
2. DEVIATION if any movement check fails (all failing checks are reported).
3. GOOD otherwise.
"""

from __future__ import annotations

from kinevra.config import ExerciseConfig
from kinevra.movement.reps import RepMeasurement
from kinevra.schemas import RepMetrics, RepQuality

# Quality flags that make measurements unreliable (see kinevra/vision/quality.py)
DATA_QUALITY_FLAGS = (
    "low_light",
    "overexposed",
    "low_contrast",
    "blurry",
    "out_of_frame",
    "no_person",
    "multiple_people",
)


def movement_deviations(
    m: RepMeasurement, baseline_rom: float | None, cfg: ExerciseConfig
) -> list[str]:
    r, reps = cfg.rules, cfg.reps
    out: list[str] = []
    if (
        baseline_rom is not None
        and baseline_rom > 0
        and m.rom < r.rom_ratio_deviation * baseline_rom
    ):
        out.append(
            f"reduced_rom: {m.rom:.1f}° < {r.rom_ratio_deviation:.0%} of baseline "
            f"{baseline_rom:.1f}° ({m.rom / baseline_rom:.0%})"
        )
    if abs(m.max_trunk_lean) > r.max_trunk_lean_deg:
        direction = "away from" if m.max_trunk_lean > 0 else "toward"
        out.append(
            f"trunk_lean: {abs(m.max_trunk_lean):.1f}° {direction} the arm > "
            f"{r.max_trunk_lean_deg:g}°"
        )
    if m.max_elbow_flexion > r.max_elbow_flexion_deg:
        out.append(f"elbow_flexion: {m.max_elbow_flexion:.1f}° > {r.max_elbow_flexion_deg:g}°")
    if m.max_shoulder_elevation is not None and m.max_shoulder_elevation > r.max_shoulder_elevation:
        out.append(
            f"shoulder_elevation: +{m.max_shoulder_elevation:.0%} > +{r.max_shoulder_elevation:.0%}"
        )
    if m.duration_s > reps.max_duration_s:
        out.append(f"abnormal_duration: {m.duration_s:.1f} s > {reps.max_duration_s:g} s")
    return out


def data_problems(m: RepMeasurement, cfg: ExerciseConfig) -> list[str]:
    r = cfg.rules
    out: list[str] = []
    if m.confidence < r.min_confidence:
        out.append(f"low_tracking_confidence: {m.confidence:.2f} < {r.min_confidence:g}")
    for flag in DATA_QUALITY_FLAGS:
        frac = m.flag_fractions.get(flag, 0.0)
        if frac > r.uncertain_flag_fraction:
            out.append(f"quality_flag: {flag} on {frac:.0%} of frames")
    return out


def evaluate_rep(m: RepMeasurement, baseline_rom: float | None, cfg: ExerciseConfig) -> RepMetrics:
    problems = data_problems(m, cfg)
    deviations = movement_deviations(m, baseline_rom, cfg)
    if problems:
        quality = RepQuality.UNCERTAIN
        reasons = problems + [f"possible_{d}" for d in deviations]
    elif deviations:
        quality, reasons = RepQuality.DEVIATION, deviations
    else:
        quality, reasons = RepQuality.GOOD, []
    return RepMetrics(
        rep_index=m.rep_index,
        side=m.side,
        t_start=m.t_start,
        t_peak=m.t_peak,
        t_end=m.t_end,
        min_angle=m.min_angle,
        max_angle=m.max_angle,
        rom=m.rom,
        duration_s=m.duration_s,
        mean_velocity_dps=m.mean_velocity_dps,
        peak_velocity_dps=m.peak_velocity_dps,
        smoothness=m.smoothness,
        max_elbow_flexion=m.max_elbow_flexion,
        max_trunk_lean=m.max_trunk_lean,
        confidence=m.confidence,
        rule_quality=quality,
        rule_reasons=reasons,
        max_shoulder_elevation=m.max_shoulder_elevation,
    )
