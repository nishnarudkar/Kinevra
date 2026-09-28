"""PoseFrame → FrameFeatures for the configured side (PROJECT.md §2.4).

- shoulder_abduction_deg: hip → shoulder → elbow (primary ROM signal)
- elbow_flexion_deg:      180 - (shoulder → elbow → wrist); 0 = straight arm
- trunk_lean_deg:         mid-hip → mid-shoulder vs image vertical; + = away from the arm
- shoulder_elevation:     (shoulder height above hip - rest baseline) ÷ baseline; + = shrug
- angular_velocity_dps:   d(abduction)/dt from the smoothed signal and real timestamps
- confidence:             min visibility of hip, shoulder, elbow (the primary angle's points)

All angles are measured in pixels (aspect-ratio correct). A measurement is None when any of
its landmarks is below the visibility threshold; we never guess a hidden joint.

Two modes share `raw_features`:
- `FeatureExtractor.update()` (live): One Euro filters, causal.
- `extract_features()` (clip mode): Savitzky-Golay over the whole clip, non-causal.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from kinevra.config import ExerciseConfig
from kinevra.movement.geometry import (
    Point,
    angle_deg,
    elbow_flexion_deg,
    midpoint,
    signed_lean_deg,
    to_px,
)
from kinevra.movement.smoothing import OneEuroFilter, savgol_smooth
from kinevra.schemas import FrameFeatures, PoseFrame, Side


@dataclass(frozen=True, slots=True)
class RawFeatures:
    frame_idx: int
    t: float
    abduction: float | None
    elbow_flexion: float | None
    trunk_lean: float | None
    shoulder_height_px: float | None  # exercising shoulder above its hip, image px
    confidence: float


def _other(side: Side) -> Side:
    return "left" if side == "right" else "right"


def raw_features(pose: PoseFrame, side: Side, visibility_threshold: float) -> RawFeatures:
    w, h = pose.image_width, pose.image_height

    def pt(name: str) -> Point | None:
        lm = pose.landmarks.get(name)
        if lm is None or lm.visibility < visibility_threshold:
            return None
        return to_px(lm, w, h)

    hip, shoulder = pt(f"{side}_hip"), pt(f"{side}_shoulder")
    elbow, wrist = pt(f"{side}_elbow"), pt(f"{side}_wrist")
    o_hip, o_shoulder = pt(f"{_other(side)}_hip"), pt(f"{_other(side)}_shoulder")

    abduction = angle_deg(hip, shoulder, elbow) if hip and shoulder and elbow else None
    flexion = (
        elbow_flexion_deg(angle_deg(shoulder, elbow, wrist))
        if shoulder and elbow and wrist
        else None
    )
    lean = None
    if hip and o_hip and shoulder and o_shoulder:
        lateral = (o_hip[0] - hip[0], o_hip[1] - hip[1])
        lean = signed_lean_deg(midpoint(hip, o_hip), midpoint(shoulder, o_shoulder), lateral)
    height = hip[1] - shoulder[1] if hip and shoulder else None

    primary = [pose.landmarks.get(f"{side}_{j}") for j in ("hip", "shoulder", "elbow")]
    confidence = min((lm.visibility if lm else 0.0) for lm in primary)
    return RawFeatures(pose.frame_idx, pose.t, abduction, flexion, lean, height, confidence)


class ElevationBaseline:
    """Median shoulder height while the arm rests (abduction < rest_angle_max) for a while."""

    def __init__(self, rest_angle_max: float, duration_s: float, max_gap_s: float) -> None:
        self.rest_angle_max = rest_angle_max
        self.duration_s = duration_s
        self.max_gap_s = max_gap_s
        self.value: float | None = None
        self._heights: list[float] = []
        self._rest_s = 0.0
        self._last_t: float | None = None

    def add(self, t: float, abduction: float | None, height: float | None) -> float | None:
        if self.value is not None:
            return self.value
        if abduction is None or height is None or height <= 0 or abduction > self.rest_angle_max:
            self._last_t = None
            return None
        if self._last_t is not None:
            self._rest_s += min(t - self._last_t, self.max_gap_s)
        self._last_t = t
        self._heights.append(height)
        if self._rest_s >= self.duration_s - 1e-9:
            self.value = statistics.median(self._heights)
        return self.value

    @staticmethod
    def elevation(height: float | None, baseline: float | None) -> float | None:
        if height is None or baseline is None:
            return None
        return (height - baseline) / baseline


class _GapFilter:
    """One Euro filter that restarts after a tracking gap longer than `max_gap_s`."""

    def __init__(self, cfg: ExerciseConfig) -> None:
        oe = cfg.smoothing.one_euro
        self.filter = OneEuroFilter(oe.min_cutoff, oe.beta, oe.d_cutoff)
        self.max_gap_s = cfg.smoothing.max_gap_s
        self._last_t: float | None = None

    def __call__(self, value: float | None, t: float) -> float | None:
        if value is None:
            return None
        if self._last_t is not None and t - self._last_t > self.max_gap_s:
            self.filter.reset()
        self._last_t = t
        return self.filter(value, t)


class FeatureExtractor:
    """Streaming (live mode) features with One Euro smoothing."""

    def __init__(self, cfg: ExerciseConfig) -> None:
        self.cfg = cfg
        self.side: Side = cfg.side
        self._abduction = _GapFilter(cfg)
        self._flexion = _GapFilter(cfg)
        self._lean = _GapFilter(cfg)
        self._height = _GapFilter(cfg)
        self._baseline = ElevationBaseline(
            cfg.reps.rest_angle_max, cfg.features.elevation_baseline_s, cfg.smoothing.max_gap_s
        )

    def update(self, pose: PoseFrame) -> FrameFeatures:
        raw = raw_features(pose, self.side, self.cfg.visibility_threshold)
        abduction = self._abduction(raw.abduction, raw.t)
        velocity = self._abduction.filter.velocity if abduction is not None else None
        height = self._height(raw.shoulder_height_px, raw.t)
        baseline = self._baseline.add(raw.t, abduction, height)
        return FrameFeatures(
            frame_idx=raw.frame_idx,
            t=raw.t,
            side=self.side,
            shoulder_abduction_deg=abduction,
            elbow_flexion_deg=self._flexion(raw.elbow_flexion, raw.t),
            trunk_lean_deg=self._lean(raw.trunk_lean, raw.t),
            angular_velocity_dps=velocity,
            confidence=raw.confidence,
            shoulder_elevation=ElevationBaseline.elevation(height, baseline),
        )


def extract_features(poses: Sequence[PoseFrame], cfg: ExerciseConfig) -> list[FrameFeatures]:
    """Offline (clip mode) features with Savitzky-Golay smoothing over the whole clip."""
    raws = [raw_features(p, cfg.side, cfg.visibility_threshold) for p in poses]
    t = [r.t for r in raws]
    sg = cfg.smoothing.savgol

    def smooth(values: list[float | None]) -> tuple[list[float | None], list[float | None]]:
        return savgol_smooth(
            t,
            values,
            window_s=sg.window_s,
            polyorder=sg.polyorder,
            max_gap_s=cfg.smoothing.max_gap_s,
        )

    abduction, velocity = smooth([r.abduction for r in raws])
    flexion, _ = smooth([r.elbow_flexion for r in raws])
    lean, _ = smooth([r.trunk_lean for r in raws])
    height, _ = smooth([r.shoulder_height_px for r in raws])

    baseline_tracker = ElevationBaseline(
        cfg.reps.rest_angle_max, cfg.features.elevation_baseline_s, cfg.smoothing.max_gap_s
    )
    baseline = None
    for i in range(len(raws)):
        baseline = baseline_tracker.add(t[i], abduction[i], height[i])
        if baseline is not None:
            break

    return [
        FrameFeatures(
            frame_idx=r.frame_idx,
            t=r.t,
            side=cfg.side,
            shoulder_abduction_deg=abduction[i],
            elbow_flexion_deg=flexion[i],
            trunk_lean_deg=lean[i],
            angular_velocity_dps=velocity[i],
            confidence=r.confidence,
            shoulder_elevation=ElevationBaseline.elevation(height[i], baseline),
        )
        for i, r in enumerate(raws)
    ]


def smooth_live(
    t: Sequence[float], values: Sequence[float | None], cfg: ExerciseConfig
) -> tuple[list[float | None], list[float | None]]:
    """Apply the live-mode filter (One Euro + gap reset) to a series: (smoothed, velocity)."""
    gf = _GapFilter(cfg)
    smoothed: list[float | None] = []
    velocity: list[float | None] = []
    for ti, v in zip(t, values, strict=True):
        s = gf(v, ti)
        smoothed.append(s)
        velocity.append(gf.filter.velocity if s is not None else None)
    return smoothed, velocity
