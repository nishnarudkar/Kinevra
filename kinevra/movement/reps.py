"""Repetition state machine for shoulder abduction: REST → RAISING → PEAK → LOWERING → REST.

Hysteresis on the smoothed abduction angle (thresholds from `reps:` in the exercise YAML):
- REST tracks the arm's low point (the "anchor"); RAISING starts when the angle exceeds
  anchor + raise_margin.
- PEAK: the angle stops rising (slow velocity, or a small drop below the running max).
  PEAK returns to RAISING only on a real further rise, so a noisy pause at the top is one rep.
- LOWERING starts once the angle is raise_margin below the max.
- The rep closes when the arm is back down (≤ max(rest_angle_max, anchor + raise_margin)), or
  when it rises again by raise_margin from a trough without getting back down
  (`incomplete_return`: the rep closes at the trough and the next one starts there).

ROM uses robust endpoints so noise extremes do not inflate it: start = median of the rest
angles in the 0.5 s before the rise; peak = the maximum, unless it is a one-frame spike
(> 3° above the 3-sample median around it), then that median.

A closed rep is counted only if ROM ≥ min_rom and duration ≥ min_duration_s; otherwise it is
logged as a `partial_rep` / `too_short` event. Long reps are counted; the rules judge them.
Frames with no angle (tracking lost) never trigger transitions but lower the rep confidence.
"""

from __future__ import annotations

import statistics
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise

from kinevra.config import RepsCfg
from kinevra.schemas import FrameFeatures, Side

ANCHOR_TOLERANCE_DEG = 1.0  # rest samples within this of the low point move the anchor forward
REST_BASELINE_S = 0.5  # ROM start = median rest angle over this window before the rise
PEAK_SPIKE_DEG = 3.0  # a max this far above its 3-sample median is a one-frame spike
PEAK_VELOCITY_DPS = 10.0  # slower than this near the top counts as "stopped rising"


class RepPhase(str, Enum):
    REST = "REST"
    RAISING = "RAISING"
    PEAK = "PEAK"
    LOWERING = "LOWERING"


@dataclass(frozen=True)
class RepMeasurement:
    """Deterministic measurements of one counted rep (rules add the quality label)."""

    rep_index: int  # 1-based count of valid reps
    side: Side
    t_start: float
    t_peak: float
    t_end: float
    min_angle: float
    max_angle: float
    rom: float
    duration_s: float
    mean_velocity_dps: float
    peak_velocity_dps: float
    smoothness: float  # ideal path / travelled path of the angle, 0..1 (1 = one clean up-down)
    max_elbow_flexion: float
    max_trunk_lean: float  # signed value with the largest magnitude
    max_shoulder_elevation: float | None
    confidence: float  # mean frame confidence over the rep
    frames: int
    flag_fractions: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RepEvent:
    kind: str  # partial_rep | too_short | incomplete_return | unfinished_rep
    t_start: float
    t_end: float
    max_angle: float
    rom: float
    detail: str = ""


@dataclass
class _Point:
    t: float
    angle: float


class RepCounter:
    def __init__(self, cfg: RepsCfg, side: Side, max_gap_s: float = 0.3) -> None:
        self.cfg = cfg
        self.side = side
        self.max_gap_s = max_gap_s
        self.peak_margin = cfg.raise_margin / 3.0
        self.count = 0
        self.events: list[RepEvent] = []
        self.phase = RepPhase.REST
        self._reset_rest(None, None)
        self._frames: list[tuple[FrameFeatures, Sequence[str]]] = []
        self._rest_history: deque[_Point] = deque()
        self._start_level = 0.0

    # --- state helpers ------------------------------------------------------------------------

    def _reset_rest(self, feat: FrameFeatures | None, flags: Sequence[str] | None) -> None:
        self.phase = RepPhase.REST
        self._anchor: _Point | None = None
        self._tail: list[tuple[FrameFeatures, Sequence[str]]] = []
        if feat is not None and feat.shoulder_abduction_deg is not None:
            self._anchor = _Point(feat.t, feat.shoulder_abduction_deg)
            self._tail = [(feat, flags or ())]

    def _start_rep(self, start: _Point, tail: list[tuple[FrameFeatures, Sequence[str]]]) -> None:
        self.phase = RepPhase.RAISING
        self._start = start
        self._start_level = start.angle
        self._frames = list(tail)
        self._max = _Point(start.t, start.angle)
        self._peak_ref = start.angle
        self._trough: _Point | None = None
        self._trough_last: _Point | None = None

    @property
    def _end_threshold(self) -> float:
        return max(self.cfg.rest_angle_max, self._start.angle + self.cfg.raise_margin)

    # --- main entry ---------------------------------------------------------------------------

    def update(
        self, feat: FrameFeatures, flags: Sequence[str] = ()
    ) -> RepMeasurement | RepEvent | None:
        """Feed one frame; returns a counted rep or an event when one closes on this frame."""
        angle = feat.shoulder_abduction_deg
        if self.phase is RepPhase.REST:
            if angle is None:
                return None
            self._rest_history.append(_Point(feat.t, angle))
            while self._rest_history[0].t < feat.t - REST_BASELINE_S:
                self._rest_history.popleft()
            if self._anchor is None or angle <= self._anchor.angle + ANCHOR_TOLERANCE_DEG:
                if self._anchor is None or angle < self._anchor.angle:
                    self._anchor = _Point(feat.t, angle)
                else:
                    self._anchor = _Point(feat.t, self._anchor.angle)
                self._tail = [(feat, flags)]
                return None
            self._tail.append((feat, flags))
            if angle >= self._anchor.angle + self.cfg.raise_margin:
                window = [p.angle for p in self._rest_history if p.t <= self._anchor.t]
                self._start_rep(self._anchor, self._tail)
                self._start_level = statistics.median(window) if window else self._anchor.angle
                self._rest_history.clear()
            return None

        self._frames.append((feat, flags))
        if angle is None:
            return None
        if angle > self._max.angle:
            self._max = _Point(feat.t, angle)
        vel = feat.angular_velocity_dps

        if self.phase is RepPhase.RAISING:
            slowed = vel is not None and vel <= PEAK_VELOCITY_DPS
            if angle <= self._max.angle - self.peak_margin or (
                slowed and angle >= self._start.angle + self.cfg.raise_margin
            ):
                self.phase = RepPhase.PEAK
                self._peak_ref = self._max.angle
        elif self.phase is RepPhase.PEAK and angle > self._peak_ref + self.peak_margin:
            self.phase = RepPhase.RAISING

        if self.phase in (RepPhase.RAISING, RepPhase.PEAK):
            if angle <= self._max.angle - self.cfg.raise_margin:
                self.phase = RepPhase.LOWERING
                self._trough = self._trough_last = _Point(feat.t, angle)
            else:
                return None

        # LOWERING
        assert self._trough is not None and self._trough_last is not None
        if angle < self._trough.angle:
            self._trough = self._trough_last = _Point(feat.t, angle)
        elif angle <= self._trough.angle + ANCHOR_TOLERANCE_DEG:
            self._trough_last = _Point(feat.t, self._trough.angle)

        if angle <= self._end_threshold:
            out = self._close(feat.t, angle, len(self._frames))
            self._reset_rest(feat, flags)
            return out
        if angle >= self._trough.angle + self.cfg.raise_margin:
            # rose again without getting back down: close at the trough, start the next rep
            trough, trough_last = self._trough, self._trough_last
            cut = next(
                (i for i, (f, _) in enumerate(self._frames) if f.t >= trough.t),
                len(self._frames) - 1,
            )
            tail = [fr for fr in self._frames[cut:] if fr[0].t >= trough_last.t]
            out = self._close(trough.t, trough.angle, cut + 1)
            self.events.append(
                RepEvent(
                    "incomplete_return",
                    self._start.t,
                    trough.t,
                    self._max.angle,
                    self._max.angle - self._start.angle,
                    f"returned only to {trough.angle:.1f}°",
                )
            )
            self._start_rep(trough_last, tail)
            self._rest_history.clear()
            return out
        return None

    def finish(self) -> RepEvent | None:
        """End of stream: a rep still in progress is logged, never counted."""
        if self.phase is RepPhase.REST or not self._frames:
            return None
        last_t = self._frames[-1][0].t
        event = RepEvent(
            "unfinished_rep",
            self._start.t,
            last_t,
            self._max.angle,
            self._max.angle - self._start.angle,
            f"stream ended in {self.phase.value}",
        )
        self.events.append(event)
        self.phase = RepPhase.REST
        return event

    # --- measurement --------------------------------------------------------------------------

    def _close(self, t_end: float, end_angle: float, n_frames: int) -> RepMeasurement | RepEvent:
        frames = self._frames[:n_frames]
        start, peak = self._start, self._max
        series = [
            (f.t, f.shoulder_abduction_deg)
            for f, _ in frames
            if f.shoulder_abduction_deg is not None
        ]
        i_peak = next((i for i, (t, _) in enumerate(series) if t == peak.t), 0)
        peak_median = statistics.median(a for _, a in series[max(0, i_peak - 1) : i_peak + 2])
        peak_level = peak_median if peak.angle - peak_median > PEAK_SPIKE_DEG else peak.angle
        start_level = min(self._start_level, peak_level)
        rom = peak_level - start_level
        duration = t_end - start.t
        if rom < self.cfg.min_rom or duration < self.cfg.min_duration_s:
            kind = "partial_rep" if rom < self.cfg.min_rom else "too_short"
            detail = (
                f"ROM {rom:.1f}° < {self.cfg.min_rom:g}°"
                if kind == "partial_rep"
                else f"{duration:.2f} s < {self.cfg.min_duration_s:g} s"
            )
            event = RepEvent(kind, start.t, t_end, peak.angle, rom, detail)
            self.events.append(event)
            return event

        feats = [f for f, _ in frames]
        angles = [f.shoulder_abduction_deg for f in feats if f.shoulder_abduction_deg is not None]
        travelled = sum(abs(b - a) for a, b in pairwise(angles))
        ideal = (peak.angle - start.angle) + (peak.angle - end_angle)  # raw path, like travelled
        smoothness = min(1.0, ideal / travelled) if travelled > 0 else 0.0
        vels = [abs(f.angular_velocity_dps) for f in feats if f.angular_velocity_dps is not None]
        elbows = [f.elbow_flexion_deg for f in feats if f.elbow_flexion_deg is not None]
        leans = [f.trunk_lean_deg for f in feats if f.trunk_lean_deg is not None]
        elevs = [f.shoulder_elevation for f in feats if f.shoulder_elevation is not None]
        flag_counts: Counter[str] = Counter(fl for _, fls in frames for fl in set(fls))

        self.count += 1
        return RepMeasurement(
            rep_index=self.count,
            side=self.side,
            t_start=start.t,
            t_peak=peak.t,
            t_end=t_end,
            min_angle=start_level,
            max_angle=peak_level,
            rom=rom,
            duration_s=duration,
            mean_velocity_dps=sum(vels) / len(vels) if vels else 0.0,
            peak_velocity_dps=max(vels) if vels else 0.0,
            smoothness=smoothness,
            max_elbow_flexion=max(elbows) if elbows else 0.0,
            max_trunk_lean=max(leans, key=abs) if leans else 0.0,
            max_shoulder_elevation=max(elevs) if elevs else None,
            confidence=sum(f.confidence for f in feats) / len(feats),
            frames=len(feats),
            flag_fractions={k: v / len(frames) for k, v in flag_counts.items()},
        )
