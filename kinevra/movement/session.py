"""Session state over time → SessionEvidence (the agent's input), plus the per-frame driver.

- baseline ROM: median ROM of the first `first_n_good_reps` GOOD reps; fixed once set.
- recent ROMs: last `recent_k` trusted reps (GOOD or DEVIATION; UNCERTAIN ROMs are unreliable).
- ROM trend: least-squares slope over the recent ROMs, with a dead-band → stable.
- consistency: 1 - coefficient of variation of the recent ROMs (clamped to 0..1).
- consecutive deviations: DEVIATION increments, GOOD resets, UNCERTAIN leaves it unchanged.
- tracking confidence / data-quality flags: rolling window over the last frames.

`MovementSession` wires RepCounter → rules → SessionState for live and clip mode alike.
"""

from __future__ import annotations

import statistics
from collections import Counter, deque
from collections.abc import Sequence
from typing import Literal

import numpy as np

from kinevra.config import ExerciseConfig
from kinevra.evaluation.rules import evaluate_rep
from kinevra.movement.reps import RepCounter, RepEvent, RepMeasurement
from kinevra.schemas import FrameFeatures, RepMetrics, RepQuality, SessionEvidence

Trend = Literal["increasing", "stable", "decreasing", "insufficient_data"]
MIN_REPS_FOR_TREND = 3


class SessionState:
    def __init__(
        self,
        cfg: ExerciseConfig,
        session_id: str,
        mode: Literal["live", "clip"],
        reps_target: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.session_id = session_id
        self.mode = mode
        self.reps_target = reps_target
        self.reps: list[RepMetrics] = []
        self.baseline_rom: float | None = None
        self.consecutive_deviations = 0
        self._window: deque[tuple[float, float, frozenset[str]]] = deque()

    def observe_frame(self, feat: FrameFeatures, flags: Sequence[str] = ()) -> None:
        self._window.append((feat.t, feat.confidence, frozenset(flags)))
        cutoff = feat.t - self.cfg.baseline.confidence_window_s
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def add_rep(self, rep: RepMetrics) -> None:
        self.reps.append(rep)
        if rep.rule_quality is RepQuality.DEVIATION:
            self.consecutive_deviations += 1
        elif rep.rule_quality is RepQuality.GOOD:
            self.consecutive_deviations = 0
        if self.baseline_rom is None:
            good = [r.rom for r in self.reps if r.rule_quality is RepQuality.GOOD]
            n = self.cfg.baseline.first_n_good_reps
            if len(good) >= n:
                self.baseline_rom = statistics.median(good[:n])

    @property
    def recent_roms(self) -> list[float]:
        trusted = [r.rom for r in self.reps if r.rule_quality is not RepQuality.UNCERTAIN]
        return trusted[-self.cfg.baseline.recent_k :]

    def rom_trend(self) -> Trend:
        roms = self.recent_roms
        if len(roms) < MIN_REPS_FOR_TREND:
            return "insufficient_data"
        slope = float(np.polyfit(np.arange(len(roms)), np.asarray(roms), 1)[0])
        if abs(slope) <= self.cfg.baseline.trend_deadband_deg_per_rep:
            return "stable"
        return "increasing" if slope > 0 else "decreasing"

    def consistency(self) -> float:
        roms = self.recent_roms
        if len(roms) < 2 or statistics.fmean(roms) <= 0:
            return 1.0  # no evidence of inconsistency yet
        cv = statistics.pstdev(roms) / statistics.fmean(roms)
        return float(min(1.0, max(0.0, 1.0 - cv)))

    def tracking_confidence(self) -> float:
        if not self._window:
            return 0.0
        return statistics.fmean(c for _, c, _ in self._window)

    def data_quality_flags(self) -> list[str]:
        if not self._window:
            return []
        counts: Counter[str] = Counter(f for _, _, fs in self._window for f in fs)
        limit = self.cfg.rules.uncertain_flag_fraction * len(self._window)
        return sorted(f for f, c in counts.items() if c > limit)

    def evidence(self) -> SessionEvidence:
        return SessionEvidence(
            session_id=self.session_id,
            mode=self.mode,
            exercise=self.cfg.exercise,
            side=self.cfg.side,
            reps_completed=len(self.reps),
            reps_target=self.reps_target,
            latest_rep=self.reps[-1] if self.reps else None,
            baseline_rom=self.baseline_rom,
            recent_roms=self.recent_roms,
            rom_trend=self.rom_trend(),
            consecutive_deviations=self.consecutive_deviations,
            movement_consistency=self.consistency(),
            tracking_confidence=self.tracking_confidence(),
            data_quality_flags=self.data_quality_flags(),
        )


class MovementSession:
    """Per-frame driver: features → rep counter → rules → session state."""

    def __init__(
        self,
        cfg: ExerciseConfig,
        session_id: str,
        mode: Literal["live", "clip"] = "live",
        reps_target: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.counter = RepCounter(cfg.reps, cfg.side, cfg.smoothing.max_gap_s)
        self.state = SessionState(cfg, session_id, mode, reps_target)

    @property
    def events(self) -> list[RepEvent]:
        return self.counter.events

    def update(self, feat: FrameFeatures, flags: Sequence[str] = ()) -> RepMetrics | None:
        """Feed one frame; returns the rep (with its rule quality) if one completed."""
        self.state.observe_frame(feat, flags)
        out = self.counter.update(feat, flags)
        if not isinstance(out, RepMeasurement):
            return None
        rep = evaluate_rep(out, self.state.baseline_rom, self.cfg)
        self.state.add_rep(rep)
        return rep

    def finish(self) -> RepEvent | None:
        return self.counter.finish()

    def evidence(self) -> SessionEvidence:
        return self.state.evidence()
