"""Synthetic FrameFeatures sequences for rep / rules / session tests (no camera, no model).

Each rep is rest → raise (smooth cosine) → optional hold at the top → lower → rest.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from kinevra.schemas import FrameFeatures

FPS = 15.0


@dataclass(frozen=True)
class RepSpec:
    peak: float = 150.0
    start: float = 15.0
    rise_s: float = 1.2
    hold_s: float = 0.0
    lower_s: float = 1.2
    end: float | None = None  # angle reached after lowering (default: back to `start`)
    lean: float = 0.0  # peak trunk lean during the rep (signed)
    elbow: float = 5.0  # peak elbow flexion during the rep
    elevation: float = 0.0  # peak shoulder elevation during the rep
    confidence: float = 0.95
    dropout: tuple[float, float] | None = None  # (from, to) fraction of the rep with no angle
    flags: tuple[str, ...] = ()  # quality flags on every frame of this rep


@dataclass
class Session:
    features: list[FrameFeatures] = field(default_factory=list)
    flags: list[list[str]] = field(default_factory=list)


def _ease(u: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(max(u, 0.0), 1.0))


def build_session(
    reps: Sequence[RepSpec],
    *,
    rest_s: float = 1.0,
    noise: float = 0.0,
    seed: int = 0,
    side: str = "right",
) -> Session:
    rng = np.random.default_rng(seed)
    samples: list[tuple[float | None, float, float, float, float, tuple[str, ...]]] = []

    def rest(angle: float, seconds: float) -> None:
        for _ in range(round(seconds * FPS)):
            samples.append((angle, 0.0, 5.0, 0.0, 0.95, ()))

    rest(reps[0].start if reps else 15.0, rest_s)
    for r in reps:
        end = r.start if r.end is None else r.end
        total = r.rise_s + r.hold_s + r.lower_s
        n = round(total * FPS)
        for i in range(n):
            s = i / FPS
            if s < r.rise_s:
                a = r.start + (r.peak - r.start) * _ease(s / r.rise_s)
            elif s < r.rise_s + r.hold_s:
                a = r.peak
            else:
                a = r.peak + (end - r.peak) * _ease((s - r.rise_s - r.hold_s) / r.lower_s)
            k = (a - min(r.start, end)) / max(r.peak - min(r.start, end), 1e-9)  # 0..1 effort
            frac = i / max(n - 1, 1)
            angle: float | None = a
            if r.dropout and r.dropout[0] <= frac <= r.dropout[1]:
                angle = None
            samples.append(
                (angle, r.lean * k, max(5.0, r.elbow * k), r.elevation * k, r.confidence, r.flags)
            )
        rest(end, rest_s)

    session = Session()
    prev: tuple[float, float] | None = None
    for i, (angle, lean, elbow, elev, conf, flags) in enumerate(samples):
        t = i / FPS
        if angle is not None and noise:
            angle += float(rng.normal(0, noise))
        vel = None
        if angle is not None:
            vel = 0.0 if prev is None else (angle - prev[1]) / (t - prev[0])
            prev = (t, angle)
        session.features.append(
            FrameFeatures(
                frame_idx=i,
                t=t,
                side=side,
                shoulder_abduction_deg=angle,
                elbow_flexion_deg=elbow,
                trunk_lean_deg=lean,
                angular_velocity_dps=vel,
                confidence=conf if angle is not None else 0.1,
                shoulder_elevation=elev,
            )
        )
        session.flags.append(list(flags))
    return session
