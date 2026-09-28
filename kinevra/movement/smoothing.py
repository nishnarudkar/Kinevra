"""Signal smoothing for joint angles.

- Live mode: One Euro filter (Casiez et al., CHI 2012) — adaptive low-pass that uses the
  real time step, removes jitter when the arm is still and adds little lag when it moves fast.
- Clip mode (offline): Savitzky-Golay on contiguous segments, which keeps peak heights (ROM)
  better than a moving average and also gives the derivative (angular velocity).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.signal import savgol_filter


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0 or beta < 0:
            raise ValueError("min_cutoff and d_cutoff must be > 0, beta >= 0")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._t: float | None = None
        # Casiez's speed estimate (new sample vs lagging filtered value). It drives the
        # adaptive cutoff but overshoots the true speed during motion, so do not report it.
        self.derivative = 0.0
        # Low-passed rate of change of the smoothed output: converges to the true speed.
        self.velocity = 0.0

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._t, self.derivative, self.velocity = x, t, 0.0, 0.0
            return x
        dt = t - self._t
        if dt <= 0:  # duplicate / out-of-order sample: keep the previous estimate
            return self._x
        a_d = _alpha(self.d_cutoff, dt)
        self.derivative += a_d * ((x - self._x) / dt - self.derivative)
        cutoff = self.min_cutoff + self.beta * abs(self.derivative)
        previous = self._x
        self._x += _alpha(cutoff, dt) * (x - self._x)
        self.velocity += a_d * ((self._x - previous) / dt - self.velocity)
        self._t = t
        return self._x


def _segments(t: np.ndarray, valid: np.ndarray, max_gap_s: float) -> list[tuple[int, int]]:
    """[start, end) index ranges; a range may contain invalid samples in gaps <= max_gap_s."""
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return []
    out: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if t[i] - t[prev] > max_gap_s:
            out.append((start, prev + 1))
            start = i
        prev = i
    out.append((start, prev + 1))
    return out


def savgol_smooth(
    t: Sequence[float],
    values: Sequence[float | None],
    *,
    window_s: float,
    polyorder: int,
    max_gap_s: float = 0.3,
) -> tuple[list[float | None], list[float | None]]:
    """Smooth `values` sampled at (near-uniform) times `t`; returns (smoothed, d/dt).

    Missing values (None) in gaps up to `max_gap_s` are linearly bridged so the filter can
    run across them, but stay None in the output (we never invent measurements). Longer gaps
    split the series into independently filtered segments. Segments too short for the filter
    are passed through, with a finite-difference derivative.
    """
    n = len(values)
    smooth: list[float | None] = [None] * n
    deriv: list[float | None] = [None] * n
    if n == 0:
        return smooth, deriv
    tt = np.asarray(t, dtype=np.float64)
    valid = np.array([v is not None for v in values])
    yy = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)

    for a, b in _segments(tt, valid, max_gap_s):
        ts, ys, ok = tt[a:b], yy[a:b].copy(), valid[a:b]
        if not ok.all():
            ys[~ok] = np.interp(ts[~ok], ts[ok], ys[ok])
        m = b - a
        dt = float(np.median(np.diff(ts))) if m > 1 else 0.0
        window = 0
        if dt > 0:
            window = max(polyorder + 2, round(window_s / dt))
            window = min(window, m if m % 2 else m - 1)
            window += 0 if window % 2 else -1
        if window > polyorder:
            ys_s = savgol_filter(ys, window, polyorder, mode="interp")
            dys = savgol_filter(ys, window, polyorder, deriv=1, delta=dt, mode="interp")
        else:
            ys_s = ys
            dys = np.gradient(ys, ts) if m > 1 else np.zeros(m)
        for k in range(m):
            if ok[k]:
                smooth[a + k] = float(ys_s[k])
                deriv[a + k] = float(dys[k])
    return smooth, deriv
