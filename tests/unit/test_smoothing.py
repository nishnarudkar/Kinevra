"""Smoothing tests (written before the implementation)."""

import math

import numpy as np
import pytest

from kinevra.movement.smoothing import OneEuroFilter, savgol_smooth


def _noisy(n: int, fps: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(n) / fps
    clean = 60 + 50 * np.sin(2 * math.pi * 0.25 * t)  # 4 s period, like a slow arm raise
    noisy = clean + np.random.default_rng(seed).normal(0, 3.0, n)
    return t, clean, noisy


# --- One Euro (live) --------------------------------------------------------------------------


def test_one_euro_first_sample_and_constant_input() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    assert f(42.0, 0.0) == 42.0
    for i in range(1, 50):
        assert f(42.0, i / 30) == pytest.approx(42.0)
    assert f.derivative == pytest.approx(0.0)


def test_one_euro_removes_jitter_when_still() -> None:
    t = np.arange(150) / 30.0
    noisy = 20.0 + np.random.default_rng(3).normal(0, 3.0, t.size)  # arm resting
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    out = np.array([f(float(x), float(ti)) for x, ti in zip(noisy, t, strict=True)])
    assert out[30:].std() < 0.5 * noisy[30:].std()


def test_one_euro_moving_is_no_worse_than_raw() -> None:
    # during motion the filter opens up (beta) to limit lag, so it smooths less
    t, clean, noisy = _noisy(300, 30.0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    out = np.array([f(float(x), float(ti)) for x, ti in zip(noisy, t, strict=True)])
    assert np.abs(out - clean)[30:].mean() < np.abs(noisy - clean)[30:].mean()


def test_one_euro_higher_beta_follows_fast_motion_better() -> None:
    t = np.arange(60) / 30.0
    ramp = 200.0 * t  # 200 °/s, a fast raise
    slow, fast = OneEuroFilter(1.0, 0.0, 1.0), OneEuroFilter(1.0, 0.5, 1.0)
    lag_slow = ramp[-1] - [slow(float(x), float(ti)) for x, ti in zip(ramp, t, strict=True)][-1]
    lag_fast = ramp[-1] - [fast(float(x), float(ti)) for x, ti in zip(ramp, t, strict=True)][-1]
    assert 0 < lag_fast < lag_slow
    assert fast.velocity == pytest.approx(200.0, rel=0.1)


def test_one_euro_velocity_converges_to_true_speed_despite_lag() -> None:
    t = np.arange(45) / 15.0
    f = OneEuroFilter(1.0, 0.05, 1.0)  # project defaults, 15 FPS
    for x, ti in zip(60.0 * t, t, strict=True):
        f(float(x), float(ti))
    assert f.velocity == pytest.approx(60.0, rel=0.05)
    assert f.derivative > 70.0  # the internal estimate overshoots — why we don't report it


def test_one_euro_uses_real_dt_and_resets() -> None:
    f = OneEuroFilter(1.0, 0.0, 1.0)
    f(0.0, 0.0)
    small_dt = f(10.0, 0.01)
    g = OneEuroFilter(1.0, 0.0, 1.0)
    g(0.0, 0.0)
    big_dt = g(10.0, 1.0)
    assert small_dt < big_dt < 10.0  # longer gap → trust the new sample more
    assert f(5.0, 0.01) == pytest.approx(small_dt)  # non-increasing t: sample ignored
    f.reset()
    assert f(7.0, 5.0) == 7.0


def test_one_euro_invalid_params() -> None:
    with pytest.raises(ValueError):
        OneEuroFilter(0.0, 0.1, 1.0)


# --- Savitzky-Golay (offline) -----------------------------------------------------------------


def test_savgol_preserves_quadratic_and_derivative() -> None:
    t = np.arange(40) / 15.0
    y = 3 * t**2 - 2 * t + 5
    smooth, deriv = savgol_smooth(t.tolist(), y.tolist(), window_s=0.5, polyorder=2)
    assert np.allclose(np.array(smooth, dtype=float), y, atol=1e-6)
    assert np.allclose(np.array(deriv, dtype=float), 6 * t - 2, atol=1e-4)


def test_savgol_reduces_noise() -> None:
    t, clean, noisy = _noisy(150, 15.0)
    smooth, _ = savgol_smooth(t.tolist(), noisy.tolist(), window_s=0.5, polyorder=2)
    s = np.array(smooth, dtype=float)
    assert np.abs(s - clean).mean() < 0.7 * np.abs(noisy - clean).mean()


def test_savgol_short_gap_is_bridged_but_stays_none() -> None:
    t = (np.arange(30) / 15.0).tolist()
    y: list[float | None] = [10.0 * ti for ti in t]
    y[10] = None
    y[11] = None  # 2 frames ≈ 0.13 s < max_gap_s
    smooth, deriv = savgol_smooth(t, y, window_s=0.4, polyorder=2, max_gap_s=0.3)
    assert smooth[10] is None and smooth[11] is None and deriv[10] is None
    assert smooth[9] == pytest.approx(10.0 * t[9], abs=1e-6)
    assert deriv[12] == pytest.approx(10.0, abs=1e-6)


def test_savgol_long_gap_splits_segments_and_short_segments_pass_through() -> None:
    t = (np.arange(40) / 15.0).tolist()
    y: list[float | None] = [float(i) for i in range(40)]
    for i in range(15, 30):
        y[i] = None  # 1 s gap
    y[32] = 1000.0  # spike in the second segment would leak across a bridged gap
    smooth, _ = savgol_smooth(t, y, window_s=0.4, polyorder=2, max_gap_s=0.3)
    assert all(v is None for v in smooth[15:30])
    assert smooth[5] == pytest.approx(5.0, abs=1e-6)  # first segment untouched by the spike

    two = savgol_smooth([0.0, 0.1], [1.0, 2.0], window_s=1.0, polyorder=2)
    assert two[0] == [1.0, 2.0] and two[1] == pytest.approx([10.0, 10.0])
    assert savgol_smooth([], [], window_s=0.4, polyorder=2) == ([], [])
    assert savgol_smooth([0.0], [None], window_s=0.4, polyorder=2) == ([None], [None])
