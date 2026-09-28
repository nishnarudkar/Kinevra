"""Plot raw vs smoothed shoulder abduction and angular velocity (PROJECT.md Phase 3).

Sources (pick one):
    --video clip.mp4   run pose + features on a clip (live One Euro and offline Savitzky-Golay)
    --csv run.csv      a CSV written by `run_live.py --csv` (offline smoothing added here)
    --synthetic        noisy synthetic session with known ground truth (method check only)

    uv run python scripts/plot_features.py --video data/clips/good_01.mp4
    uv run python scripts/plot_features.py --synthetic

Writes eval/smoothing_<name>.png and prints error numbers (vs ground truth when synthetic).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kinevra.config import ExerciseConfig, load_config
from kinevra.movement.export import read_feature_csv
from kinevra.movement.features import extract_features, raw_features, smooth_live
from kinevra.movement.smoothing import savgol_smooth

ROOT = Path(__file__).resolve().parents[1]

# Reference palette (dataviz skill): raw = neutral context, smoothers = categorical slots 1-2.
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RAW, LIVE, OFFLINE, TRUTH = "#9b9a96", "#2a78d6", "#eb6834", "#0b0b0b"

Series = list[float | None]


@dataclass
class Signals:
    name: str
    t: list[float]
    raw: Series
    live: Series
    live_vel: Series
    offline: Series
    offline_vel: Series
    truth: list[float] | None = None
    note: str = ""


def from_video(path: Path, cfg_path: Path | None) -> Signals:
    from kinevra.pose.factory import create_estimator
    from kinevra.vision.capture import FrameSource

    cfg = load_config(cfg_path)
    ex = cfg.exercise
    est = create_estimator(cfg)
    poses = []
    with FrameSource(path, target_fps=ex.processing_fps) as src:
        for frame in src:
            poses.append(est.estimate(frame.image, frame.t, frame.idx))
    t = [p.t for p in poses]
    raw = [raw_features(p, ex.side, ex.visibility_threshold).abduction for p in poses]
    live, live_vel = smooth_live(t, raw, ex)
    feats = extract_features(poses, ex)
    return Signals(
        path.stem,
        t,
        raw,
        live,
        live_vel,
        [f.shoulder_abduction_deg for f in feats],
        [f.angular_velocity_dps for f in feats],
        note=f"{len(poses)} frames at {ex.processing_fps:g} FPS, {est.name}",
    )


def from_csv(path: Path, ex: ExerciseConfig) -> Signals:
    rows = read_feature_csv(path)
    t = [float(r["t"]) for r in rows]
    raw = [r["raw_abduction_deg"] for r in rows]
    sg = ex.smoothing.savgol
    offline, offline_vel = savgol_smooth(
        t, raw, window_s=sg.window_s, polyorder=sg.polyorder, max_gap_s=ex.smoothing.max_gap_s
    )
    return Signals(
        path.stem,
        t,
        raw,
        [r["shoulder_abduction_deg"] for r in rows],
        [r["angular_velocity_dps"] for r in rows],
        offline,
        offline_vel,
        note=f"{len(rows)} frames from run_live CSV",
    )


def synthetic(ex: ExerciseConfig, seed: int = 0) -> Signals:
    """3 raises (15° → 150°) at 15 FPS with 3° jitter and two short tracking dropouts."""
    fps, n = ex.processing_fps, int(ex.processing_fps * 15)
    rng = np.random.default_rng(seed)
    t = [i / fps for i in range(n)]
    truth = []
    for ti in t:
        phase = (ti - 1.5) % 4.5
        up = 1.5 <= ti <= 15 and phase < 3.0  # 3 s raise+lower, 1.5 s rest
        truth.append(15 + 135 * math.sin(math.pi * phase / 3.0) ** 2 if up else 15.0)
    raw: Series = [a + float(rng.normal(0, 3.0)) for a in truth]
    for i in (70, 71, 150):
        raw[i] = None
    live, live_vel = smooth_live(t, raw, ex)
    sg = ex.smoothing.savgol
    offline, offline_vel = savgol_smooth(
        t, raw, window_s=sg.window_s, polyorder=sg.polyorder, max_gap_s=ex.smoothing.max_gap_s
    )
    return Signals(
        "synthetic",
        t,
        raw,
        live,
        live_vel,
        offline,
        offline_vel,
        truth,
        note="SYNTHETIC: 3 raises 15°→150°, sd=3° jitter, 3 dropped frames",
    )


def _arr(s: Series) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in s], dtype=float)


def report(sig: Signals) -> list[str]:
    lines = []
    raw = _arr(sig.raw)
    for label, s in (("One Euro (live)", sig.live), ("Savitzky-Golay (offline)", sig.offline)):
        sm = _arr(s)
        if sig.truth is not None:
            truth = np.array(sig.truth)
            mae_raw = np.nanmean(np.abs(raw - truth))
            mae = np.nanmean(np.abs(sm - truth))
            lines.append(f"{label}: MAE vs truth {mae:.2f}° (raw {mae_raw:.2f}°)")
        else:
            diff = np.nanmean(np.abs(sm - raw))
            lines.append(f"{label}: mean |smoothed - raw| {diff:.2f}°")
    return lines


def plot(sig: Signals, out: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
        }
    )
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
        facecolor=SURFACE,
    )
    t = np.array(sig.t)
    if sig.truth is not None:
        ax1.plot(t, sig.truth, color=TRUTH, lw=1, ls=(0, (4, 3)), label="ground truth")
    ax1.plot(t, _arr(sig.raw), color=RAW, lw=1, marker="o", ms=2.5, label="raw")
    ax1.plot(t, _arr(sig.live), color=LIVE, lw=2, label="One Euro (live)")
    ax1.plot(t, _arr(sig.offline), color=OFFLINE, lw=2, label="Savitzky-Golay (offline)")
    ax1.set_ylabel("shoulder abduction (°)")
    ax2.plot(t, _arr(sig.live_vel), color=LIVE, lw=2, label="One Euro (live)")
    ax2.plot(t, _arr(sig.offline_vel), color=OFFLINE, lw=2, label="Savitzky-Golay (offline)")
    ax2.axhline(0, color=GRID, lw=1)
    ax2.set_ylabel("angular velocity (°/s)")
    ax2.set_xlabel("time (s)")
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.0),
            frameon=False,
            borderaxespad=0.2,
            ncol=4 if ax is ax1 else 2,
            fontsize=9,
        )
    fig.suptitle(
        f"Kinevra — raw vs smoothed abduction: {sig.name}",
        x=0.01,
        ha="left",
        fontsize=13,
        color=INK,
    )
    fig.text(0.01, 0.935, sig.note + "   |   " + "   ".join(report(sig)), fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path)
    src.add_argument("--csv", type=Path)
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--config", type=Path)
    p.add_argument("--out", type=Path, help="PNG path (default eval/smoothing_<name>.png)")
    args = p.parse_args(argv)

    ex = load_config(args.config).exercise
    if args.video:
        sig = from_video(args.video, args.config)
    elif args.csv:
        sig = from_csv(args.csv, ex)
    else:
        sig = synthetic(ex)
    out = args.out or ROOT / "eval" / f"smoothing_{sig.name}.png"
    plot(sig, out)
    ascii_report = "\n".join(report(sig)).replace("°", " deg")
    print(ascii_report)  # Windows consoles lack UTF-8
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
