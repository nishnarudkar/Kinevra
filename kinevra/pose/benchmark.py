"""Pose benchmark: latency, FPS, usable frames, still-pose jitter, full-frame vs ROI confidence.

Usage:
    uv run python -m kinevra.pose.benchmark data/clips/*.mp4
    uv run python -m kinevra.pose.benchmark --models rtmpose mediapipe --roi clip.mp4
    uv run python -m kinevra.pose.benchmark --resize-width 320 --roi far_clip.mp4

Images (.jpg/.png) are accepted too and repeated `--repeat` times. Results are printed as a
Markdown table and saved as JSON under eval/benchmarks/ (with machine + OpenCV info, so
x86 laptop and arm64 Lambda runs can be compared).

Jitter is only meaningful on clips where the subject holds still.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from kinevra.config import CONFIG_DIR, load_config
from kinevra.pose.base import PoseEstimator
from kinevra.pose.factory import MODEL_NAMES, create_estimator
from kinevra.schemas import PoseFrame
from kinevra.vision.capture import FrameSource
from kinevra.vision.preprocess import resize_to_width, roi_from_points
from kinevra.vision.types import Frame

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class BenchmarkResult:
    model: str
    source: str
    frames: int
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    pose_fps: float
    usable_pct: float  # % frames with all required landmarks visible
    required_conf_mean: float  # mean visibility of required landmarks (full frame)
    jitter_px_median: float | None  # frame-to-frame displacement of required landmarks
    jitter_torso_pct: float | None  # same, as % of torso length
    roi_frames: int = 0
    roi_conf_mean: float | None = None
    roi_conf_gain: float | None = None  # roi_conf_mean - required_conf_mean on the same frames
    roi_latency_ms_mean: float | None = None
    roi_usable_pct: float | None = None  # % of ROI runs with all required landmarks visible
    roi_recovered_pct: float | None = None  # % of full-frame failures rescued by ROI


def _required_conf(pf: PoseFrame, required: Sequence[str]) -> float:
    vals = [pf.landmarks[n].visibility if n in pf.landmarks else 0.0 for n in required]
    return float(np.mean(vals)) if vals else 0.0


def _usable(pf: PoseFrame, required: Sequence[str], thr: float) -> bool:
    return all(n in pf.landmarks and pf.landmarks[n].visibility >= thr for n in required)


def run_benchmark(
    estimator: PoseEstimator,
    frames: Iterable[Frame],
    *,
    required: Sequence[str],
    visibility_threshold: float,
    source: str = "",
    roi: bool = False,
    roi_padding: float = 0.6,
    roi_scale: float | None = None,
) -> BenchmarkResult:
    side = required[0].split("_")[0] if required else "right"
    latencies: list[float] = []
    confs: list[float] = []
    usable = 0
    disp_px: list[float] = []
    disp_torso: list[float] = []
    prev: dict[str, tuple[float, float]] | None = None
    roi_pairs: list[tuple[float, float]] = []
    roi_lat: list[float] = []
    roi_usable = 0
    full_failures = 0
    recovered = 0
    # ROI seed: this frame's landmarks, else the last frame where they were all visible
    # (how the agent re-analyses a rep whose full-frame pass lost the person).
    seed: list[tuple[float, float]] | None = None

    for frame in frames:
        h, w = frame.image.shape[:2]
        pf = estimator.estimate(frame.image, frame.t, frame.idx)
        latencies.append(estimator.last_latency_ms)
        conf = _required_conf(pf, required)
        confs.append(conf)
        ok = _usable(pf, required, visibility_threshold)
        usable += ok

        if ok:
            pts = {n: (pf.landmarks[n].x * w, pf.landmarks[n].y * h) for n in required}
            sh, hip = pts.get(f"{side}_shoulder"), pts.get(f"{side}_hip")
            torso = float(np.hypot(sh[0] - hip[0], sh[1] - hip[1])) if sh and hip else 0.0
            if prev is not None:
                d = [float(np.hypot(pts[n][0] - prev[n][0], pts[n][1] - prev[n][1])) for n in pts]
                disp_px.append(float(np.median(d)))
                if torso > 0:
                    disp_torso.append(100.0 * float(np.median(d)) / torso)
            prev = pts
        else:
            prev = None

        if ok:
            seed = [(pf.landmarks[n].x, pf.landmarks[n].y) for n in required]
        if roi and seed is not None:
            box = roi_from_points(seed, w, h, padding=roi_padding)
            pf_roi = estimator.estimate(
                frame.image, frame.t, frame.idx, roi=box, roi_scale=roi_scale
            )
            roi_lat.append(estimator.last_latency_ms)
            roi_pairs.append((conf, _required_conf(pf_roi, required)))
            roi_ok = _usable(pf_roi, required, visibility_threshold)
            roi_usable += roi_ok
            if not ok:
                full_failures += 1
                recovered += roi_ok

    n = len(latencies)
    lat = np.asarray(latencies) if n else np.zeros(1)
    result = BenchmarkResult(
        model=getattr(estimator, "name", type(estimator).__name__),
        source=source,
        frames=n,
        latency_ms_mean=round(float(lat.mean()), 2),
        latency_ms_p50=round(float(np.percentile(lat, 50)), 2),
        latency_ms_p95=round(float(np.percentile(lat, 95)), 2),
        pose_fps=round(1000.0 / float(lat.mean()), 1) if n and lat.mean() > 0 else 0.0,
        usable_pct=round(100.0 * usable / n, 1) if n else 0.0,
        required_conf_mean=round(float(np.mean(confs)), 3) if confs else 0.0,
        jitter_px_median=round(float(np.median(disp_px)), 2) if disp_px else None,
        jitter_torso_pct=round(float(np.median(disp_torso)), 2) if disp_torso else None,
    )
    if roi_pairs:
        full, zoomed = np.asarray(roi_pairs).T
        result.roi_frames = len(roi_pairs)
        result.roi_conf_mean = round(float(zoomed.mean()), 3)
        result.roi_conf_gain = round(float(zoomed.mean() - full.mean()), 3)
        result.roi_latency_ms_mean = round(float(np.mean(roi_lat)), 2)
        result.roi_usable_pct = round(100.0 * roi_usable / len(roi_pairs), 1)
        if full_failures:
            result.roi_recovered_pct = round(100.0 * recovered / full_failures, 1)
    return result


def iter_source(
    path: Path, *, repeat: int, fps: float | None, resize_width: int | None
) -> Iterator[Frame]:
    if path.suffix.lower() in IMAGE_SUFFIXES:
        raw = cv2.imread(str(path))
        if raw is None:
            raise FileNotFoundError(path)
        image = np.asarray(raw, dtype=np.uint8)
        if resize_width:
            image = resize_to_width(image, resize_width)
        for i in range(repeat):
            yield Frame(idx=i, t=i / 30.0, image=np.asarray(image, dtype=np.uint8))
        return
    with FrameSource(path, target_fps=fps) as src:
        for frame in src:
            if resize_width:
                frame = Frame(frame.idx, frame.t, resize_to_width(frame.image, resize_width))
            yield frame


def to_markdown(results: Sequence[BenchmarkResult]) -> str:
    head = (
        "| model | source | frames | p50 ms | p95 ms | pose FPS | usable % | conf | "
        "jitter px | jitter % torso | ROI conf | ROI gain | ROI usable % | ROI recovered % | "
        "ROI ms |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = [
        f"| {r.model} | {Path(r.source).name} | {r.frames} | {r.latency_ms_p50} | "
        f"{r.latency_ms_p95} | {r.pose_fps} | {r.usable_pct} | {r.required_conf_mean} | "
        f"{r.jitter_px_median} | {r.jitter_torso_pct} | {r.roi_conf_mean} | "
        f"{r.roi_conf_gain} | {r.roi_usable_pct} | {r.roi_recovered_pct} | "
        f"{r.roi_latency_ms_mean} |"
        for r in results
    ]
    return head + "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("sources", nargs="+", type=Path, help="video or image files")
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--roi", action="store_true", help="also run ROI re-analysis per frame")
    p.add_argument("--fps", type=float, help="downsample videos to this processing FPS")
    p.add_argument("--resize-width", type=int, help="shrink frames (lower resolution)")
    p.add_argument("--repeat", type=int, default=50, help="repeats for image sources")
    p.add_argument("--out", type=Path, default=CONFIG_DIR.parent / "eval" / "benchmarks")
    args = p.parse_args(argv)

    cfg = load_config()
    required = [f"{cfg.exercise.side}_{j}" for j in ("hip", "shoulder", "elbow", "wrist")]
    results: list[BenchmarkResult] = []
    for model in args.models:
        estimator = create_estimator(cfg, model)
        for src in args.sources:
            first = next(iter_source(src, repeat=1, fps=None, resize_width=args.resize_width))
            for _ in range(3):  # warm-up: first inferences include DNN graph setup
                estimator.estimate(first.image, 0.0, 0)
            frames = iter_source(
                src, repeat=args.repeat, fps=args.fps, resize_width=args.resize_width
            )
            results.append(
                run_benchmark(
                    estimator,
                    frames,
                    required=required,
                    visibility_threshold=cfg.exercise.visibility_threshold,
                    source=str(src),
                    roi=args.roi,
                    roi_padding=cfg.pose.roi_padding,
                    roi_scale=cfg.exercise.tools.reanalyze_roi_scale,
                )
            )

    table = to_markdown(results)
    print(table)
    env = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "engine": cfg.pose.engine,
        "resize_width": args.resize_width,
        "side": cfg.exercise.side,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_file = args.out / f"pose_{platform.machine().lower()}_{stamp}.json"
    out_file.write_text(
        json.dumps({"env": env, "results": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    print(f"\nsaved {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
