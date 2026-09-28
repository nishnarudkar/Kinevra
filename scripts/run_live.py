"""Live viewer: capture → quality → frame buffer → pose (OpenCV 5 DNN) → features → HUD.

Examples:
    uv run python scripts/run_live.py                      # webcam 0
    uv run python scripts/run_live.py --video clip.mp4     # same path on a file
    uv run python scripts/run_live.py --record data/clips/test.mp4
    uv run python scripts/run_live.py --csv eval/features.csv       # per-frame features
    uv run python scripts/run_live.py --video clip.mp4 --headless   # no window, stats only
    uv run python scripts/run_live.py --pose mediapipe --roi        # other model + ROI check

Keys: q / Esc quit. Prints a JSON summary (mean FPS, flag counts) at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from kinevra.config import load_config
from kinevra.movement.export import FeatureCsvWriter
from kinevra.movement.features import FeatureExtractor, raw_features
from kinevra.pose.base import COCO17_EDGES, DnnPoseEstimator
from kinevra.pose.factory import MODEL_NAMES, create_estimator
from kinevra.schemas import FrameFeatures, Landmark, PoseFrame
from kinevra.vision.buffer import FrameBuffer
from kinevra.vision.capture import CaptureError, FrameSource
from kinevra.vision.fps import FpsMeter
from kinevra.vision.overlay import (
    AMBER,
    draw_angle_arc,
    draw_hud,
    draw_recording,
    draw_skeleton,
    mirror_landmarks,
)
from kinevra.vision.preprocess import Roi, mirror_for_display, roi_from_points
from kinevra.vision.quality import FrameQuality, assess_frame
from kinevra.vision.recording import ClipWriter
from kinevra.vision.types import Image

WINDOW = "Kinevra — live"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video", type=Path, help="video file instead of the webcam")
    src.add_argument("--camera", type=int, help="webcam index (default from config)")
    p.add_argument("--record", type=Path, help="save raw (unannotated) frames to this file")
    p.add_argument("--csv", type=Path, help="write per-frame raw + smoothed features here")
    p.add_argument("--fps", type=float, help="processing FPS for video files (downsample)")
    p.add_argument("--max-frames", type=int, help="stop after this many frames")
    p.add_argument("--headless", action="store_true", help="no window; print stats only")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the webcam preview")
    p.add_argument("--config", type=Path, help="path to default.yaml")
    p.add_argument(
        "--pose", choices=[*MODEL_NAMES, "none"], help="pose model (default from config)"
    )
    p.add_argument("--roi", action="store_true", help="also run ROI re-analysis on the arm")
    return p.parse_args(argv)


def _fmt(value: float | None, spec: str, unit: str = "") -> str:
    return "--" if value is None else f"{value:{spec}}{unit}"


def _mean_vis(landmarks: dict[str, Landmark], names: list[str]) -> float:
    vals = [landmarks[n].visibility if n in landmarks else 0.0 for n in names]
    return sum(vals) / len(vals) if vals else 0.0


def _render(
    image: Image,
    pose: PoseFrame | None,
    feat: FrameFeatures | None,
    roi_box: Roi | None,
    *,
    mirror: bool,
    side: str,
    visibility_threshold: float,
) -> Image:
    """Mirror first (display only), then draw landmarks in display coordinates."""
    view = mirror_for_display(image) if mirror else image.copy()
    if pose is None:
        return view
    w, h = pose.image_width, pose.image_height
    lms = mirror_landmarks(pose.landmarks) if mirror else pose.landmarks
    view = draw_skeleton(
        view, lms, COCO17_EDGES, visibility_threshold=visibility_threshold, highlight_side=side
    )
    if roi_box is not None:
        x0 = w - (roi_box.x + roi_box.w) if mirror else roi_box.x
        cv2.rectangle(view, (x0, roi_box.y), (x0 + roi_box.w, roi_box.y + roi_box.h), AMBER, 1)
    names = (f"{side}_hip", f"{side}_shoulder", f"{side}_elbow")
    if (
        feat is not None
        and feat.shoulder_abduction_deg is not None
        and all(n in lms for n in names)
    ):
        hip, sh, el = ((lms[n].x * w, lms[n].y * h) for n in names)
        view = draw_angle_arc(view, sh, hip, el, feat.shoulder_abduction_deg)
    return view


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    cap_cfg = cfg.capture
    source: int | Path = (
        args.video
        if args.video
        else (args.camera if args.camera is not None else cap_cfg.camera_index)
    )
    frame_source = FrameSource(
        source,
        width=cap_cfg.width,
        height=cap_cfg.height,
        target_fps=cap_cfg.target_fps if args.video is None else args.fps,
    )
    buffer = FrameBuffer(cfg.buffer.seconds, cfg.buffer.downscale_width)
    meter = FpsMeter()
    flag_counts: Counter[str] = Counter()
    quality_sum = 0.0
    pose_model = args.pose or cfg.pose.model
    estimator: DnnPoseEstimator | None = (
        None if pose_model == "none" else create_estimator(cfg, pose_model)
    )
    extractor = FeatureExtractor(cfg.exercise)
    side = cfg.exercise.side
    vis_thr = cfg.exercise.visibility_threshold
    required = [f"{side}_{j}" for j in ("hip", "shoulder", "elbow", "wrist")]
    pose_ms: list[float] = []
    roi_gain: list[float] = []
    abductions: list[float] = []
    writer = (
        ClipWriter(args.record, frame_source.fps, (frame_source.width, frame_source.height))
        if args.record
        else None
    )
    csv_writer = FeatureCsvWriter(args.csv) if args.csv and estimator is not None else None
    mirror = frame_source.is_live and not args.no_mirror

    try:
        for frame in frame_source:
            quality = assess_frame(frame.image, cfg.quality)
            buffer.push(frame)
            if writer:
                writer.write(frame.image)
            fps = meter.tick(time.perf_counter())
            quality_sum += quality.score
            pose: PoseFrame | None = None
            feat: FrameFeatures | None = None
            roi_line = ""
            roi_box: Roi | None = None
            if estimator is None:
                flag_counts.update(quality.flags)
            else:
                pose = estimator.estimate(frame.image, frame.t, frame.idx, quality=quality)
                pose_ms.append(estimator.last_latency_ms)
                flag_counts.update(pose.quality_flags)
                feat = extractor.update(pose)
                if feat.shoulder_abduction_deg is not None:
                    abductions.append(feat.shoulder_abduction_deg)
                if csv_writer:
                    raw = raw_features(pose, side, vis_thr)
                    csv_writer.write(raw, feat, pose.quality_flags)
                pts = [
                    (pose.landmarks[n].x, pose.landmarks[n].y)
                    for n in required
                    if n in pose.landmarks
                ]
                if args.roi and pts:
                    h, w = frame.image.shape[:2]
                    roi_box = roi_from_points(pts, w, h, padding=cfg.pose.roi_padding)
                    zoom = estimator.estimate(frame.image, frame.t, frame.idx, roi=roi_box)
                    full_c = _mean_vis(pose.landmarks, required)
                    roi_c = _mean_vis(zoom.landmarks, required)
                    roi_gain.append(roi_c - full_c)
                    roi_line = f"arm conf full {full_c:.2f} / roi {roi_c:.2f}"

            if not args.headless:
                view = _render(
                    frame.image,
                    pose,
                    feat,
                    roi_box,
                    mirror=mirror,
                    side=side,
                    visibility_threshold=vis_thr,
                )
                lines = [
                    f"fps {fps:5.1f}   frame {frame.idx}  t {frame.t:6.2f}s",
                    f"buffer {len(buffer)} frames / {buffer.span_s:4.1f}s",
                ]
                if pose is not None and feat is not None and estimator is not None:
                    lines += [
                        f"{estimator.name} {pose_ms[-1]:5.1f} ms  persons {pose.person_count}",
                        f"{side} abduction {_fmt(feat.shoulder_abduction_deg, '5.1f')}"
                        f"  vel {_fmt(feat.angular_velocity_dps, '+5.0f', '/s')}",
                        f"elbow flex {_fmt(feat.elbow_flexion_deg, '4.0f')}"
                        f"  lean {_fmt(feat.trunk_lean_deg, '+4.1f')}",
                        f"shrug {_fmt(feat.shoulder_elevation, '+.2f')}"
                        f"  conf {feat.confidence:.2f}",
                    ]
                    if roi_line:
                        lines.append(roi_line)
                    extra = [f for f in pose.quality_flags if f not in quality.flags]
                    quality = FrameQuality(
                        quality.brightness,
                        quality.contrast,
                        quality.sharpness,
                        quality.score,
                        [*quality.flags, *extra],
                    )
                view = draw_hud(view, lines, quality)
                if writer:
                    view = draw_recording(view)
                cv2.imshow(WINDOW, view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and meter.frames >= args.max_frames:
                break
    finally:
        frame_source.close()
        if writer:
            writer.close()
        if csv_writer:
            csv_writer.close()
        if not args.headless:
            cv2.destroyAllWindows()

    n = meter.frames
    return {
        "source": str(source),
        "resolution": f"{frame_source.width}x{frame_source.height}",
        "frames": n,
        "elapsed_s": round(meter.elapsed_s, 2),
        "mean_fps": round(meter.mean_fps, 1),
        "mean_quality": round(quality_sum / n, 3) if n else None,
        "flag_frames": dict(flag_counts),
        "recorded": str(args.record) if args.record else None,
        "csv": str(args.csv) if csv_writer else None,
        "pose_model": estimator.name if estimator else None,
        "pose_ms_mean": round(sum(pose_ms) / len(pose_ms), 1) if pose_ms else None,
        "roi_conf_gain_mean": round(sum(roi_gain) / len(roi_gain), 3) if roi_gain else None,
        "abduction_min_max": (
            [round(min(abductions), 1), round(max(abductions), 1)] if abductions else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run(args)
    except CaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
