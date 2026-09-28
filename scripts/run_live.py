"""Live viewer: capture → quality check → frame buffer → pose (OpenCV 5 DNN) → HUD.

Examples:
    uv run python scripts/run_live.py                      # webcam 0
    uv run python scripts/run_live.py --video clip.mp4     # same path on a file
    uv run python scripts/run_live.py --record data/clips/test.mp4
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
from kinevra.pose.base import COCO17_EDGES, DnnPoseEstimator
from kinevra.pose.factory import MODEL_NAMES, create_estimator
from kinevra.schemas import Landmark
from kinevra.vision.buffer import FrameBuffer
from kinevra.vision.capture import CaptureError, FrameSource
from kinevra.vision.fps import FpsMeter
from kinevra.vision.overlay import AMBER, draw_hud, draw_recording, draw_skeleton
from kinevra.vision.preprocess import mirror_for_display, roi_from_points
from kinevra.vision.quality import FrameQuality, assess_frame
from kinevra.vision.recording import ClipWriter

WINDOW = "Kinevra — live"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video", type=Path, help="video file instead of the webcam")
    src.add_argument("--camera", type=int, help="webcam index (default from config)")
    p.add_argument("--record", type=Path, help="save raw (unannotated) frames to this file")
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
    side = cfg.exercise.side
    required = [f"{side}_{j}" for j in ("hip", "shoulder", "elbow", "wrist")]
    pose_ms: list[float] = []
    roi_gain: list[float] = []
    writer: ClipWriter | None = None
    if args.record:
        writer = ClipWriter(
            args.record, frame_source.fps, (frame_source.width, frame_source.height)
        )

    try:
        for frame in frame_source:
            quality = assess_frame(frame.image, cfg.quality)
            buffer.push(frame)
            if writer:
                writer.write(frame.image)
            fps = meter.tick(time.perf_counter())
            quality_sum += quality.score
            pose = None
            roi_line = ""
            roi_box = None
            if estimator is not None:
                pose = estimator.estimate(frame.image, frame.t, frame.idx, quality=quality)
                pose_ms.append(estimator.last_latency_ms)
                flag_counts.update(pose.quality_flags)
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
            else:
                flag_counts.update(quality.flags)

            if not args.headless:
                view = frame.image
                if pose is not None:
                    view = draw_skeleton(
                        view,
                        pose.landmarks,
                        COCO17_EDGES,
                        visibility_threshold=cfg.exercise.visibility_threshold,
                        highlight_side=side,
                    )
                if roi_box is not None:
                    view = view.copy()
                    x, y, bw, bh = roi_box.x, roi_box.y, roi_box.w, roi_box.h
                    cv2.rectangle(view, (x, y), (x + bw, y + bh), AMBER, 1)
                if frame_source.is_live and not args.no_mirror:
                    view = mirror_for_display(view)  # display only; analysis stays unmirrored
                lines = [
                    f"fps {fps:5.1f}",
                    f"frame {frame.idx}  t {frame.t:6.2f}s",
                    f"buffer {len(buffer)} frames / {buffer.span_s:4.1f}s",
                    f"bright {quality.brightness:5.1f}  sharp {quality.sharpness:6.1f}",
                ]
                if pose is not None:
                    elbow = pose.landmarks.get(f"{side}_elbow")
                    lines.append(
                        f"{estimator.name if estimator else ''} {pose_ms[-1]:5.1f} ms"
                        f"  persons {pose.person_count}"
                    )
                    if elbow is not None:
                        lines.append(f"{side}_elbow y {elbow.y:.2f} vis {elbow.visibility:.2f}")
                    if roi_line:
                        lines.append(roi_line)
                    flags = [f for f in pose.quality_flags if f not in quality.flags]
                    quality = FrameQuality(
                        quality.brightness,
                        quality.contrast,
                        quality.sharpness,
                        quality.score,
                        [*quality.flags, *flags],
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
        "pose_model": estimator.name if estimator else None,
        "pose_ms_mean": round(sum(pose_ms) / len(pose_ms), 1) if pose_ms else None,
        "roi_conf_gain_mean": round(sum(roi_gain) / len(roi_gain), 3) if roi_gain else None,
    }


def _mean_vis(landmarks: dict[str, Landmark], names: list[str]) -> float:
    vals = [landmarks[n].visibility if n in landmarks else 0.0 for n in names]
    return sum(vals) / len(vals) if vals else 0.0


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
