"""Live viewer (Phase 1): capture → quality check → frame buffer → HUD.

Examples:
    uv run python scripts/run_live.py                      # webcam 0
    uv run python scripts/run_live.py --video clip.mp4     # same path on a file
    uv run python scripts/run_live.py --record data/clips/test.mp4
    uv run python scripts/run_live.py --video clip.mp4 --headless   # no window, stats only

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
from kinevra.vision.buffer import FrameBuffer
from kinevra.vision.capture import CaptureError, FrameSource
from kinevra.vision.fps import FpsMeter
from kinevra.vision.overlay import draw_hud, draw_recording
from kinevra.vision.preprocess import mirror_for_display
from kinevra.vision.quality import assess_frame
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
            flag_counts.update(quality.flags)

            if not args.headless:
                view = frame.image
                if frame_source.is_live and not args.no_mirror:
                    view = mirror_for_display(view)  # display only; analysis stays unmirrored
                lines = [
                    f"fps {fps:5.1f}",
                    f"frame {frame.idx}  t {frame.t:6.2f}s",
                    f"buffer {len(buffer)} frames / {buffer.span_s:4.1f}s",
                    f"bright {quality.brightness:5.1f}  sharp {quality.sharpness:6.1f}",
                ]
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
