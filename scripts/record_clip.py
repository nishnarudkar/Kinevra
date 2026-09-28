"""Record an evaluation / sample clip from the webcam with a metadata JSON sidecar.

Only record yourself or consenting volunteers (PROJECT.md §13). Clips go to data/clips/,
which is git-ignored; upload sample clips to S3 later.

Example:
    uv run python scripts/record_clip.py --name good_01 --consent --lighting normal \
        --distance-m 2.0 --camera-angle-deg 0 --seconds 40 --notes "10 reps, good form"

Keys: q / Esc stop early.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2

import kinevra
from kinevra.config import load_config
from kinevra.schemas import ClipMetadata
from kinevra.vision.capture import CaptureError, FrameSource
from kinevra.vision.fps import FpsMeter
from kinevra.vision.overlay import draw_countdown, draw_hud, draw_recording
from kinevra.vision.preprocess import mirror_for_display
from kinevra.vision.quality import assess_frame
from kinevra.vision.recording import ClipWriter

WINDOW = "Kinevra — record"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--name", required=True, help="clip name, e.g. fatigue_01")
    p.add_argument("--out-dir", type=Path, default=Path("data/clips"))
    p.add_argument("--camera", type=int, help="webcam index (default from config)")
    p.add_argument("--seconds", type=float, default=60.0, help="max recording length")
    p.add_argument("--countdown", type=int, default=3)
    p.add_argument("--side", choices=["left", "right"], help="exercising arm (default config)")
    p.add_argument("--lighting", choices=["bright", "normal", "dim", "backlit"])
    p.add_argument("--distance-m", type=float)
    p.add_argument("--camera-angle-deg", type=float)
    p.add_argument("--notes")
    p.add_argument(
        "--consent",
        action="store_true",
        help="confirm the subject is you or a consenting volunteer (required)",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.consent:
        print(
            "error: pass --consent to confirm the subject consented to recording", file=sys.stderr
        )
        return 2
    video_path = args.out_dir / f"{args.name}.mp4"
    meta_path = video_path.with_suffix(".json")
    if video_path.exists() and not args.overwrite:
        print(f"error: {video_path} exists (use --overwrite)", file=sys.stderr)
        return 2

    cfg = load_config()
    camera = args.camera if args.camera is not None else cfg.capture.camera_index
    try:
        source = FrameSource(
            camera,
            width=cfg.capture.width,
            height=cfg.capture.height,
            target_fps=cfg.capture.target_fps,
        )
    except CaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    meter = FpsMeter()
    stopped = False
    with source, ClipWriter(video_path, source.fps, (source.width, source.height)) as writer:
        countdown_end = time.monotonic() + args.countdown
        rec_start: float | None = None
        for frame in source:
            now = time.monotonic()
            view = mirror_for_display(frame.image)
            if now < countdown_end:  # preview + countdown, nothing recorded yet
                view = draw_countdown(view, int(countdown_end - now) + 1)
            else:
                if rec_start is None:
                    rec_start = now
                writer.write(frame.image)  # raw, unmirrored, unannotated
                meter.tick(now)
                quality = assess_frame(frame.image, cfg.quality)
                elapsed = now - rec_start
                view = draw_recording(
                    draw_hud(
                        view, [f"{args.name}", f"{elapsed:5.1f}s / {args.seconds:.0f}s"], quality
                    ),
                    "REC",
                )
                if elapsed >= args.seconds:
                    break
            cv2.imshow(WINDOW, view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                stopped = True
                break
        cv2.destroyAllWindows()
        frames = writer.frames

    if frames == 0:
        video_path.unlink(missing_ok=True)
        print("error: nothing recorded", file=sys.stderr)
        return 1

    meta = ClipMetadata(
        name=args.name,
        video_file=video_path.name,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source=f"camera:{camera}",
        width=source.width,
        height=source.height,
        fps_nominal=source.fps,
        fps_measured=round(meter.mean_fps, 2),
        frames=frames,
        duration_s=round(meter.elapsed_s, 2),
        side=args.side or cfg.exercise.side,
        consent=True,
        lighting=args.lighting,
        distance_m=args.distance_m,
        camera_angle_deg=args.camera_angle_deg,
        notes=args.notes,
        kinevra_version=kinevra.__version__,
        opencv_version=cv2.__version__,
    )
    meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"saved {video_path} ({frames} frames, {meta.fps_measured} fps measured)"
        f"{' — stopped early' if stopped else ''}"
    )
    print(f"saved {meta_path}")
    if abs(meta.fps_measured - meta.fps_nominal) > 0.2 * meta.fps_nominal:
        print(
            f"warning: measured FPS {meta.fps_measured} differs from nominal "
            f"{meta.fps_nominal}; playback speed will be off. Use fps_measured for timing.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
