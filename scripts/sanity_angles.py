"""Held-pose sanity check: measured shoulder abduction vs a phone inclinometer.

1. Stand facing the camera, full upper body in view; strap/hold a phone on the upper arm with
   an inclinometer app (0° = arm hanging down).
2. Hold a pose and press its key; hold still ~2 s while it samples:
       1 → 0° (arm at side)   2 → 45°   3 → 90° (horizontal)   4 → 135°   5 → 170° (overhead)
   Repeat keys as often as you like; every hold is kept.
3. Press q. For each hold the terminal asks for the inclinometer reading (Enter = target).
4. Results go to eval/sanity_angles.md.

    uv run python scripts/sanity_angles.py [--camera 1] [--seconds 2]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from kinevra.config import load_config
from kinevra.evaluation.angle_sanity import HeldPose, markdown_table
from kinevra.movement.features import FeatureExtractor
from kinevra.pose.base import COCO17_EDGES
from kinevra.pose.factory import create_estimator
from kinevra.vision.capture import CaptureError, FrameSource
from kinevra.vision.overlay import draw_hud, draw_skeleton, mirror_landmarks
from kinevra.vision.preprocess import mirror_for_display

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {ord("1"): 0.0, ord("2"): 45.0, ord("3"): 90.0, ord("4"): 135.0, ord("5"): 170.0}
WINDOW = "Kinevra — angle sanity check"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--camera", type=int)
    p.add_argument("--seconds", type=float, default=2.0, help="hold duration per capture")
    p.add_argument("--out", type=Path, default=ROOT / "eval" / "sanity_angles.md")
    args = p.parse_args(argv)

    cfg = load_config()
    est = create_estimator(cfg)
    fx = FeatureExtractor(cfg.exercise)
    side = cfg.exercise.side
    cam = args.camera if args.camera is not None else cfg.capture.camera_index
    try:
        src = FrameSource(
            cam,
            width=cfg.capture.width,
            height=cfg.capture.height,
            target_fps=cfg.capture.target_fps,
        )
    except CaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    holds: list[tuple[float, list[float]]] = []  # (target, samples)
    active: tuple[float, float, list[float]] | None = None  # (target, end time, samples)
    with src:
        for frame in src:
            pose = est.estimate(frame.image, frame.t, frame.idx)
            feat = fx.update(pose)
            angle = feat.shoulder_abduction_deg
            now = time.monotonic()
            if active is not None:
                target, end, samples = active
                if angle is not None and feat.confidence >= cfg.exercise.visibility_threshold:
                    samples.append(angle)
                if now >= end:
                    holds.append((target, samples))
                    active = None

            view = draw_skeleton(
                mirror_for_display(frame.image),
                mirror_landmarks(pose.landmarks),
                COCO17_EDGES,
                highlight_side=side,
            )
            status = (
                f"SAMPLING {active[0]:.0f} deg ... hold still"
                if active
                else "keys 1-5: 0/45/90/135/170 deg   q: finish"
            )
            lines = [
                f"{side} abduction {'--' if angle is None else f'{angle:5.1f}'} deg"
                f"   conf {feat.confidence:.2f}",
                status,
                f"captured {len(holds)}: " + ", ".join(f"{t:.0f}" for t, _ in holds[-6:]),
            ]
            cv2.imshow(WINDOW, draw_hud(view, lines, None))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in TARGETS and active is None:
                active = (TARGETS[key], now + args.seconds, [])
    cv2.destroyAllWindows()

    poses: list[HeldPose] = []
    for i, (target, samples) in enumerate(holds, start=1):
        if not samples:
            print(f"hold {i} ({target:.0f} deg): no confident samples, skipped")
            continue
        reply = input(
            f"hold {i}: target {target:.0f} deg, measured {sum(samples) / len(samples):.1f}"
            f" deg. Inclinometer reading [Enter = {target:.0f}]: "
        ).strip()
        try:
            reference = float(reply) if reply else target
        except ValueError:
            reference = target
        poses.append(HeldPose(f"{target:.0f}°", reference, samples))

    context = input("Setup notes (camera, distance, lighting) [optional]: ").strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown_table(poses, context=context), encoding="utf-8")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
