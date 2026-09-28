"""Hand-label repetitions in a clip (ground truth for scripts/eval_reps.py).

The system's own rep counter is deliberately NOT shown, so labels stay independent.
Watch the clip and press a key when each repetition finishes (arm back down):

    g  good rep                         l  rep with trunk lean
    r  rep with reduced ROM (half way)  e  rep with bent elbow
    s  rep with shoulder shrug          p  partial raise (NOT a rep)
    space  pause/resume    ,/.  step one frame back/forward when paused
    u  undo last key       q  finish and save

    uv run python scripts/label_reps.py data/clips/fatigue_01.mp4 [--labeller NN] [--speed 0.75]

Writes data/labels/<clip stem>.json (ClipLabel in kinevra/schemas.py).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from kinevra.config import load_config
from kinevra.schemas import ClipLabel, RepLabel
from kinevra.vision.capture import FrameSource
from kinevra.vision.overlay import draw_hud
from kinevra.vision.types import Frame

ROOT = Path(__file__).resolve().parents[1]
DEVIATION_KEYS = {
    ord("l"): "trunk_lean",
    ord("r"): "reduced_rom",
    ord("e"): "elbow_flexion",
    ord("s"): "shoulder_elevation",
}
WINDOW = "Kinevra — label reps"
HISTORY_FRAMES = 150  # ~5 s of frames kept for stepping back


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("clip", type=Path)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "labels")
    p.add_argument("--labeller")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    p.add_argument("--notes")
    args = p.parse_args(argv)

    source = FrameSource(args.clip)
    fps = source.native_fps
    history: deque[Frame] = deque(maxlen=HISTORY_FRAMES)  # for stepping back while paused
    marks: list[tuple[float, str]] = []  # (t, key label)
    back, paused = 0, False  # back = frames behind the newest decoded frame
    while True:
        if (back == 0 and not paused) or not history:
            frame = source.read()
            if frame is None:
                break
            history.append(frame)
        frame = history[-1 - back]
        reps = sum(1 for _, k in marks if k != "partial")
        partials = sum(1 for _, k in marks if k == "partial")
        lines = [
            f"t {frame.t:6.2f}s  {'PAUSED' if paused else ''}",
            f"marked reps {reps}   partials {partials}",
            "g good  l lean  r reduced ROM  e elbow  s shrug  p partial",
            "space pause  ,/. step  u undo  q save",
        ]
        cv2.imshow(WINDOW, draw_hud(frame.image, lines, None))
        started = time.perf_counter()
        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
            back = 0 if not paused else back
        elif key == ord("u") and marks:
            marks.pop()
        elif key == ord("g"):
            marks.append((frame.t, "good"))
        elif key == ord("p"):
            marks.append((frame.t, "partial"))
        elif key in DEVIATION_KEYS:
            marks.append((frame.t, DEVIATION_KEYS[key]))
        if paused:
            if key == ord(","):
                back = min(len(history) - 1, back + 1)
            elif key == ord("."):
                if back > 0:
                    back -= 1
                else:
                    nxt = source.read()
                    if nxt is not None:
                        history.append(nxt)
            continue
        wait = 1.0 / (fps * args.speed) - (time.perf_counter() - started)
        if wait > 0:
            time.sleep(wait)
    source.close()
    cv2.destroyAllWindows()

    rep_labels = [
        RepLabel(t=round(t, 3), quality="GOOD")
        if k == "good"
        else RepLabel(t=round(t, 3), quality="DEVIATION", deviations=[k])
        for t, k in marks
        if k != "partial"
    ]
    label = ClipLabel(
        clip=args.clip.name,
        side=load_config().exercise.side,
        reps=len(rep_labels),
        partial_reps=sum(1 for _, k in marks if k == "partial"),
        rep_labels=rep_labels,
        labeller=args.labeller,
        notes=args.notes,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.clip.stem}.json"
    out.write_text(label.model_dump_json(indent=2), encoding="utf-8")
    print(f"saved {out}: {label.reps} reps, {label.partial_reps} partials")
    return 0


if __name__ == "__main__":
    sys.exit(main())
