"""Evaluate rep counting + rule baseline on hand-labelled clips (PROJECT.md Phase 4).

For every data/labels/<name>.json (see docs/data_protocol.md) the matching clip in data/clips/
is analysed in clip mode and compared with the labels. Writes eval/rep_counting.md.

    uv run python scripts/eval_reps.py
    uv run python scripts/eval_reps.py --labels data/labels --clips data/clips --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kinevra.config import load_config
from kinevra.evaluation.rep_eval import ClipScore, markdown_report, score_clip, summarise
from kinevra.pipeline.clip import analyze_clip
from kinevra.pose.factory import create_estimator
from kinevra.schemas import ClipLabel

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--labels", type=Path, default=ROOT / "data" / "labels")
    p.add_argument("--clips", type=Path, default=ROOT / "data" / "clips")
    p.add_argument("--out", type=Path, default=ROOT / "eval" / "rep_counting.md")
    p.add_argument("--verbose", action="store_true", help="print every rep and event")
    args = p.parse_args(argv)

    label_files = sorted(args.labels.glob("*.json"))
    if not label_files:
        print(f"no label files in {args.labels} (see docs/data_protocol.md)", file=sys.stderr)
        return 1
    cfg = load_config()
    estimator = create_estimator(cfg)
    scores: list[ClipScore] = []
    for lf in label_files:
        label = ClipLabel.model_validate_json(lf.read_text(encoding="utf-8"))
        clip = args.clips / label.clip
        if not clip.is_file():
            print(f"skip {lf.name}: clip {clip} not found", file=sys.stderr)
            continue
        if label.side != cfg.exercise.side:
            print(
                f"skip {lf.name}: labelled side {label.side} != config side "
                f"{cfg.exercise.side} (set KINEVRA_SIDE)",
                file=sys.stderr,
            )
            continue
        result = analyze_clip(clip, cfg, estimator=estimator)
        partials = sum(e.kind == "partial_rep" for e in result.events)
        score = score_clip(label, result.reps, partials)
        scores.append(score)
        status = "ok " if score.exact else "MISS"
        print(
            f"{status} {label.clip}: labelled {label.reps}, counted {len(result.reps)} "
            f"({result.timings_s})"
        )
        if args.verbose:
            for r in result.reps:
                print(
                    f"     rep {r.rep_index}: t {r.t_start:.1f}-{r.t_end:.1f}s ROM {r.rom:.0f} "
                    f"{r.rule_quality.value} {r.rule_reasons}"
                )
            for e in result.events:
                print(f"     event {e.kind} t {e.t_start:.1f}-{e.t_end:.1f}s {e.detail}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown_report(scores), encoding="utf-8")
    s = summarise(scores)
    exact = s["exact_count_pct"]
    print(
        f"\nexact count: {'n/a' if exact is None else f'{exact:.0f}%'} of {len(scores)} clips"
        f"  ->  saved {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
