"""Replay a clip through the full pipeline + agent and print every decision with its trace.

    uv run python scripts/replay.py data/clips/sample_fatigue.mp4
    uv run python scripts/replay.py clip.mp4 --llm policy --quiet

LLM: `policy` (deterministic, no AWS) now; `bedrock` arrives in Phase 6. Decisions, traces,
review events and snapshots are saved under data/sessions/<session_id>/.
Look for "<== CHANGED ASSESSMENT": an OpenCV tool result that changed the agent's decision.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from kinevra.agent.llm import PolicyLLM
from kinevra.agent.trace import format_trace
from kinevra.config import load_config
from kinevra.pipeline.clip import analyze_clip, default_store


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("clip", type=Path)
    p.add_argument("--llm", choices=["policy"], default="policy")
    p.add_argument("--session-id")
    p.add_argument("--quiet", action="store_true", help="decisions only, no traces")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles (cp1252) lack some symbols
        sys.stdout.reconfigure(errors="replace")
    if not args.clip.is_file():
        print(f"error: {args.clip} not found", file=sys.stderr)
        return 2

    cfg = load_config()
    store = default_store(cfg)
    result = analyze_clip(args.clip, cfg, llm=PolicyLLM(), store=store, session_id=args.session_id)

    for d in result.decisions:
        head = f"rep {d.rep_index}" if d.rep_index is not None else "quality check"
        llm = f"reasoned by {args.llm}" if d.llm_used else "triage only"
        print(f"\n== {head}: {d.action.value}  [{llm}, {d.total_latency_ms} ms]")
        print(f"   rationale: {d.rationale}")
        if d.feedback_text:
            print(f'   says: "{d.feedback_text}"')
        for o in d.guardrail_overrides:
            print(f"   override: {o}")
        if not args.quiet:
            print(format_trace(d.trace))

    actions = Counter(d.action.value for d in result.decisions)
    tools = Counter(s.tool for d in result.decisions for s in d.trace if s.node == "tool")
    changed = sum(s.changed_assessment for d in result.decisions for s in d.trace)
    print("\n--- summary ---")
    print(
        f"session {result.session_id}: {len(result.reps)} reps, "
        f"{len(result.decisions)} decisions, timings {result.timings_s}"
    )
    print(f"actions: {dict(actions)}")
    print(f"tool calls: {dict(tools)}; OpenCV results that changed the assessment: {changed}")
    print(f"review events: {len(result.review_events)}")
    print(f"saved: {', '.join(str(store.root / k) for k in result.artifact_keys)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
