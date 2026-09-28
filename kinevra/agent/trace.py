"""Structured decision trace: one TraceStep per graph node or tool call."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from kinevra.schemas import TraceStep

MAX_SUMMARY_ITEMS = 12  # keep stored summaries small (lists truncated)


def summarise(obj: Any) -> Any:
    """Compact, JSON-safe copy of a tool input/output for the trace."""
    if isinstance(obj, dict):
        return {str(k): summarise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = [summarise(v) for v in obj[:MAX_SUMMARY_ITEMS]]
        if len(obj) > MAX_SUMMARY_ITEMS:
            items.append(f"... {len(obj) - MAX_SUMMARY_ITEMS} more")
        return items
    if isinstance(obj, float):
        return round(obj, 3)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


class TraceRecorder:
    def __init__(self) -> None:
        self.steps: list[TraceStep] = []

    def add(
        self,
        node: str,
        *,
        tool: str | None = None,
        tool_input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        changed_assessment: bool = False,
        latency_ms: float = 0.0,
    ) -> TraceStep:
        step = TraceStep(
            step=len(self.steps) + 1,
            node=node,
            tool=tool,
            tool_input=summarise(tool_input) if tool_input is not None else None,
            tool_output_summary=summarise(output) if output is not None else None,
            changed_assessment=changed_assessment,
            latency_ms=round(latency_ms),
        )
        self.steps.append(step)
        return step

    @contextmanager
    def timed(self) -> Iterator[dict[str, float]]:
        """`with rec.timed() as t: ...` then use t["ms"]."""
        box = {"ms": 0.0}
        start = time.perf_counter()
        try:
            yield box
        finally:
            box["ms"] = (time.perf_counter() - start) * 1000.0


def format_trace(steps: list[TraceStep]) -> str:
    """Human-readable one line per step (replay output, logs)."""
    lines = []
    for s in steps:
        marker = "  <== CHANGED ASSESSMENT" if s.changed_assessment else ""
        what = s.tool or s.node
        detail = s.tool_output_summary or {}
        short = ", ".join(f"{k}={v}" for k, v in list(detail.items())[:6])
        lines.append(f"  {s.step:>2}. [{s.node}] {what} ({s.latency_ms} ms) {short}{marker}")
    return "\n".join(lines)
