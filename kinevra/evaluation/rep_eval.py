"""Compare pipeline output with hand labels: rep count accuracy and per-rep rule agreement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from kinevra.schemas import ClipLabel, RepMetrics, RepQuality


@dataclass
class ClipScore:
    clip: str
    labelled_reps: int
    counted_reps: int
    labelled_partials: int
    logged_partials: int
    exact: bool
    # per-rep DEVIATION agreement, only when per-rep labels exist and the counts match
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    uncertain: int = 0
    reason_hits: dict[str, int] = field(default_factory=dict)  # labelled deviation → detected


def score_clip(label: ClipLabel, reps: Sequence[RepMetrics], partial_events: int) -> ClipScore:
    s = ClipScore(
        label.clip,
        label.reps,
        len(reps),
        label.partial_reps,
        partial_events,
        exact=len(reps) == label.reps,
    )
    if label.rep_labels and len(label.rep_labels) == len(reps):
        for truth, rep in zip(label.rep_labels, reps, strict=True):
            if rep.rule_quality is RepQuality.UNCERTAIN:
                s.uncertain += 1
                continue
            predicted = rep.rule_quality is RepQuality.DEVIATION
            actual = truth.quality == "DEVIATION"
            s.tp += predicted and actual
            s.fp += predicted and not actual
            s.fn += actual and not predicted
            s.tn += not predicted and not actual
            codes = {r.split(":")[0] for r in rep.rule_reasons}
            for dev in truth.deviations:
                s.reason_hits[dev] = s.reason_hits.get(dev, 0) + (dev in codes)
    return s


def summarise(scores: Sequence[ClipScore]) -> dict[str, float | None]:
    n = len(scores)
    tp, fp, fn = (sum(getattr(s, k) for s in scores) for k in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "clips": float(n),
        "exact_count_pct": 100.0 * sum(s.exact for s in scores) / n if n else None,
        "mean_abs_count_error": (
            sum(abs(s.counted_reps - s.labelled_reps) for s in scores) / n if n else None
        ),
        "deviation_precision": precision,
        "deviation_recall": recall,
        "deviation_f1": f1,
    }


def markdown_report(scores: Sequence[ClipScore]) -> str:
    s = summarise(scores)

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.0%}"

    lines = [
        "# Rep counting and rule baseline vs hand labels",
        "",
        "| Clip | Labelled reps | Counted | Exact | Labelled partials | Logged partials | "
        "DEV TP/FP/FN/TN | Uncertain |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in scores:
        lines.append(
            f"| {c.clip} | {c.labelled_reps} | {c.counted_reps} | {'yes' if c.exact else 'NO'} | "
            f"{c.labelled_partials} | {c.logged_partials} | {c.tp}/{c.fp}/{c.fn}/{c.tn} | "
            f"{c.uncertain} |"
        )
    exact = s["exact_count_pct"]
    lines += [
        "",
        f"**Exact rep count:** {'n/a' if exact is None else f'{exact:.0f}%'} of "
        f"{int(s['clips'] or 0)} clips (target ≥ 90%)",
        f"**DEVIATION precision / recall / F1:** {pct(s['deviation_precision'])} / "
        f"{pct(s['deviation_recall'])} / {pct(s['deviation_f1'])} (UNCERTAIN reps excluded)",
        "",
    ]
    return "\n".join(lines)
