"""Held-pose angle check against a phone inclinometer (PROJECT.md Phase 3, §13.2).

`scripts/sanity_angles.py` collects the samples; this module summarises them into the
`eval/sanity_angles.md` table (also reused for the ROM-error metric in Phase 8).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class HeldPose:
    label: str  # e.g. "90°"
    reference_deg: float  # inclinometer reading
    samples: Sequence[float]  # measured abduction over the hold (degrees)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples)

    @property
    def std(self) -> float:
        return statistics.pstdev(self.samples) if len(self.samples) > 1 else 0.0

    @property
    def error(self) -> float:
        return self.mean - self.reference_deg


def summarise(poses: Sequence[HeldPose]) -> dict[str, float]:
    if not poses:
        return {"n": 0.0}
    errors = [abs(p.error) for p in poses]
    return {
        "n": float(len(poses)),
        "mae_deg": statistics.fmean(errors),
        "max_abs_error_deg": max(errors),
        "mean_jitter_std_deg": statistics.fmean(p.std for p in poses),
    }


def markdown_table(poses: Sequence[HeldPose], *, context: str = "") -> str:
    lines = [
        "# Angle sanity check (held poses vs phone inclinometer)",
        "",
        "Shoulder abduction = hip → shoulder → elbow, frontal view. Measured value = mean of the",
        "smoothed live angle over a ~2 s hold; std = jitter during the hold.",
        "",
    ]
    if context:
        lines += [context, ""]
    lines += [
        "| Pose | Inclinometer (°) | Measured mean (°) | Std (°) | Error (°) | Samples |",
        "|---|---|---|---|---|---|",
    ]
    for p in poses:
        lines.append(
            f"| {p.label} | {p.reference_deg:.1f} | {p.mean:.1f} | {p.std:.1f} | "
            f"{p.error:+.1f} | {len(p.samples)} |"
        )
    s = summarise(poses)
    if poses:
        lines += [
            "",
            f"**Mean absolute error:** {s['mae_deg']:.1f}° · **max:** {s['max_abs_error_deg']:.1f}°"
            f" · **mean hold jitter (std):** {s['mean_jitter_std_deg']:.1f}°",
        ]
    lines += [
        "",
        "Expected bands (PROJECT.md Phase 3): arm at side 0-20°, horizontal 85-95°, "
        "overhead 160-180°.",
        "",
    ]
    return "\n".join(lines)
