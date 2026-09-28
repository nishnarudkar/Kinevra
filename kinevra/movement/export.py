"""Per-frame feature CSV (raw + smoothed) for plotting and debugging."""

from __future__ import annotations

import csv
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from kinevra.movement.features import RawFeatures
from kinevra.schemas import FrameFeatures

FIELDS = (
    "frame_idx",
    "t",
    "raw_abduction_deg",
    "shoulder_abduction_deg",
    "raw_elbow_flexion_deg",
    "elbow_flexion_deg",
    "raw_trunk_lean_deg",
    "trunk_lean_deg",
    "shoulder_elevation",
    "angular_velocity_dps",
    "confidence",
    "quality_flags",
)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def feature_row(raw: RawFeatures, feat: FrameFeatures, flags: list[str]) -> dict[str, str]:
    return {
        "frame_idx": str(feat.frame_idx),
        "t": f"{feat.t:.4f}",
        "raw_abduction_deg": _fmt(raw.abduction),
        "shoulder_abduction_deg": _fmt(feat.shoulder_abduction_deg),
        "raw_elbow_flexion_deg": _fmt(raw.elbow_flexion),
        "elbow_flexion_deg": _fmt(feat.elbow_flexion_deg),
        "raw_trunk_lean_deg": _fmt(raw.trunk_lean),
        "trunk_lean_deg": _fmt(feat.trunk_lean_deg),
        "shoulder_elevation": _fmt(feat.shoulder_elevation),
        "angular_velocity_dps": _fmt(feat.angular_velocity_dps),
        "confidence": f"{feat.confidence:.4f}",
        "quality_flags": ";".join(flags),
    }


class FeatureCsvWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=FIELDS)
        self._writer.writeheader()
        self.rows = 0

    def write(self, raw: RawFeatures, feat: FrameFeatures, flags: list[str]) -> None:
        self._writer.writerow(feature_row(raw, feat, flags))
        self.rows += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> FeatureCsvWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_feature_csv(path: str | Path) -> list[dict[str, Any]]:
    """Rows with numbers parsed (empty cells → None) and flags as a list."""
    out: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key == "quality_flags":
                    parsed[key] = [f for f in value.split(";") if f]
                elif key == "frame_idx":
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value) if value != "" else None
            out.append(parsed)
    return out
