"""Offline pipeline on geometric poses: PoseFrame → features → reps → rules → evidence."""

import math

from kinevra.config import load_config
from kinevra.pipeline.clip import analyze_poses
from kinevra.schemas import PoseFrame, RepQuality
from tests.conftest import make_pose

FPS = 15.0


def _poses(peaks: list[float], lean_on: int | None = None) -> list[PoseFrame]:
    angles: list[tuple[float, float]] = []  # (abduction, lean)
    angles += [(15.0, 0.0)] * 15
    for k, peak in enumerate(peaks):
        n = 36  # 2.4 s
        for i in range(n):
            a = 15 + (peak - 15) * math.sin(math.pi * i / (n - 1)) ** 2
            angles.append((a, 14.0 * (a - 15) / (peak - 15) if k == lean_on else 0.0))
        angles += [(15.0, 0.0)] * 15
    return [
        make_pose(a, t=i / FPS, frame_idx=i, lean_deg=lean) for i, (a, lean) in enumerate(angles)
    ]


def test_analyze_poses_counts_and_judges() -> None:
    cfg = load_config(env={})
    result = analyze_poses(_poses([150, 150, 150, 150, 45, 100], lean_on=3), cfg, session_id="t1")
    assert [r.rep_index for r in result.reps] == [1, 2, 3, 4, 5]
    assert [e.kind for e in result.events] == ["partial_rep"]
    q = [r.rule_quality for r in result.reps]
    assert q == [RepQuality.GOOD] * 3 + [RepQuality.DEVIATION] * 2
    assert result.reps[3].rule_reasons[0].startswith("trunk_lean")
    assert result.reps[4].rule_reasons[0].startswith("reduced_rom")
    ev = result.evidence
    assert ev.mode == "clip" and ev.session_id == "t1" and ev.reps_completed == 5
    assert abs(ev.baseline_rom - 135) < 2  # type: ignore[operator]
    assert len(result.features) == len(result.poses)
