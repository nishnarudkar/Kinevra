import pytest

from kinevra.evaluation.angle_sanity import HeldPose, markdown_table, summarise


def test_held_pose_stats() -> None:
    p = HeldPose("90°", 90.0, [88.0, 90.0, 92.0])
    assert p.mean == pytest.approx(90.0)
    assert p.std == pytest.approx((8 / 3) ** 0.5)
    assert p.error == pytest.approx(0.0)
    assert HeldPose("x", 10.0, [12.0]).std == 0.0


def test_summary_and_table() -> None:
    poses = [HeldPose("0°", 5.0, [8.0, 8.0]), HeldPose("90°", 90.0, [86.0, 86.0])]
    s = summarise(poses)
    assert s["n"] == 2 and s["mae_deg"] == pytest.approx(3.5) and s["max_abs_error_deg"] == 4.0
    md = markdown_table(poses, context="laptop webcam, 2 m")
    assert "| 0° | 5.0 | 8.0 | 0.0 | +3.0 | 2 |" in md
    assert "| 90° | 90.0 | 86.0 | 0.0 | -4.0 | 2 |" in md
    assert "Mean absolute error:** 3.5°" in md and "laptop webcam, 2 m" in md
    assert summarise([]) == {"n": 0.0}
    assert "| Pose |" in markdown_table([])
