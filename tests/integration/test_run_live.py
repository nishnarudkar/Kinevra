"""Runs scripts/run_live.py headless on a synthetic video (same path as the webcam)."""

import json
import subprocess
import sys
from pathlib import Path

from tests.conftest import VIDEO_FRAMES

ROOT = Path(__file__).resolve().parents[2]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_live.py"), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
        check=False,
    )


def test_headless_run_on_video_and_record(synthetic_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "rec.avi"
    proc = _run(
        "--video", str(synthetic_video), "--headless", "--pose", "none", "--record", str(out)
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["frames"] == VIDEO_FRAMES
    assert summary["resolution"] == "160x120"
    assert summary["mean_fps"] > 20
    assert 0 <= summary["mean_quality"] <= 1
    assert out.exists() and out.stat().st_size > 0


def test_max_frames_and_downsample(synthetic_video: Path) -> None:
    proc = _run(
        "--video",
        str(synthetic_video),
        "--headless",
        "--pose",
        "none",
        "--fps",
        "5",
        "--max-frames",
        "4",
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["frames"] == 4


def test_missing_video_exits_with_error(tmp_path: Path) -> None:
    proc = _run("--video", str(tmp_path / "missing.mp4"), "--headless")
    assert proc.returncode == 2
    assert "not found" in proc.stderr


def test_csv_export_with_pose(synthetic_video: Path, tmp_path: Path) -> None:
    from kinevra.config import load_config
    from kinevra.movement.export import FIELDS, read_feature_csv
    from kinevra.pose.factory import model_dir
    from kinevra.pose.opencv_dnn import RTMDET_FILE, RTMPOSE_FILE

    mdir = model_dir(load_config(env={}))
    if not all((mdir / f).is_file() for f in (RTMDET_FILE, RTMPOSE_FILE)):
        import pytest

        pytest.skip("rtmpose files missing")
    out = tmp_path / "features.csv"
    proc = _run("--video", str(synthetic_video), "--headless", "--csv", str(out))
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["csv"] == str(out)
    rows = read_feature_csv(out)
    assert len(rows) == VIDEO_FRAMES
    assert set(rows[0]) == set(FIELDS)
    assert rows[0]["shoulder_abduction_deg"] is None  # no person in the synthetic video
    assert "no_person" in rows[0]["quality_flags"]
