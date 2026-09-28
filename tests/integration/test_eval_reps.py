"""scripts/eval_reps.py end to end on a synthetic clip with a label file."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kinevra.config import load_config
from kinevra.pose.factory import model_dir
from kinevra.pose.opencv_dnn import RTMDET_FILE, RTMPOSE_FILE

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.models
def test_eval_reps_writes_report(synthetic_video: Path, tmp_path: Path) -> None:
    mdir = model_dir(load_config(env={}))
    if not all((mdir / f).is_file() for f in (RTMDET_FILE, RTMPOSE_FILE)):
        pytest.skip("rtmpose files missing")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "synthetic.json").write_text(
        json.dumps({"clip": synthetic_video.name, "side": "right", "reps": 0}), encoding="utf-8"
    )
    out = tmp_path / "report.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_reps.py"),
            "--labels",
            str(labels),
            "--clips",
            str(synthetic_video.parent),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = out.read_text(encoding="utf-8")
    assert "| synthetic.avi | 0 | 0 | yes |" in report
    assert "100%" in report
