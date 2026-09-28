"""Real ONNX models through cv2.dnn. Skipped when models/ is empty (e.g. CI without download)."""

from pathlib import Path

import numpy as np
import pytest

from kinevra.config import load_config
from kinevra.pose.factory import create_estimator, model_dir
from kinevra.pose.mediapipe_dnn import PERSONDET_FILE, POSE_FILE
from kinevra.pose.opencv_dnn import RTMDET_FILE, RTMPOSE_FILE
from kinevra.vision.preprocess import Roi

CFG = load_config(env={})
FILES = {"rtmpose": (RTMDET_FILE, RTMPOSE_FILE), "mediapipe": (PERSONDET_FILE, POSE_FILE)}


def _available(model: str) -> bool:
    return all((model_dir(CFG) / f).is_file() for f in FILES[model])


@pytest.mark.models
@pytest.mark.parametrize("model", list(FILES))
def test_empty_scene_has_no_person(model: str) -> None:
    if not _available(model):
        pytest.skip(f"{model} files missing; run scripts/download_models.py")
    est = create_estimator(CFG, model)
    img = np.full((480, 640, 3), 120, dtype=np.uint8)
    pf = est.estimate(img, 0.0, 0)
    assert pf.person_count == 0 and pf.landmarks == {}
    assert "no_person" in pf.quality_flags
    assert est.last_latency_ms > 0


@pytest.mark.models
def test_rtmpose_roi_mode_runs_without_detector() -> None:
    if not _available("rtmpose"):
        pytest.skip("rtmpose files missing")
    est = create_estimator(CFG, "rtmpose")
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    pf = est.estimate(img, 0.0, 0, roi=Roi(200, 100, 160, 200, 640, 480))
    assert len(pf.landmarks) == 17  # top-down on the crop always returns 17 points
    for lm in pf.landmarks.values():
        assert 200 / 640 - 0.05 <= lm.x <= 360 / 640 + 0.05


def test_model_dir_is_repo_relative() -> None:
    assert model_dir(CFG) == Path(__file__).resolve().parents[2] / "models"


@pytest.mark.models
def test_rtmdet_handles_changing_frames() -> None:
    """Regression: the official end2end export failed on the 2nd different frame in cv2.dnn."""
    if not _available("rtmpose"):
        pytest.skip("rtmpose files missing")
    est = create_estimator(CFG, "rtmpose")
    rng = np.random.default_rng(1)
    for i in range(5):
        img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        img[100 + i * 10 : 300, 200:260] = 255  # content changes every frame
        est.estimate(img, i / 15, i)
