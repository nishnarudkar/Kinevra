from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kinevra.config import CONFIG_DIR, load_config, load_exercise_config


def test_default_config_loads() -> None:
    cfg = load_config(env={})
    assert cfg.exercise.exercise == "shoulder_abduction"
    assert cfg.exercise.side == "right"
    assert cfg.exercise.landmarks.primary_angle == ("hip", "shoulder", "elbow")
    assert cfg.exercise.tools.reanalyze_trigger_confidence == (0.45, 0.75)
    assert cfg.exercise.agent.escalation.consecutive_confirmed_deviations == 3
    assert cfg.capture.width == 640
    assert cfg.storage.backend == "local"


def test_env_overrides() -> None:
    cfg = load_config(
        env={
            "KINEVRA_SIDE": "left",
            "KINEVRA_LLM_MODEL_ID": "some-model",
            "KINEVRA_STORAGE_BACKEND": "aws",
            "KINEVRA_BUCKET": "bucket-x",
        }
    )
    assert cfg.exercise.side == "left"
    assert cfg.exercise.llm.model_id == "some-model"
    assert cfg.storage.backend == "aws"
    assert cfg.storage.bucket == "bucket-x"


def test_invalid_env_value_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(env={"KINEVRA_SIDE": "middle"})


def _write_modified(tmp_path: Path, mutate: dict[str, object]) -> Path:
    data = yaml.safe_load((CONFIG_DIR / "exercises" / "shoulder_abduction.yaml").read_text("utf-8"))
    data.update(mutate)
    out = tmp_path / "ex.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


def test_unknown_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_exercise_config(_write_modified(tmp_path, {"typo_key": 1}))


def test_confidence_band_must_be_ordered(tmp_path: Path) -> None:
    tools = {
        "reanalyze_roi_scale": 2.0,
        "reanalyze_trigger_confidence": [0.8, 0.4],
        "flow_roi_padding": 0.1,
    }
    with pytest.raises(ValidationError):
        load_exercise_config(_write_modified(tmp_path, {"tools": tools}))
