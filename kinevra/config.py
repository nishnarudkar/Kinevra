"""Typed configuration loader for configs/*.yaml.

Unknown keys are rejected so typos in YAML fail loudly. A few deployment-specific values
can be overridden with environment variables (see .env.example); secrets never live here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kinevra.schemas import Side

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- exercise config (configs/exercises/*.yaml) -------------------------------------------


class LandmarksCfg(_Strict):
    primary_angle: tuple[str, str, str]
    elbow_angle: tuple[str, str, str]


class OneEuroCfg(_Strict):
    min_cutoff: float = Field(gt=0)
    beta: float = Field(ge=0)
    d_cutoff: float = Field(gt=0)


class SmoothingCfg(_Strict):
    one_euro: OneEuroCfg


class RepsCfg(_Strict):
    rest_angle_max: float
    raise_margin: float = Field(ge=0)
    min_rom: float = Field(gt=0)
    min_duration_s: float = Field(gt=0)
    max_duration_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _durations_ordered(self) -> RepsCfg:
        if self.min_duration_s >= self.max_duration_s:
            raise ValueError("reps.min_duration_s must be < reps.max_duration_s")
        return self


class BaselineCfg(_Strict):
    first_n_good_reps: int = Field(ge=1)
    recent_k: int = Field(ge=1)


class RulesCfg(_Strict):
    rom_ratio_deviation: float = Field(gt=0, le=1)
    max_trunk_lean_deg: float = Field(ge=0)
    max_elbow_flexion_deg: float = Field(ge=0)
    min_confidence: float = Field(ge=0, le=1)


class ToolsCfg(_Strict):
    reanalyze_roi_scale: float = Field(ge=1)
    reanalyze_trigger_confidence: tuple[float, float]
    flow_roi_padding: float = Field(ge=0)

    @model_validator(mode="after")
    def _band_ordered(self) -> ToolsCfg:
        lo, hi = self.reanalyze_trigger_confidence
        if not 0 <= lo < hi <= 1:
            raise ValueError("tools.reanalyze_trigger_confidence must satisfy 0 <= lo < hi <= 1")
        return self


class EscalationCfg(_Strict):
    consecutive_confirmed_deviations: int = Field(ge=1)
    rom_ratio_below: float = Field(gt=0, le=1)


class AgentCfg(_Strict):
    max_tool_calls: int = Field(ge=0)
    feedback_min_reps_between: int = Field(ge=0)
    escalation: EscalationCfg


class LLMCfg(_Strict):
    provider: Literal["bedrock", "fake"]
    model_id: str
    region: str
    temperature: float = Field(ge=0, le=1)
    timeout_s: float = Field(gt=0)


class LimitsCfg(_Strict):
    max_clip_seconds: float = Field(gt=0)
    max_upload_mb: float = Field(gt=0)


class ExerciseConfig(_Strict):
    exercise: Literal["shoulder_abduction"]
    side: Side
    landmarks: LandmarksCfg
    visibility_threshold: float = Field(ge=0, le=1)
    processing_fps: float = Field(gt=0)
    smoothing: SmoothingCfg
    reps: RepsCfg
    baseline: BaselineCfg
    rules: RulesCfg
    tools: ToolsCfg
    agent: AgentCfg
    llm: LLMCfg
    limits: LimitsCfg


# --- runtime config (configs/default.yaml) ------------------------------------------------


class CaptureCfg(_Strict):
    camera_index: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    target_fps: float = Field(gt=0)


class BufferCfg(_Strict):
    seconds: float = Field(gt=0)
    downscale_width: int = Field(gt=0)


class StorageCfg(_Strict):
    backend: Literal["local", "aws"]
    local_dir: str
    bucket: str | None
    table: str | None


class LoggingCfg(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    json_: bool = Field(alias="json")


class AppConfig(_Strict):
    exercise_config: str
    capture: CaptureCfg
    buffer: BufferCfg
    storage: StorageCfg
    logging: LoggingCfg
    exercise: ExerciseConfig


# --- loading ------------------------------------------------------------------------------

# env var -> (dotted path inside the merged config dict)
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "KINEVRA_SIDE": ("exercise", "side"),
    "KINEVRA_LLM_PROVIDER": ("exercise", "llm", "provider"),
    "KINEVRA_LLM_MODEL_ID": ("exercise", "llm", "model_id"),
    "KINEVRA_LLM_REGION": ("exercise", "llm", "region"),
    "KINEVRA_STORAGE_BACKEND": ("storage", "backend"),
    "KINEVRA_BUCKET": ("storage", "bucket"),
    "KINEVRA_TABLE": ("storage", "table"),
    "KINEVRA_LOG_LEVEL": ("logging", "level"),
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _apply_env(data: dict[str, Any], env: dict[str, str]) -> None:
    for var, path in ENV_OVERRIDES.items():
        if var not in env:
            continue
        node = data
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = env[var]


def load_exercise_config(path: str | Path) -> ExerciseConfig:
    return ExerciseConfig.model_validate(_read_yaml(Path(path)))


def load_config(path: str | Path | None = None, env: dict[str, str] | None = None) -> AppConfig:
    """Load configs/default.yaml (or `path`) plus its exercise config, then env overrides."""
    main_path = Path(path) if path is not None else CONFIG_DIR / "default.yaml"
    data = _read_yaml(main_path)
    data["exercise"] = _read_yaml(main_path.parent / data["exercise_config"])
    _apply_env(data, dict(os.environ) if env is None else env)
    return AppConfig.model_validate(data)
