"""Build the configured pose estimator."""

from __future__ import annotations

from pathlib import Path

from kinevra.config import CONFIG_DIR, AppConfig
from kinevra.pose.base import DnnPoseEstimator, required_for_side
from kinevra.pose.mediapipe_dnn import MediaPipePoseEstimator
from kinevra.pose.opencv_dnn import RTMPoseEstimator

MODEL_NAMES = ("rtmpose", "mediapipe")
FRAMING_JOINTS = ("hip", "shoulder", "elbow", "wrist")


def model_dir(cfg: AppConfig) -> Path:
    path = Path(cfg.pose.model_dir)
    return path if path.is_absolute() else CONFIG_DIR.parent / path


def create_estimator(cfg: AppConfig, model: str | None = None) -> DnnPoseEstimator:
    pose = cfg.pose
    model = model or pose.model
    required = required_for_side(cfg.exercise.side, FRAMING_JOINTS)
    if model == "rtmpose":
        return RTMPoseEstimator(
            model_dir(cfg),
            engine=pose.engine,
            det_score_threshold=pose.det_score_threshold,
            det_nms_threshold=pose.det_nms_threshold,
            bbox_padding=pose.bbox_padding,
            roi_scale=cfg.exercise.tools.reanalyze_roi_scale,
            roi_clahe=pose.roi_clahe,
            required_landmarks=required,
            visibility_threshold=cfg.exercise.visibility_threshold,
        )
    if model == "mediapipe":
        return MediaPipePoseEstimator(
            model_dir(cfg),
            engine=pose.engine,
            det_score_threshold=pose.det_score_threshold,
            roi_scale=cfg.exercise.tools.reanalyze_roi_scale,
            roi_clahe=pose.roi_clahe,
            required_landmarks=required,
            visibility_threshold=cfg.exercise.visibility_threshold,
        )
    raise ValueError(f"unknown pose model {model!r}; choose from {MODEL_NAMES}")
