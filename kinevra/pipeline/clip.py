"""Clip mode (offline) analysis: video → pose → features (Savitzky-Golay) → reps → rules.

Phase 4 scope: the deterministic measurement pipeline. The agent loop (Phase 5) and the AWS
Lambda handler (Phase 7) build on `analyze_clip`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from kinevra.config import AppConfig
from kinevra.movement.features import extract_features
from kinevra.movement.reps import RepEvent
from kinevra.movement.session import MovementSession
from kinevra.pose.base import PoseEstimator
from kinevra.pose.factory import create_estimator
from kinevra.schemas import FrameFeatures, PoseFrame, RepMetrics, SessionEvidence
from kinevra.vision.capture import FrameSource
from kinevra.vision.quality import assess_frame


@dataclass
class ClipResult:
    session_id: str
    source: str
    poses: list[PoseFrame]
    features: list[FrameFeatures]
    reps: list[RepMetrics]
    events: list[RepEvent]
    evidence: SessionEvidence
    timings_s: dict[str, float] = field(default_factory=dict)


def analyze_poses(
    poses: list[PoseFrame],
    cfg: AppConfig,
    *,
    session_id: str,
    source: str = "",
    reps_target: int | None = None,
) -> ClipResult:
    """Features + reps + rules on already-estimated poses (lets tests skip the model)."""
    features = extract_features(poses, cfg.exercise)
    session = MovementSession(cfg.exercise, session_id, "clip", reps_target)
    reps: list[RepMetrics] = []
    for pose, feat in zip(poses, features, strict=True):
        rep = session.update(feat, pose.quality_flags)
        if rep is not None:
            reps.append(rep)
    session.finish()
    return ClipResult(
        session_id, source, poses, features, reps, list(session.events), session.evidence()
    )


def analyze_clip(
    path: str | Path,
    cfg: AppConfig,
    *,
    estimator: PoseEstimator | None = None,
    session_id: str | None = None,
    reps_target: int | None = None,
) -> ClipResult:
    sid = session_id or uuid.uuid4().hex[:12]
    est = estimator or create_estimator(cfg)
    t0 = time.perf_counter()
    poses: list[PoseFrame] = []
    with FrameSource(path, target_fps=cfg.exercise.processing_fps) as src:
        for frame in src:
            quality = assess_frame(frame.image, cfg.quality)
            poses.append(
                est.estimate(frame.image, frame.t, frame.idx, session_id=sid, quality=quality)
            )
    t1 = time.perf_counter()
    result = analyze_poses(poses, cfg, session_id=sid, source=str(path), reps_target=reps_target)
    result.timings_s = {"pose": round(t1 - t0, 3), "analysis": round(time.perf_counter() - t1, 3)}
    return result
