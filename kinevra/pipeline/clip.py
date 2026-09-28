"""Clip mode (offline): video → pose → features (Savitzky-Golay) → reps → rules → agent.

`analyze_clip` runs the measurement pipeline; pass an `llm` to also run the agent. The agent
is asked after every completed rep, and every `quality_check_interval_s` while data quality is
poor (or a camera fix is pending). Poses are fed to the agent's history in time order, so its
tools only ever see the past; rep frames are re-read from the video file (full resolution).
The AWS Lambda handler (Phase 7) wraps this.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kinevra.agent.context import FrameProvider, PoseHistory, ToolContext, VideoFrameProvider
from kinevra.agent.graph import KinevraAgent
from kinevra.agent.llm import LLMClient
from kinevra.config import CONFIG_DIR, AppConfig
from kinevra.movement.features import extract_features
from kinevra.movement.reps import RepEvent
from kinevra.movement.session import MovementSession
from kinevra.pose.base import PoseEstimator
from kinevra.pose.factory import create_estimator
from kinevra.schemas import (
    AgentDecision,
    FrameFeatures,
    PoseFrame,
    RepMetrics,
    ReviewEvent,
    SessionEvidence,
)
from kinevra.storage.base import ArtifactStore, put_json
from kinevra.storage.local import LocalStore
from kinevra.vision.capture import FrameSource
from kinevra.vision.quality import assess_frame

QUALITY_CHECK_WARMUP_S = 1.0  # let the rolling window fill before judging data quality


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
    decisions: list[AgentDecision] = field(default_factory=list)
    review_events: list[ReviewEvent] = field(default_factory=list)
    logged_events: list[dict[str, Any]] = field(default_factory=list)
    artifact_keys: list[str] = field(default_factory=list)


def default_store(cfg: AppConfig) -> LocalStore:
    root = Path(cfg.storage.local_dir)
    return LocalStore(root if root.is_absolute() else CONFIG_DIR.parent / root)


def estimate_poses(
    path: str | Path, cfg: AppConfig, estimator: PoseEstimator, session_id: str
) -> list[PoseFrame]:
    poses: list[PoseFrame] = []
    with FrameSource(path, target_fps=cfg.exercise.processing_fps) as src:
        for frame in src:
            quality = assess_frame(frame.image, cfg.quality)
            poses.append(
                estimator.estimate(
                    frame.image, frame.t, frame.idx, session_id=session_id, quality=quality
                )
            )
    return poses


def analyze_poses(
    poses: list[PoseFrame],
    cfg: AppConfig,
    *,
    session_id: str,
    source: str = "",
    reps_target: int | None = None,
    llm: LLMClient | None = None,
    frames: FrameProvider | None = None,
    estimator: PoseEstimator | None = None,
    store: ArtifactStore | None = None,
) -> ClipResult:
    """Features + reps + rules (+ agent if `llm` and `frames` are given) on estimated poses."""
    features = extract_features(poses, cfg.exercise)
    session = MovementSession(cfg.exercise, session_id, "clip", reps_target)
    agent: KinevraAgent | None = None
    if llm is not None and frames is not None:
        ctx = ToolContext(
            cfg=cfg,
            session=session,
            frames=frames,
            poses=PoseHistory(),
            store=store or default_store(cfg),
            estimator=estimator,
        )
        agent = KinevraAgent(ctx, llm)
    interval = cfg.exercise.agent.quality_check_interval_s

    reps: list[RepMetrics] = []
    for pose, feat in zip(poses, features, strict=True):
        rep = session.update(feat, pose.quality_flags)
        if rep is not None:
            reps.append(rep)
        if agent is None:
            continue
        agent.ctx.poses.add(pose)
        if rep is not None:
            agent.decide("rep", rep, session.evidence())
            continue
        mem = agent.ctx.memory
        poor = bool(session.state.data_quality_flags()) or mem.awaiting_camera_fix
        due = mem.last_quality_check_t is None or feat.t - mem.last_quality_check_t >= interval
        if poor and due and feat.t >= QUALITY_CHECK_WARMUP_S:
            mem.last_quality_check_t = feat.t
            agent.decide("quality", None, session.evidence())
    session.finish()

    result = ClipResult(
        session_id, source, poses, features, reps, list(session.events), session.evidence()
    )
    if agent is not None:
        result.decisions = list(agent.decisions)
        result.review_events = list(agent.ctx.review_events)
        result.logged_events = list(agent.ctx.logged_events)
        result.artifact_keys = save_session(agent.ctx.store, result)
    return result


def save_session(store: ArtifactStore, result: ClipResult) -> list[str]:
    """Decisions (with traces), review events, reps and the summary as JSON artifacts."""
    sid = result.session_id
    return [
        put_json(
            store, f"{sid}/decisions.json", [d.model_dump(mode="json") for d in result.decisions]
        ),
        put_json(
            store,
            f"{sid}/review_events.json",
            [e.model_dump(mode="json") for e in result.review_events],
        ),
        put_json(store, f"{sid}/reps.json", [r.model_dump(mode="json") for r in result.reps]),
        put_json(
            store,
            f"{sid}/summary.json",
            {
                "session_id": sid,
                "source": result.source,
                "evidence": result.evidence.model_dump(mode="json"),
                "rep_events": [e.__dict__ for e in result.events],
                "logged_events": result.logged_events,
                "timings_s": result.timings_s,
            },
        ),
    ]


def analyze_clip(
    path: str | Path,
    cfg: AppConfig,
    *,
    estimator: PoseEstimator | None = None,
    session_id: str | None = None,
    reps_target: int | None = None,
    llm: LLMClient | None = None,
    store: ArtifactStore | None = None,
) -> ClipResult:
    sid = session_id or uuid.uuid4().hex[:12]
    est = estimator or create_estimator(cfg)
    t0 = time.perf_counter()
    poses = estimate_poses(path, cfg, est, sid)
    t1 = time.perf_counter()
    result = analyze_poses(
        poses,
        cfg,
        session_id=sid,
        source=str(path),
        reps_target=reps_target,
        llm=llm,
        frames=VideoFrameProvider(path) if llm is not None else None,
        estimator=est,
        store=store,
    )
    result.timings_s = {"pose": round(t1 - t0, 3), "analysis": round(time.perf_counter() - t1, 3)}
    return result
