"""Everything the agent's tools can read or write: frames, poses, session, memory, storage."""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from kinevra.config import AppConfig
from kinevra.movement.session import MovementSession
from kinevra.pose.base import PoseEstimator
from kinevra.schemas import PoseFrame, RepMetrics, ReviewEvent
from kinevra.storage.base import ArtifactStore
from kinevra.vision.buffer import FrameBuffer
from kinevra.vision.types import Frame


class FrameProvider(Protocol):
    def frames_between(self, t0: float, t1: float) -> list[Frame]: ...


class BufferFrameProvider:
    """Live mode: the last N seconds kept in the ring buffer (downscaled)."""

    def __init__(self, buffer: FrameBuffer) -> None:
        self.buffer = buffer

    def frames_between(self, t0: float, t1: float) -> list[Frame]:
        return self.buffer.segment(t0, t1)


class VideoFrameProvider:
    """Clip mode: seek back into the video file (full resolution, nothing held in memory)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        self.fps = reported if 0 < reported < 1000 else 30.0

    def frames_between(self, t0: float, t1: float) -> list[Frame]:
        cap = cv2.VideoCapture(self.path)
        try:
            idx = max(0, int(np.floor(t0 * self.fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            out: list[Frame] = []
            while True:
                ok, image = cap.read()
                t = idx / self.fps
                if not ok or t > t1 + 1e-6:
                    break
                if t >= t0 - 1e-6:
                    out.append(Frame(idx=idx, t=t, image=np.asarray(image, dtype=np.uint8)))
                idx += 1
            return out
        finally:
            cap.release()


class PoseHistory:
    """Time-ordered PoseFrames; `keep_s=None` keeps everything (clip mode)."""

    def __init__(self, keep_s: float | None = None) -> None:
        self.keep_s = keep_s
        self._poses: list[PoseFrame] = []
        self._times: list[float] = []

    def add(self, pose: PoseFrame) -> None:
        self._poses.append(pose)
        self._times.append(pose.t)
        if self.keep_s is not None:
            cut = bisect.bisect_left(self._times, pose.t - self.keep_s)
            if cut:
                del self._poses[:cut]
                del self._times[:cut]

    def extend(self, poses: Iterable[PoseFrame]) -> None:
        for p in poses:
            self.add(p)

    def segment(self, t0: float, t1: float) -> list[PoseFrame]:
        lo = bisect.bisect_left(self._times, t0 - 1e-6)
        hi = bisect.bisect_right(self._times, t1 + 1e-6)
        return self._poses[lo:hi]

    def nearest(self, t: float) -> PoseFrame | None:
        if not self._poses:
            return None
        i = bisect.bisect_left(self._times, t)
        candidates = [j for j in (i - 1, i) if 0 <= j < len(self._poses)]
        return self._poses[min(candidates, key=lambda j: abs(self._times[j] - t))]

    def window(self, seconds: float) -> list[PoseFrame]:
        if not self._poses:
            return []
        return self.segment(self._times[-1] - seconds, self._times[-1])

    def __len__(self) -> int:
        return len(self._poses)


@dataclass
class AgentMemory:
    """State the agent carries from one decision to the next."""

    confirmed_streak: int = 0  # consecutive reps whose deviation was confirmed
    escalated: bool = False  # already escalated for the current streak
    last_feedback_rep: int | None = None
    awaiting_camera_fix: bool = False
    last_camera_instruction: str | None = None
    last_quality_check_t: float | None = None


@dataclass
class ToolContext:
    cfg: AppConfig
    session: MovementSession
    frames: FrameProvider
    poses: PoseHistory
    store: ArtifactStore
    estimator: PoseEstimator | None = None
    memory: AgentMemory = field(default_factory=AgentMemory)
    review_events: list[ReviewEvent] = field(default_factory=list)
    logged_events: list[dict[str, Any]] = field(default_factory=list)
    snapshots: dict[int, list[str]] = field(default_factory=dict)

    @property
    def session_id(self) -> str:
        return self.session.state.session_id

    def rep(self, rep_index: int) -> RepMetrics | None:
        return next((r for r in self.session.state.reps if r.rep_index == rep_index), None)
