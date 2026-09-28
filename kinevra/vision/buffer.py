"""Ring buffer of the last N seconds of (downscaled) frames.

The agent's "look again" tools (Phase 5) pull a rep's frames back out of this buffer by
time range and re-run OpenCV analysis on them. Landmarks are normalised 0..1, so they stay
valid on the downscaled copies.
"""

from __future__ import annotations

from collections import deque

from kinevra.vision.preprocess import resize_to_width
from kinevra.vision.types import Frame


class FrameBuffer:
    def __init__(self, seconds: float, downscale_width: int | None = None) -> None:
        if seconds <= 0:
            raise ValueError("seconds must be > 0")
        self.seconds = seconds
        self.downscale_width = downscale_width
        self._frames: deque[Frame] = deque()

    def push(self, frame: Frame) -> None:
        if self._frames and frame.t <= self._frames[-1].t:
            raise ValueError("frames must be pushed with strictly increasing timestamps")
        image = frame.image
        if self.downscale_width is not None and image.shape[1] > self.downscale_width:
            image = resize_to_width(image, self.downscale_width)
        else:
            image = image.copy()  # never alias the caller's (possibly reused) array
        self._frames.append(Frame(idx=frame.idx, t=frame.t, image=image))
        cutoff = frame.t - self.seconds
        while self._frames[0].t < cutoff:
            self._frames.popleft()

    def segment(self, t_start: float, t_end: float) -> list[Frame]:
        """Frames with t_start <= t <= t_end (inclusive), oldest first."""
        if t_end < t_start:
            raise ValueError("t_end must be >= t_start")
        return [f for f in self._frames if t_start <= f.t <= t_end]

    def window(self, seconds: float) -> list[Frame]:
        """Frames from the most recent `seconds`."""
        if not self._frames:
            return []
        latest = self._frames[-1].t
        return self.segment(latest - seconds, latest)

    def latest(self) -> Frame | None:
        return self._frames[-1] if self._frames else None

    @property
    def span_s(self) -> float:
        return self._frames[-1].t - self._frames[0].t if self._frames else 0.0

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)
