"""Frame sources: webcam (live mode) or video file (clip mode), with monotonic timestamps.

- Webcam: timestamps come from a monotonic clock, starting at 0 on the first frame.
- Video file: timestamps come from the frame index and the file's FPS, so they are
  deterministic and reproducible. `target_fps` downsamples a file to the processing FPS.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from kinevra.vision.types import Frame

DEFAULT_FILE_FPS = 30.0


class CaptureError(RuntimeError):
    pass


class FrameSource:
    def __init__(
        self,
        source: int | str | Path,
        *,
        width: int | None = None,
        height: int | None = None,
        target_fps: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if target_fps is not None and target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self.is_live = isinstance(source, int)
        self._clock = clock
        self._target_fps = target_fps

        if isinstance(source, int):
            # DirectShow opens webcams much faster than Media Foundation on Windows.
            api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self._cap = cv2.VideoCapture(source, api)
        else:
            path = Path(source)
            if not path.is_file():
                raise CaptureError(f"video file not found: {path}")
            self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise CaptureError(f"could not open video source: {source!r}")

        if self.is_live:
            if width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if target_fps:
                self._cap.set(cv2.CAP_PROP_FPS, target_fps)

        reported = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.native_fps = reported if 0 < reported < 1000 else DEFAULT_FILE_FPS
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_count: int | None = None if self.is_live or count <= 0 else count

        self._decoded = 0  # frames decoded from the source so far
        self._t0: float | None = None
        self._last_t = -1.0
        self._next_keep_t = 0.0

    @property
    def fps(self) -> float:
        """Nominal output FPS (after downsampling for files)."""
        if self._target_fps is None or self.is_live:
            return self.native_fps
        return min(self._target_fps, self.native_fps)

    def _timestamp(self, idx: int) -> float:
        if self.is_live:
            now = self._clock()
            if self._t0 is None:
                self._t0 = now
            t = now - self._t0
        else:
            t = idx / self.native_fps
        if t <= self._last_t:  # guarantee strictly increasing timestamps
            t = self._last_t + 1e-6
        return t

    def read(self) -> Frame | None:
        """Return the next frame, or None at end of stream / camera failure."""
        while True:
            ok, image = self._cap.read()
            if not ok or image is None:
                return None
            idx = self._decoded
            self._decoded += 1
            t = self._timestamp(idx)
            if not self.is_live and self._target_fps is not None:
                if t + 1e-9 < self._next_keep_t:
                    continue
                self._next_keep_t += 1.0 / self._target_fps
                # never fall behind if the file's FPS is lower than target
                self._next_keep_t = max(self._next_keep_t, t)
            self._last_t = t
            return Frame(idx=idx, t=t, image=np.asarray(image, dtype=np.uint8))

    def __iter__(self) -> Iterator[Frame]:
        while (frame := self.read()) is not None:
            yield frame

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
