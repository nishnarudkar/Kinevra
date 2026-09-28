"""Write frames to a video file (used by `run_live.py --record` and `record_clip.py`)."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import cv2

from kinevra.vision.types import Image


class ClipWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        """`size` is (width, height). MP4 (mp4v) for .mp4, MJPG for .avi."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        codec = "MJPG" if self.path.suffix.lower() == ".avi" else "mp4v"
        fourcc = cv2.VideoWriter.fourcc(*codec)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, float(fps), size)
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open video writer for {self.path}")
        self.frames = 0

    def write(self, image: Image) -> None:
        h, w = image.shape[:2]
        if (w, h) != self.size:
            raise ValueError(f"frame is {w}x{h}, writer expects {self.size[0]}x{self.size[1]}")
        self._writer.write(image)
        self.frames += 1

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> ClipWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
