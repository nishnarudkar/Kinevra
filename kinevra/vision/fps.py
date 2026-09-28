"""Frame-rate measurement for the live HUD and benchmarks."""

from __future__ import annotations


class FpsMeter:
    """Exponential moving average of instantaneous FPS plus overall mean."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.fps = 0.0
        self.frames = 0
        self._first: float | None = None
        self._last: float | None = None

    def tick(self, now: float) -> float:
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = (
                    inst if self.fps == 0 else self.alpha * inst + (1 - self.alpha) * self.fps
                )
        else:
            self._first = now
        self._last = now
        self.frames += 1
        return self.fps

    @property
    def elapsed_s(self) -> float:
        if self._first is None or self._last is None:
            return 0.0
        return self._last - self._first

    @property
    def mean_fps(self) -> float:
        """Average over the whole run: (frames - 1) intervals / elapsed time."""
        return (self.frames - 1) / self.elapsed_s if self.elapsed_s > 0 else 0.0
