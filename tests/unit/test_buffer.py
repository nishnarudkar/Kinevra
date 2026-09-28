import numpy as np
import pytest

from kinevra.vision.buffer import FrameBuffer
from kinevra.vision.types import Frame


def _frame(idx: int, t: float, w: int = 64, h: int = 48) -> Frame:
    return Frame(idx=idx, t=t, image=np.full((h, w, 3), idx % 256, dtype=np.uint8))


def test_evicts_frames_older_than_window() -> None:
    buf = FrameBuffer(seconds=1.0)
    for i in range(31):  # 0.0 .. 3.0 s at 10 FPS
        buf.push(_frame(i, i * 0.1))
    assert buf.latest() is not None
    assert buf.latest().t == pytest.approx(3.0)  # type: ignore[union-attr]
    assert len(buf) == 11  # 2.0 .. 3.0 inclusive
    assert buf.span_s == pytest.approx(1.0)


def test_segment_inclusive_and_ordered() -> None:
    buf = FrameBuffer(seconds=10.0)
    for i in range(20):
        buf.push(_frame(i, i * 0.1))
    seg = buf.segment(0.5, 0.8)
    assert [f.idx for f in seg] == [5, 6, 7, 8]
    assert buf.segment(5.0, 6.0) == []
    with pytest.raises(ValueError):
        buf.segment(1.0, 0.5)


def test_window_returns_most_recent_seconds() -> None:
    buf = FrameBuffer(seconds=10.0)
    assert buf.window(1.0) == []
    for i in range(20):
        buf.push(_frame(i, i * 0.1))
    assert [f.idx for f in buf.window(0.3)] == [16, 17, 18, 19]


def test_downscales_but_keeps_aspect() -> None:
    buf = FrameBuffer(seconds=5.0, downscale_width=32)
    buf.push(_frame(0, 0.0, w=64, h=48))
    stored = buf.latest()
    assert stored is not None
    assert stored.image.shape == (24, 32, 3)


def test_small_frames_not_upscaled_and_copied() -> None:
    buf = FrameBuffer(seconds=5.0, downscale_width=640)
    original = _frame(0, 0.0)
    buf.push(original)
    original.image[:] = 99  # caller reuses its array
    stored = buf.latest()
    assert stored is not None
    assert stored.image.shape == original.image.shape
    assert int(stored.image[0, 0, 0]) == 0


def test_rejects_non_increasing_timestamps() -> None:
    buf = FrameBuffer(seconds=5.0)
    buf.push(_frame(0, 1.0))
    with pytest.raises(ValueError):
        buf.push(_frame(1, 1.0))


def test_clear_and_invalid_args() -> None:
    buf = FrameBuffer(seconds=5.0)
    buf.push(_frame(0, 0.0))
    buf.clear()
    assert len(buf) == 0 and buf.latest() is None and buf.span_s == 0.0
    with pytest.raises(ValueError):
        FrameBuffer(seconds=0)
