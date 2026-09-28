from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from kinevra.vision.capture import CaptureError, FrameSource
from kinevra.vision.recording import ClipWriter
from tests.conftest import VIDEO_FPS, VIDEO_FRAMES, VIDEO_SIZE


def test_reads_all_frames_with_file_timestamps(synthetic_video: Path) -> None:
    with FrameSource(synthetic_video) as src:
        assert not src.is_live
        assert src.native_fps == pytest.approx(VIDEO_FPS)
        assert (src.width, src.height) == VIDEO_SIZE
        frames = list(src)
    assert len(frames) == VIDEO_FRAMES
    assert [f.idx for f in frames] == list(range(VIDEO_FRAMES))
    assert [f.t for f in frames] == pytest.approx([i / VIDEO_FPS for i in range(VIDEO_FRAMES)])
    assert frames[0].image.shape == (VIDEO_SIZE[1], VIDEO_SIZE[0], 3)
    assert frames[0].image.dtype == np.uint8


def test_target_fps_downsamples(synthetic_video: Path) -> None:
    with FrameSource(synthetic_video, target_fps=5.0) as src:
        assert src.fps == pytest.approx(5.0)
        frames = list(src)
    assert [f.idx for f in frames] == list(range(0, VIDEO_FRAMES, 2))
    ts = [f.t for f in frames]
    assert all(b > a for a, b in pairwise(ts))


def test_target_fps_above_native_keeps_every_frame(synthetic_video: Path) -> None:
    with FrameSource(synthetic_video, target_fps=30.0) as src:
        assert src.fps == pytest.approx(VIDEO_FPS)
        assert len(list(src)) == VIDEO_FRAMES


def test_missing_file_and_bad_fps(tmp_path: Path) -> None:
    with pytest.raises(CaptureError):
        FrameSource(tmp_path / "nope.mp4")
    with pytest.raises(ValueError):
        FrameSource(tmp_path / "nope.mp4", target_fps=0)


def test_live_timestamps_use_clock_and_stay_monotonic(synthetic_video: Path) -> None:
    # Drive the live-timestamp logic with a fake clock that stalls and jumps backwards.
    ticks = iter([100.0, 100.5, 100.5, 100.2, 101.0])
    src = FrameSource(synthetic_video, clock=lambda: next(ticks))
    src.is_live = True
    ts = [f.t for f in (src.read() for _ in range(5)) if f is not None]
    src.close()
    assert ts[0] == 0.0 and ts[1] == pytest.approx(0.5) and ts[4] == pytest.approx(1.0)
    assert all(b > a for a, b in pairwise(ts))


def test_clip_writer_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "clip.avi"
    with ClipWriter(out, 15.0, (64, 48)) as writer:
        for i in range(12):
            writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
        with pytest.raises(ValueError):
            writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
    assert writer.frames == 12
    with FrameSource(out) as src:
        assert src.native_fps == pytest.approx(15.0)
        assert len(list(src)) == 12
