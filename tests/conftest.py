"""Shared fixtures: synthetic images and a tiny synthetic video (no camera needed)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from kinevra.config import QualityCfg, load_config
from kinevra.vision.types import Image

VIDEO_FPS = 10.0
VIDEO_FRAMES = 30
VIDEO_SIZE = (160, 120)  # (w, h)


def checkerboard(h: int = 240, w: int = 320, cell: int = 16) -> Image:
    ys, xs = np.indices((h, w))
    board = (((ys // cell) + (xs // cell)) % 2 * 200 + 30).astype(np.uint8)
    return np.asarray(cv2.cvtColor(board, cv2.COLOR_GRAY2BGR), dtype=np.uint8)


@pytest.fixture
def quality_cfg() -> QualityCfg:
    return load_config(env={}).quality


@pytest.fixture
def sharp_image() -> Image:
    return checkerboard()


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """30 frames at 10 FPS; a white square moves right one step per frame."""
    path = tmp_path / "synthetic.avi"
    w, h = VIDEO_SIZE
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), VIDEO_FPS, (w, h))
    assert writer.isOpened()
    for i in range(VIDEO_FRAMES):
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        cv2.rectangle(img, (5 + i * 4, 40), (25 + i * 4, 60), (255, 255, 255), -1)
        writer.write(img)
    writer.release()
    return path
