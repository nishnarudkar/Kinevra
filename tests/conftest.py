"""Shared fixtures: synthetic images and a tiny synthetic video (no camera needed)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from kinevra.config import QualityCfg, load_config
from kinevra.vision.types import Image

if TYPE_CHECKING:
    from kinevra.schemas import PoseFrame

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


def make_pose(
    abduction_deg: float,
    *,
    t: float = 0.0,
    frame_idx: int = 0,
    side: str = "right",
    elbow_flexion_deg: float = 0.0,
    lean_deg: float = 0.0,
    shoulder_raise_px: float = 0.0,
    width: int = 1280,
    height: int = 720,
    visibility: float = 0.95,
    hidden: tuple[str, ...] = (),
) -> PoseFrame:
    """Frontal-view pose with exact joint angles, built in pixels then normalised.

    The subject faces the camera, so their right side is on the image left (-x). `lean_deg`
    tilts the trunk away from the exercising side (positive, like the features convention).
    """
    import math

    from kinevra.schemas import Landmark, PoseFrame

    out = -1.0 if side == "right" else 1.0  # image x-direction of the exercising side
    other = "left" if side == "right" else "right"
    mid_hip = (640.0, 520.0)
    torso = 260.0
    lean = math.radians(lean_deg)
    # trunk axis tilted from vertical, away from the exercising side; pelvis stays level
    up = (-out * math.sin(lean), -math.cos(lean))
    mid_sh = (mid_hip[0] + up[0] * torso, mid_hip[1] + up[1] * torso)
    across = (-up[1] * out, up[0] * out)  # perpendicular to the trunk, toward exercising side
    half_sh, half_hip = 90.0, 60.0
    pts = {
        f"{side}_hip": (mid_hip[0] + out * half_hip, mid_hip[1]),
        f"{other}_hip": (mid_hip[0] - out * half_hip, mid_hip[1]),
        f"{side}_shoulder": (mid_sh[0] + across[0] * half_sh, mid_sh[1] + across[1] * half_sh),
        f"{other}_shoulder": (mid_sh[0] - across[0] * half_sh, mid_sh[1] - across[1] * half_sh),
    }
    sx, sy = pts[f"{side}_shoulder"]
    sy -= shoulder_raise_px
    pts[f"{side}_shoulder"] = (sx, sy)
    # abduction is measured from the shoulder→hip direction, rotating outward
    hx, hy = pts[f"{side}_hip"]
    down = math.atan2(hy - sy, hx - sx)
    arm_dir = down - out * math.radians(abduction_deg)
    upper, fore = 150.0, 130.0
    ex, ey = sx + upper * math.cos(arm_dir), sy + upper * math.sin(arm_dir)
    fore_dir = arm_dir - out * math.radians(elbow_flexion_deg)
    pts[f"{side}_elbow"] = (ex, ey)
    pts[f"{side}_wrist"] = (ex + fore * math.cos(fore_dir), ey + fore * math.sin(fore_dir))
    landmarks = {
        n: Landmark(
            name=n, x=x / width, y=y / height, visibility=0.1 if n in hidden else visibility
        )
        for n, (x, y) in pts.items()
    }
    return PoseFrame(
        session_id="test",
        frame_idx=frame_idx,
        t=t,
        image_width=width,
        image_height=height,
        landmarks=landmarks,
        person_count=1,
        frame_quality=1.0,
        quality_flags=[],
    )
