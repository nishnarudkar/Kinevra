import numpy as np
import pytest

from kinevra.vision.fps import FpsMeter
from kinevra.vision.overlay import (
    AMBER,
    GREEN,
    RED,
    draw_countdown,
    draw_hud,
    draw_recording,
    quality_colour,
)
from kinevra.vision.quality import FrameQuality


def test_draw_functions_do_not_mutate_input() -> None:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    q = FrameQuality(brightness=30, contrast=5, sharpness=10, score=0.1, flags=["low_light"])
    for out in (
        draw_hud(img, ["fps 30.0", "frame 1"], q),
        draw_recording(img),
        draw_countdown(img, 3),
    ):
        assert out.shape == img.shape
        assert out.any()
    assert not img.any()


def test_hud_without_quality_or_lines() -> None:
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    out = draw_hud(img, [], None, show_disclaimer=False)
    assert not out.any()


@pytest.mark.parametrize(("score", "colour"), [(0.9, GREEN), (0.5, AMBER), (0.1, RED)])
def test_quality_colour(score: float, colour: tuple[int, int, int]) -> None:
    assert quality_colour(score) == colour


def test_fps_meter() -> None:
    meter = FpsMeter(alpha=1.0)
    assert meter.mean_fps == 0.0
    for i in range(11):
        meter.tick(i * 0.05)  # 20 FPS
    assert meter.fps == pytest.approx(20.0)
    assert meter.mean_fps == pytest.approx(20.0)
    assert meter.elapsed_s == pytest.approx(0.5)
    with pytest.raises(ValueError):
        FpsMeter(alpha=0)


def test_draw_skeleton_colours_by_visibility() -> None:
    from kinevra.schemas import Landmark
    from kinevra.vision.overlay import draw_skeleton, visibility_colour

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    lms = {
        "right_shoulder": Landmark(name="right_shoulder", x=0.2, y=0.2, visibility=0.9),
        "right_elbow": Landmark(name="right_elbow", x=0.8, y=0.8, visibility=0.9),
    }
    out = draw_skeleton(
        img, lms, [("right_shoulder", "right_elbow"), ("nose", "left_eye")], highlight_side="right"
    )
    assert not img.any()
    assert tuple(int(v) for v in out[50, 50]) == GREEN
    assert visibility_colour(0.3, 0.5) == AMBER and visibility_colour(0.1, 0.5) == RED


def test_mirror_landmarks_and_angle_arc() -> None:
    from kinevra.schemas import Landmark
    from kinevra.vision.overlay import draw_angle_arc, mirror_landmarks

    lms = {"right_elbow": Landmark(name="right_elbow", x=0.2, y=0.5, visibility=0.9)}
    mirrored = mirror_landmarks(lms)
    assert mirrored["right_elbow"].x == pytest.approx(0.8) and lms["right_elbow"].x == 0.2

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    out = draw_angle_arc(img, (100, 100), (100, 180), (180, 100), 90.0, radius=40)
    assert not img.any()
    # the arc passes through the bisector of the 90° wedge (down-right), not the far side
    near = out[100 + 26 : 100 + 32, 100 + 26 : 100 + 32].any()
    far = out[100 - 32 : 100 - 26, 100 - 32 : 100 - 26].any()
    assert near and not far
