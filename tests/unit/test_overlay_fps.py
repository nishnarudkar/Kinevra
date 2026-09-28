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
