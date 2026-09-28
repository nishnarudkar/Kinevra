import numpy as np
import pytest

from kinevra.vision.preprocess import (
    Roi,
    apply_clahe,
    crop_roi,
    mirror_for_display,
    resize_to_width,
    roi_from_points,
    to_gray,
    to_rgb,
)


def test_resize_to_width_keeps_aspect() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    assert resize_to_width(img, 320).shape == (240, 320, 3)
    assert resize_to_width(img, 1280).shape == (960, 1280, 3)
    with pytest.raises(ValueError):
        resize_to_width(img, 0)


def test_colour_conversions() -> None:
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[..., 0] = 255  # blue in BGR
    assert to_rgb(img)[0, 0].tolist() == [0, 0, 255]
    assert to_gray(img).shape == (2, 2)
    gray = np.zeros((2, 2), dtype=np.uint8)
    assert to_gray(gray) is gray


def test_clahe_increases_contrast_in_dark_low_contrast_image() -> None:
    rng = np.random.default_rng(0)
    dark = rng.integers(20, 40, (120, 160, 3), dtype=np.uint8)
    out = apply_clahe(dark)
    assert out.shape == dark.shape and out.dtype == np.uint8
    assert out.std() > dark.std()
    assert apply_clahe(dark[..., 0]).ndim == 2


def test_mirror_is_horizontal_flip() -> None:
    img = np.zeros((2, 3, 3), dtype=np.uint8)
    img[:, 0] = 255
    assert mirror_for_display(img)[:, 2].min() == 255


def test_roi_contains_points_with_padding_and_clamps() -> None:
    roi = roi_from_points([(0.4, 0.3), (0.6, 0.5)], 640, 480, padding=0.1)
    assert roi.x < 0.4 * 640 and roi.x + roi.w > 0.6 * 640
    assert roi.y < 0.3 * 480 and roi.y + roi.h > 0.5 * 480

    edge = roi_from_points([(0.0, 0.0), (0.05, 0.05)], 640, 480, padding=0.5)
    assert edge.x == 0 and edge.y == 0
    assert edge.x + edge.w <= 640 and edge.y + edge.h <= 480

    single = roi_from_points([(0.5, 0.5)], 640, 480, min_size_px=32)
    assert single.w >= 32 and single.h >= 32

    with pytest.raises(ValueError):
        roi_from_points([], 640, 480)


def test_roi_coordinate_round_trip() -> None:
    roi = Roi(x=100, y=50, w=200, h=100, frame_w=640, frame_h=480)
    fx, fy = roi.to_full_norm(0.5, 0.5)
    assert (fx, fy) == pytest.approx((200 / 640, 100 / 480))
    assert roi.from_full_norm(fx, fy) == pytest.approx((0.5, 0.5))


def test_crop_roi_upsamples_and_maps_content() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:150, 200:260] = 255
    roi = Roi(x=190, y=90, w=80, h=70, frame_w=640, frame_h=480)
    crop = crop_roi(img, roi, scale=2.0)
    assert crop.shape == (140, 160, 3)
    assert crop[70, 80].min() == 255  # centre of the white patch
    assert crop_roi(img, roi).shape == (70, 80, 3)
    with pytest.raises(ValueError):
        crop_roi(img[:100], roi)
    with pytest.raises(ValueError):
        crop_roi(img, roi, scale=0)
