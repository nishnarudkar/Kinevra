"""Decoding tests for the OpenCV DNN pose models — no camera and no model files needed."""

from pathlib import Path

import numpy as np
import pytest

from kinevra.pose.base import COCO17
from kinevra.pose.mediapipe_dnn import (
    BLAZEPOSE_TO_COCO17,
    MPPersonDetector,
    generate_anchors,
    person_affine,
)
from kinevra.pose.opencv_dnn import (
    apply_inverse_affine,
    box_to_center_size,
    crop_affine,
    decode_rtmdet,
    decode_simcc,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


# --- RTMPose ------------------------------------------------------------------------------


def test_decode_simcc_known_peaks() -> None:
    k = 3
    sx = np.zeros((k, 384), dtype=np.float32)
    sy = np.zeros((k, 512), dtype=np.float32)
    for i, (x, y, px, py) in enumerate(
        [(10, 20, 0.9, 0.8), (100, 300, 0.4, 0.7), (383, 511, 1.2, 1.1)]
    ):
        sx[i, x], sy[i, y] = px, py
    kps = decode_simcc(sx, sy, split_ratio=2.0)
    assert kps[:, 0].tolist() == [5.0, 50.0, 191.5]
    assert kps[:, 1].tolist() == [10.0, 150.0, 255.5]
    assert kps[:, 2] == pytest.approx([0.8, 0.4, 1.1])  # min of the two axis maxima


def test_box_to_center_size_fixes_aspect() -> None:
    cx, cy, w, h = box_to_center_size((0, 0, 100, 100), padding=1.0, aspect=0.75)
    assert (cx, cy) == (50, 50)
    assert w == 100 and h == pytest.approx(100 / 0.75)
    cx, cy, w, h = box_to_center_size((10, 20, 40, 220), padding=1.25, aspect=0.75)
    assert w / h == pytest.approx(0.75) and h == pytest.approx(250)


def test_crop_affine_round_trip() -> None:
    affine = crop_affine(200, 150, 120, 160, 192, 256)
    corners_in = np.array([[0, 0, 1], [192, 256, 1], [96, 128, 1]], dtype=np.float32)
    back = apply_inverse_affine(corners_in, affine)
    assert back[:, :2] == pytest.approx(np.array([[140, 70], [260, 230], [200, 150]]), abs=1e-4)
    assert back[:, 2].tolist() == [1, 1, 1]


def test_decode_rtmdet_filters_scales_nms_and_sorts() -> None:
    raw = np.array(
        [
            [10, 10, 50, 90],  # person
            [12, 12, 52, 92],  # overlaps the first, lower score → removed by NMS
            [100, 20, 150, 120],  # person, highest score
            [200, 20, 250, 120],  # below threshold
        ],
        dtype=np.float32,
    )
    scores = np.array([0.60, 0.55, 0.90, 0.30], dtype=np.float32)
    out = decode_rtmdet(
        raw, scores, ratio=0.5, image_size=(320, 240), score_threshold=0.5, nms_threshold=0.45
    )
    assert out.shape == (2, 5)
    assert out[:, 4].tolist() == pytest.approx([0.9, 0.6])
    assert out[0, :4].tolist() == [200, 40, 300, 240]  # /0.5, y clipped to height 240
    empty = decode_rtmdet(raw, np.zeros(4, np.float32), 1.0, (10, 10), 0.5, 0.5)
    assert empty.shape == (0, 5)


def test_real_rtmpose_output_fixture() -> None:
    """Raw SimCC output of RTMPose-s on a real frontal image (stored, no image data)."""
    d = np.load(FIXTURES / "rtmpose_simcc_sample.npz")
    kps = apply_inverse_affine(decode_simcc(d["simcc_x"], d["simcc_y"], 2.0), d["affine"])
    assert kps == pytest.approx(d["expected"], abs=1e-3)
    idx = {n: i for i, n in enumerate(COCO17)}
    # frontal view: the subject's right side appears on the image's left
    assert kps[idx["right_shoulder"], 0] < kps[idx["left_shoulder"], 0]
    assert kps[idx["right_wrist"], 0] < kps[idx["left_wrist"], 0]
    # anatomical ordering: shoulders above hips, all keypoints inside the detected box region
    assert kps[idx["right_shoulder"], 1] < kps[idx["right_hip"], 1]
    x0, y0, x1, y1 = d["det_box"]
    assert (kps[:, 0] > x0 - 40).all() and (kps[:, 0] < x1 + 40).all()
    assert (kps[:, 1] > y0 - 40).all() and (kps[:, 1] < y1 + 40).all()
    assert kps[idx["right_wrist"], 2] > 0.5


# --- MediaPipe ----------------------------------------------------------------------------


def test_generate_anchors_matches_mediapipe_layout() -> None:
    a = generate_anchors()
    assert a.shape == (2254, 2)
    assert a[0] == pytest.approx([0.5 / 28, 0.5 / 28])
    assert (a[0] == a[1]).all()  # 2 anchors per cell at stride 8
    assert a[28 * 28 * 2] == pytest.approx([0.5 / 14, 0.5 / 14])  # stride 16 starts
    assert a[-1] == pytest.approx([6.5 / 7, 6.5 / 7])
    assert (a[-6:] == a[-1]).all()  # 3 stride-32 layers x 2 anchors merged


def test_mp_detector_decode() -> None:
    det = MPPersonDetector.__new__(MPPersonDetector)  # decode only, no network
    det.anchors = generate_anchors()
    det.score_threshold, det.nms_threshold = 0.5, 0.3
    n = len(det.anchors)
    reg = np.zeros((n, 12), dtype=np.float32)
    scores = np.full(n, 0.1)
    i = 1000
    reg[i, 2:4] = [22.4, 44.8]  # w, h = 0.1, 0.2 of input
    reg[i, 4:12] = [0, 11.2, 0, -22.4, 0, 0, 0, 0]  # mid-hip below, full-body above anchor
    scores[i] = 0.9
    out = det.decode(reg, scores, scale=448.0, pad=np.array([0.0, 0.0]))
    assert out.shape == (1, 13)
    ax, ay = det.anchors[i] * 448
    assert out[0, :4] == pytest.approx([ax - 22.4, ay - 44.8, ax + 22.4, ay + 44.8])
    assert out[0, 4:6] == pytest.approx([ax, ay + 22.4])
    assert out[0, 12] == pytest.approx(0.9)
    scores[i] = 0.2
    assert det.decode(reg, scores, 448.0, np.zeros(2)).shape == (0, 13)


def test_person_affine_upright_and_rotated() -> None:
    hip, top = np.array([300.0, 400.0]), np.array([300.0, 200.0])
    a = person_affine(hip, top, 256)
    assert a @ np.array([*hip, 1]) == pytest.approx([128, 128])
    assert a @ np.array([*top, 1]) == pytest.approx([128, 0])  # 2·dist square → top edge
    # person lying to the right: full-body point to the right of the hip → rotated upright
    a = person_affine(hip, np.array([500.0, 400.0]), 256)
    assert a @ np.array([500.0, 400.0, 1]) == pytest.approx([128, 0], abs=1e-6)


def test_blazepose_mapping_covers_coco17() -> None:
    assert len(BLAZEPOSE_TO_COCO17) == len(COCO17) == 17
    idx = dict(zip(COCO17, BLAZEPOSE_TO_COCO17, strict=True))
    assert idx["left_shoulder"] == 11 and idx["right_shoulder"] == 12
    assert idx["left_hip"] == 23 and idx["right_wrist"] == 16
