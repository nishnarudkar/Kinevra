import cv2
import numpy as np
import pytest

from kinevra.config import QualityCfg
from kinevra.vision.quality import (
    BLURRY,
    LOW_CONTRAST,
    LOW_LIGHT,
    MULTIPLE_PEOPLE,
    NO_PERSON,
    OVEREXPOSED,
    assess_frame,
    measure,
    person_flags,
)
from kinevra.vision.types import Image


def test_sharp_well_lit_image_has_no_flags(sharp_image: Image, quality_cfg: QualityCfg) -> None:
    q = assess_frame(sharp_image, quality_cfg)
    assert q.flags == []
    assert q.score == pytest.approx(1.0)


def test_covered_lens_is_dark_flat_and_blurry(quality_cfg: QualityCfg) -> None:
    rng = np.random.default_rng(0)
    covered = np.clip(rng.normal(8, 1.5, (480, 640, 3)), 0, 255).astype(np.uint8)
    q = assess_frame(covered, quality_cfg)
    assert {LOW_LIGHT, LOW_CONTRAST, BLURRY} <= set(q.flags)
    assert q.score == 0.0


def test_dimmed_scene_flags_low_light_only_when_dark(
    sharp_image: Image, quality_cfg: QualityCfg
) -> None:
    dim = (sharp_image.astype(np.float32) * 0.3).astype(np.uint8)
    q = assess_frame(dim, quality_cfg)
    assert LOW_LIGHT in q.flags
    assert q.score < assess_frame(sharp_image, quality_cfg).score


def test_overexposed(quality_cfg: QualityCfg) -> None:
    white = np.full((240, 320, 3), 250, dtype=np.uint8)
    assert OVEREXPOSED in assess_frame(white, quality_cfg).flags


def test_blur_detected(sharp_image: Image, quality_cfg: QualityCfg) -> None:
    blurred = np.asarray(cv2.GaussianBlur(sharp_image, (0, 0), 6), dtype=np.uint8)
    q_blur = assess_frame(blurred, quality_cfg)
    assert BLURRY in q_blur.flags
    assert q_blur.sharpness < assess_frame(sharp_image, quality_cfg).sharpness


def test_measure_is_resolution_independent(sharp_image: Image) -> None:
    big = np.asarray(
        cv2.resize(sharp_image, (640, 480), interpolation=cv2.INTER_NEAREST), dtype=np.uint8
    )
    b1, c1, _ = measure(sharp_image, 320)
    b2, c2, _ = measure(big, 320)
    assert b1 == pytest.approx(b2, abs=1.0)
    assert c1 == pytest.approx(c2, abs=1.0)


def test_score_bounded(quality_cfg: QualityCfg) -> None:
    rng = np.random.default_rng(1)
    for _ in range(10):
        img = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
        assert 0.0 <= assess_frame(img, quality_cfg).score <= 1.0


def test_grayscale_input(quality_cfg: QualityCfg) -> None:
    gray = np.full((120, 160), 128, dtype=np.uint8)
    assert LOW_CONTRAST in assess_frame(gray, quality_cfg).flags


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, [NO_PERSON]), (1, []), (2, [MULTIPLE_PEOPLE]), (3, [MULTIPLE_PEOPLE])],
)
def test_person_flags(count: int, expected: list[str]) -> None:
    assert person_flags(count) == expected
