"""Geometry tests (written before the implementation, PROJECT.md Phase 3)."""

import math

import pytest

from kinevra.movement.geometry import (
    angle_deg,
    distance,
    elbow_flexion_deg,
    midpoint,
    signed_lean_deg,
    to_px,
)
from kinevra.schemas import Landmark

# --- angle_deg: vertex is the middle point ------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "c", "expected"),
    [
        ((0, 1), (0, 0), (0, 1), 0.0),  # same direction
        ((0, 1), (0, 0), (1, 0), 90.0),  # right angle
        ((0, 1), (0, 0), (0, -1), 180.0),  # straight line
        ((1, 0), (0, 0), (1, 1), 45.0),
        ((1, 0), (0, 0), (-1, 1), 135.0),
        ((5, 5), (2, 2), (5, 2), 45.0),  # translated vertex
    ],
)
def test_known_angles(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], expected: float
) -> None:
    assert angle_deg(a, b, c) == pytest.approx(expected, abs=1e-9)


def test_angle_is_symmetric_and_unsigned() -> None:
    assert angle_deg((1, 0), (0, 0), (0, 1)) == angle_deg((0, 1), (0, 0), (1, 0)) == 90.0
    assert angle_deg((1, 0), (0, 0), (0, -1)) == pytest.approx(90.0)


def test_angle_scale_invariant_and_clamped() -> None:
    assert angle_deg((1000, 0), (0, 0), (0, 0.001)) == pytest.approx(90.0)
    assert 0.0 <= angle_deg((1, 1e-12), (0, 0), (1, 0)) <= 180.0  # type: ignore[operator]


def test_degenerate_angle_is_none() -> None:
    assert angle_deg((0, 0), (0, 0), (1, 0)) is None
    assert angle_deg((1, 0), (0, 0), (0, 0)) is None


# --- aspect-ratio correction ----------------------------------------------------------------


def test_aspect_ratio_correction() -> None:
    """A 45° pixel angle is NOT 45° in normalised coords on a 16:9 frame."""
    w, h = 1280, 720

    def lm(n: str, x_px: float, y_px: float) -> Landmark:
        return Landmark(name=n, x=x_px / w, y=y_px / h, visibility=1.0)

    vertex, a, c = lm("v", 640, 360), lm("a", 740, 360), lm("c", 740, 260)
    in_px = angle_deg(to_px(a, w, h), to_px(vertex, w, h), to_px(c, w, h))
    in_norm = angle_deg((a.x, a.y), (vertex.x, vertex.y), (c.x, c.y))
    assert in_px == pytest.approx(45.0)
    assert in_norm is not None and abs(in_norm - 45.0) > 10  # uncorrected would be wrong


def test_to_px() -> None:
    assert to_px(Landmark(name="n", x=0.5, y=0.25, visibility=1), 640, 480) == (320.0, 120.0)


# --- helpers --------------------------------------------------------------------------------


def test_midpoint_and_distance() -> None:
    assert midpoint((0, 0), (4, 2)) == (2.0, 1.0)
    assert distance((0, 0), (3, 4)) == 5.0


@pytest.mark.parametrize(("interior", "flexion"), [(180, 0), (150, 30), (90, 90), (None, None)])
def test_elbow_flexion_from_interior(interior: float | None, flexion: float | None) -> None:
    assert elbow_flexion_deg(interior) == flexion


# --- trunk lean (image y points down) -----------------------------------------------------


def test_upright_trunk_has_zero_lean() -> None:
    assert signed_lean_deg((100, 200), (100, 100), lateral=(1, 0)) == pytest.approx(0.0)


def test_lean_sign_follows_lateral_direction() -> None:
    # shoulders shifted toward +x; lateral axis points +x (towards the non-exercising side)
    away = signed_lean_deg((100, 200), (110, 100), lateral=(1, 0))
    toward = signed_lean_deg((100, 200), (90, 100), lateral=(1, 0))
    assert away is not None and toward is not None
    assert away == pytest.approx(math.degrees(math.atan2(10, 100)))
    assert toward == pytest.approx(-away)
    # flipping the lateral axis (other arm exercising / mirrored view) flips the sign
    assert signed_lean_deg((100, 200), (110, 100), lateral=(-1, 0)) == pytest.approx(-away)


def test_lean_45_degrees_and_degenerate() -> None:
    assert signed_lean_deg((0, 100), (100, 0), lateral=(1, 0)) == pytest.approx(45.0)
    assert signed_lean_deg((0, 0), (0, 0), lateral=(1, 0)) is None
