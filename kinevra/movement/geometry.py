"""2D joint geometry in image pixel space (y points down).

Always convert normalised landmarks to pixels (`to_px`) before measuring angles: normalised
coordinates squash one axis on non-square frames and distort angles (see tests).
"""

from __future__ import annotations

import math

from kinevra.schemas import Landmark

Point = tuple[float, float]
_EPS = 1e-9


def to_px(lm: Landmark, width: int, height: int) -> Point:
    return (lm.x * width, lm.y * height)


def midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_deg(a: Point, b: Point, c: Point) -> float | None:
    """Unsigned angle a-b-c at vertex b in degrees (0..180); None if a side has zero length.

    Uses atan2(|cross|, dot), which stays accurate near 0° and 180° (unlike acos).
    """
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    if math.hypot(*v1) < _EPS or math.hypot(*v2) < _EPS:
        return None
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return min(180.0, max(0.0, math.degrees(math.atan2(abs(cross), dot))))


def elbow_flexion_deg(interior_deg: float | None) -> float | None:
    """Flexion = 180° - interior shoulder-elbow-wrist angle, so a straight arm is 0°."""
    return None if interior_deg is None else 180.0 - interior_deg


def signed_lean_deg(mid_hip: Point, mid_shoulder: Point, lateral: Point) -> float | None:
    """Angle of the mid-hip → mid-shoulder line from the IMAGE VERTICAL, in degrees.

    `lateral` (e.g. exercising hip → other hip) only decides the sign, via its horizontal
    direction: positive = the trunk leans AWAY from the exercising arm (the typical
    compensation), independent of camera mirroring. The magnitude never depends on the hip
    line, so a tilted pelvis does not distort it. None if degenerate.
    """
    v = (mid_shoulder[0] - mid_hip[0], mid_shoulder[1] - mid_hip[1])
    if math.hypot(*v) < _EPS or abs(lateral[0]) < _EPS:
        return None
    sideways = v[0] if lateral[0] > 0 else -v[0]
    upward = -v[1]  # image up is -y
    return math.degrees(math.atan2(sideways, upward))
