"""Drawing helpers for the live window and evidence frames. Functions never mutate inputs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import cv2
import numpy as np

from kinevra.schemas import Landmark
from kinevra.vision.quality import FrameQuality
from kinevra.vision.types import Image

# BGR
WHITE = (255, 255, 255)
GREEN = (80, 200, 80)
AMBER = (0, 180, 255)
RED = (60, 60, 230)
PANEL = (30, 30, 30)

FONT = cv2.FONT_HERSHEY_SIMPLEX
DISCLAIMER = "Assistive prototype. Not a medical device. Stop if you feel pain."


def quality_colour(score: float) -> tuple[int, int, int]:
    if score >= 0.7:
        return GREEN
    if score >= 0.4:
        return AMBER
    return RED


def _panel(image: Image, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    """Semi-transparent dark rectangle, drawn in place on `image`."""
    h_img, w_img = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    region = image[y0:y1, x0:x1]
    shade = np.full_like(region, PANEL)
    image[y0:y1, x0:x1] = cv2.addWeighted(shade, alpha, region, 1 - alpha, 0)


def _text(
    image: Image, text: str, org: tuple[int, int], colour: tuple[int, int, int], scale: float = 0.5
) -> None:
    cv2.putText(image, text, org, FONT, scale, colour, 1, cv2.LINE_AA)


def draw_hud(
    image: Image,
    lines: Sequence[str],
    quality: FrameQuality | None = None,
    *,
    show_disclaimer: bool = True,
) -> Image:
    out = image.copy()
    h, w = out.shape[:2]
    line_h = 20

    if lines:
        _panel(out, 8, 8, 260, 10 + line_h * len(lines))
        for i, line in enumerate(lines):
            _text(out, line, (16, 26 + i * line_h), WHITE)

    if quality is not None:
        colour = quality_colour(quality.score)
        rows = [f"quality {quality.score:.2f}", *quality.flags]
        box_w = 170
        _panel(out, w - box_w - 8, 8, box_w, 10 + line_h * len(rows))
        cv2.circle(out, (w - box_w + 4, 22), 6, colour, -1, cv2.LINE_AA)
        _text(out, rows[0], (w - box_w + 16, 26), colour)
        for i, flag in enumerate(rows[1:], start=1):
            _text(out, flag, (w - box_w + 16, 26 + i * line_h), RED)

    if show_disclaimer:
        _panel(out, 0, h - 24, w, 24)
        _text(out, DISCLAIMER, (8, h - 8), WHITE, 0.42)
    return out


def draw_recording(image: Image, label: str = "REC") -> Image:
    out = image.copy()
    w = out.shape[1]
    cv2.circle(out, (w // 2 - 30, 22), 8, RED, -1, cv2.LINE_AA)
    _text(out, label, (w // 2 - 16, 28), RED, 0.6)
    return out


def draw_countdown(image: Image, seconds_left: int) -> Image:
    out = image.copy()
    h, w = out.shape[:2]
    text = str(seconds_left)
    (tw, th), _ = cv2.getTextSize(text, FONT, 4.0, 6)
    cv2.putText(out, text, ((w - tw) // 2, (h + th) // 2), FONT, 4.0, AMBER, 6, cv2.LINE_AA)
    return out


def visibility_colour(visibility: float, threshold: float) -> tuple[int, int, int]:
    if visibility >= threshold:
        return GREEN
    if visibility >= threshold / 2:
        return AMBER
    return RED


def draw_skeleton(
    image: Image,
    landmarks: Mapping[str, Landmark],
    edges: Iterable[tuple[str, str]],
    *,
    visibility_threshold: float = 0.5,
    highlight_side: str | None = None,
) -> Image:
    """Skeleton coloured by landmark visibility; the exercising side is drawn thicker."""
    out = image.copy()
    h, w = out.shape[:2]

    def px(lm: Landmark) -> tuple[int, int]:
        return round(lm.x * w), round(lm.y * h)

    for a, b in edges:
        la, lb = landmarks.get(a), landmarks.get(b)
        if la is None or lb is None:
            continue
        vis = min(la.visibility, lb.visibility)
        side = highlight_side is not None and a.startswith(highlight_side)
        side = side and highlight_side is not None and b.startswith(highlight_side)
        colour = visibility_colour(vis, visibility_threshold)
        cv2.line(out, px(la), px(lb), colour, 4 if side else 2, cv2.LINE_AA)
    for name, lm in landmarks.items():
        big = highlight_side is not None and name.startswith(highlight_side)
        colour = visibility_colour(lm.visibility, visibility_threshold)
        cv2.circle(out, px(lm), 5 if big else 3, colour, -1, cv2.LINE_AA)
    return out


def mirror_landmarks(landmarks: Mapping[str, Landmark]) -> dict[str, Landmark]:
    """Landmarks for drawing on a mirrored preview (x → 1 - x). Display only: names are kept,
    so the subject's right arm stays `right_*` even though it appears on the other side."""
    return {n: lm.model_copy(update={"x": 1.0 - lm.x}) for n, lm in landmarks.items()}


def draw_angle_arc(
    image: Image,
    vertex: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    value_deg: float,
    *,
    colour: tuple[int, int, int] = AMBER,
    radius: int = 40,
) -> Image:
    """Arc at `vertex` between the rays to `a` and `b` (pixels), labelled with `value_deg`."""
    out = image.copy()
    vx, vy = vertex
    phi_a = math.degrees(math.atan2(a[1] - vy, a[0] - vx))
    phi_b = math.degrees(math.atan2(b[1] - vy, b[0] - vx))
    sweep = (phi_b - phi_a + 180.0) % 360.0 - 180.0  # shorter way round
    centre = (round(vx), round(vy))
    cv2.ellipse(out, centre, (radius, radius), 0.0, phi_a, phi_a + sweep, colour, 2, cv2.LINE_AA)
    mid = math.radians(phi_a + sweep / 2)
    tx, ty = vx + (radius + 16) * math.cos(mid), vy + (radius + 16) * math.sin(mid)
    label = f"{value_deg:.0f} deg"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 2)
    org = (round(tx - tw / 2), round(ty + th / 2))
    cv2.putText(out, label, org, FONT, 0.55, PANEL, 4, cv2.LINE_AA)  # dark outline
    cv2.putText(out, label, org, FONT, 0.55, colour, 2, cv2.LINE_AA)
    return out
