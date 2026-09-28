"""Approved user-facing text. The agent may only say these (or a close paraphrase).

Rules for every template: movement language only (no diagnosis), short, encouraging, one
instruction, never encourage pushing through pain. Reviewed as part of docs/safety.md.
"""

from __future__ import annotations

FEEDBACK: dict[str, str] = {
    "reduced_rom": "Try to lift your arm a little higher, only as far as is comfortable.",
    "trunk_lean": "Keep your body upright and let only your arm move.",
    "elbow_flexion": "Keep your elbow straight as you lift.",
    "shoulder_elevation": "Relax your shoulder down, away from your ear, as you lift.",
    "abnormal_duration": "Move at a steady, controlled pace.",
    "encouragement": "Nice steady movement. Keep going at your own pace.",
    "rest": "Take a short rest if you need one. Stop if you feel pain.",
}

CAMERA: dict[str, str] = {
    "no_person": "Step into the camera view so your upper body and hips are visible.",
    "multiple_people": "Please make sure only you are in the camera view.",
    "low_light": "Please turn on more light or face a light source.",
    "low_contrast": "Please turn on more light or face a light source.",
    "overexposed": "Avoid bright light behind you and face the light instead.",
    "out_of_frame": "Step back or adjust the camera so your whole arm and hips are in view.",
    "too_far": "Please move a little closer to the camera.",
    "too_close": "Please step back a little from the camera.",
    "blurry": "Please hold the camera steady and check the lens is clean.",
    "generic": "Please adjust the camera so your upper body and arm are clearly visible.",
}

# Most important first: the instruction given is the first issue in this order.
CAMERA_PRIORITY = (
    "no_person",
    "multiple_people",
    "low_light",
    "low_contrast",
    "overexposed",
    "out_of_frame",
    "too_far",
    "too_close",
    "blurry",
)

# Rule reason code → feedback category.
REASON_TO_FEEDBACK = {
    "reduced_rom": "reduced_rom",
    "trunk_lean": "trunk_lean",
    "elbow_flexion": "elbow_flexion",
    "shoulder_elevation": "shoulder_elevation",
    "abnormal_duration": "abnormal_duration",
}


def camera_instruction(issues: list[str]) -> str:
    for issue in CAMERA_PRIORITY:
        if issue in issues:
            return CAMERA[issue]
    return CAMERA["generic"]


def feedback_category_for(reasons: list[str]) -> str | None:
    for reason in reasons:
        code = reason.split(":")[0].removeprefix("possible_")
        if code in REASON_TO_FEEDBACK:
            return REASON_TO_FEEDBACK[code]
    return None
