"""Agent tools (PROJECT.md §5.1). Deterministic; every call is recorded in the trace.

OpenCV 5 perception tools re-measure the video; they are how the agent "looks again":
- reanalyze_segment_roi       cv2.dnn pose in ROI mode (+CLAHE) on the rep's frames
- verify_motion_optical_flow  Farneback dense flow in the arm ROI vs the landmark path
- check_camera_setup          brightness / blur / framing / people / subject size
- render_evidence_snapshot    annotated start / peak / end frames for human review

Each tool returns a dict. `assessment` (if not None) is the tool's new working assessment:
GOOD | DEVIATION | UNCERTAIN | tracking_noise | camera_issue | camera_ok. When it differs
from the current one, the trace marks the step `changed_assessment=True`.
"""

from __future__ import annotations

import statistics
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field, ValidationError

from kinevra.agent.context import ToolContext
from kinevra.agent.templates import CAMERA_PRIORITY, FEEDBACK, camera_instruction
from kinevra.evaluation.rules import evaluate_rep
from kinevra.movement.features import raw_features
from kinevra.movement.reps import RepMeasurement
from kinevra.movement.smoothing import savgol_smooth
from kinevra.pose.base import COCO17_EDGES, required_for_side
from kinevra.pose.factory import FRAMING_JOINTS
from kinevra.schemas import Landmark, PoseFrame, RepMetrics, ReviewEvent
from kinevra.vision.overlay import draw_angle_arc, draw_hud, draw_skeleton
from kinevra.vision.preprocess import Roi, roi_from_points, to_gray
from kinevra.vision.quality import assess_frame
from kinevra.vision.types import Frame

ToolOutput = dict[str, Any]


def _error(message: str) -> ToolOutput:
    return {"ok": False, "error": message, "assessment": None}


def _require_rep(ctx: ToolContext, rep_index: int) -> RepMetrics:
    rep = ctx.rep(rep_index)
    if rep is None:
        raise LookupError(f"rep {rep_index} not found")
    return rep


def _subsample(frames: list[Frame], max_frames: int) -> list[Frame]:
    if len(frames) <= max_frames:
        return frames
    idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
    return [frames[i] for i in sorted(set(idx.tolist()))]


def _visible_points(
    poses: list[PoseFrame], names: list[str], thr: float
) -> list[tuple[float, float]]:
    return [
        (p.landmarks[n].x, p.landmarks[n].y)
        for p in poses
        for n in names
        if n in p.landmarks and p.landmarks[n].visibility >= thr
    ]


def _rep_roi(
    ctx: ToolContext, rep: RepMetrics, w: int, h: int, names: list[str], padding: float
) -> Roi | None:
    """ROI around the landmarks' whole path during the rep (or the last good pose before)."""
    thr = ctx.cfg.exercise.visibility_threshold
    points = _visible_points(ctx.poses.segment(rep.t_start, rep.t_end), names, thr)
    if not points:
        before = ctx.poses.segment(rep.t_start - 3.0, rep.t_start)
        for p in reversed(before):
            pts = _visible_points([p], names, thr)
            if len(pts) == len(names):
                points = pts
                break
    return roi_from_points(points, w, h, padding=padding) if points else None


# --- OpenCV 5 perception tools ------------------------------------------------------------


class ReanalyzeArgs(BaseModel):
    rep_index: int = Field(description="Repetition to re-examine (1-based).")
    scale: float | None = Field(None, ge=1, le=4, description="ROI upsampling factor.")


def reanalyze_segment_roi(ctx: ToolContext, args: ReanalyzeArgs) -> ToolOutput:
    """Re-run OpenCV 5 DNN pose on an arm ROI of the rep's frames and re-measure the rep."""
    rep = _require_rep(ctx, args.rep_index)
    if ctx.estimator is None:
        return _error("no pose estimator available")
    ex = ctx.cfg.exercise
    frames = _subsample(
        ctx.frames.frames_between(rep.t_start, rep.t_end), ex.tools.reanalyze_max_frames
    )
    if len(frames) < 3:
        return _error("rep frames are no longer available")
    h, w = frames[0].image.shape[:2]
    required = required_for_side(ex.side, FRAMING_JOINTS)
    roi = _rep_roi(ctx, rep, w, h, required, ctx.cfg.pose.roi_padding)
    if roi is None:
        return _error("no landmarks to place the ROI")

    scale = args.scale or ex.tools.reanalyze_roi_scale
    raws = []
    for f in frames:
        pose = ctx.estimator.estimate(
            f.image, f.t, f.idx, session_id=ctx.session_id, roi=roi, roi_scale=scale
        )
        raws.append(raw_features(pose, ex.side, ex.visibility_threshold))
    ts = [r.t for r in raws]
    sg = ex.smoothing.savgol
    smooth, _ = savgol_smooth(
        ts,
        [r.abduction for r in raws],
        window_s=sg.window_s,
        polyorder=sg.polyorder,
        max_gap_s=max(ex.smoothing.max_gap_s, 0.5),
    )
    valid = [a for a in smooth if a is not None]
    confidence = statistics.fmean(r.confidence for r in raws)
    out: ToolOutput = {
        "ok": True,
        "rep_index": rep.rep_index,
        "frames_analyzed": len(frames),
        "roi_px": [roi.x, roi.y, roi.w, roi.h],
        "scale": scale,
        "rom_before": round(rep.rom, 1),
        "confidence_before": round(rep.confidence, 3),
        "quality_before": rep.rule_quality.value,
        "reasons_before": rep.rule_reasons,
        "confidence_after": round(confidence, 3),
    }
    if len(valid) < 3:
        out.update(
            rom_after=None,
            quality_after="UNCERTAIN",
            reasons_after=["reanalysis_failed: arm not found in ROI"],
            assessment="UNCERTAIN",
        )
        return out

    start_level = statistics.median(valid[:3])
    peak = max(valid)
    leans = [r.trunk_lean for r in raws if r.trunk_lean is not None]
    elbows = [r.elbow_flexion for r in raws if r.elbow_flexion is not None]
    measurement = RepMeasurement(
        rep_index=rep.rep_index,
        side=rep.side,
        t_start=rep.t_start,
        t_peak=rep.t_peak,
        t_end=rep.t_end,
        min_angle=start_level,
        max_angle=peak,
        rom=peak - start_level,
        duration_s=rep.duration_s,
        mean_velocity_dps=rep.mean_velocity_dps,
        peak_velocity_dps=rep.peak_velocity_dps,
        smoothness=rep.smoothness,
        max_elbow_flexion=max(elbows) if elbows else rep.max_elbow_flexion,
        max_trunk_lean=max(leans, key=abs) if leans else rep.max_trunk_lean,
        max_shoulder_elevation=rep.max_shoulder_elevation,
        confidence=confidence,
        frames=len(frames),
        flag_fractions={},  # ROI + CLAHE re-measurement: judged on its own confidence
    )
    judged = evaluate_rep(measurement, ctx.session.state.baseline_rom, ex)
    out.update(
        rom_after=round(judged.rom, 1),
        max_trunk_lean_after=round(judged.max_trunk_lean, 1),
        max_elbow_flexion_after=round(judged.max_elbow_flexion, 1),
        quality_after=judged.rule_quality.value,
        reasons_after=judged.rule_reasons,
        assessment=judged.rule_quality.value,
    )
    return out


class RepArgs(BaseModel):
    rep_index: int = Field(description="Repetition index (1-based).")


def _elbow_px(pose: PoseFrame | None, side: str, w: int, h: int, thr: float) -> np.ndarray | None:
    if pose is None:
        return None
    lm = pose.landmarks.get(f"{side}_elbow")
    if lm is None or lm.visibility < thr:
        return None
    return np.array([lm.x * w, lm.y * h])


def _reversals(values: list[float], min_step: float = 1.0) -> int:
    signs = [np.sign(v) for v in values if abs(v) >= min_step]
    return sum(1 for a, b in pairwise(signs) if a != b)


def verify_motion_optical_flow(ctx: ToolContext, args: RepArgs) -> ToolOutput:
    """Dense optical flow (Farneback) in the arm ROI vs the elbow landmark path."""
    rep = _require_rep(ctx, args.rep_index)
    ex = ctx.cfg.exercise
    frames = ctx.frames.frames_between(rep.t_start, rep.t_end)
    pose_by_idx = {p.frame_idx: p for p in ctx.poses.segment(rep.t_start, rep.t_end)}
    matched = [f for f in frames if f.idx in pose_by_idx]
    frames = matched if len(matched) >= 3 else frames
    frames = _subsample(frames, 90)
    if len(frames) < 3:
        return _error("rep frames are no longer available")
    h, w = frames[0].image.shape[:2]
    names = [f"{ex.side}_{j}" for j in ("shoulder", "elbow", "wrist")]
    roi = _rep_roi(ctx, rep, w, h, names, ex.tools.flow_roi_padding)
    if roi is None:
        return _error("no landmarks to place the ROI")

    thr = ex.visibility_threshold
    lm_steps: list[np.ndarray] = []
    flow_steps: list[np.ndarray] = []
    prev_gray = to_gray(frames[0].image)[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    prev_elbow = _elbow_px(
        pose_by_idx.get(frames[0].idx) or ctx.poses.nearest(frames[0].t), ex.side, w, h, thr
    )
    half = 7
    for f in frames[1:]:
        gray = to_gray(f.image)[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
        elbow = _elbow_px(pose_by_idx.get(f.idx) or ctx.poses.nearest(f.t), ex.side, w, h, thr)
        if prev_elbow is not None and elbow is not None:
            init = np.zeros((*prev_gray.shape[:2], 2), np.float32)  # unused with flags=0
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, init, 0.5, 3, 15, 3, 5, 1.2, 0)
            cx = int(np.clip(prev_elbow[0] - roi.x, 0, roi.w - 1))
            cy = int(np.clip(prev_elbow[1] - roi.y, 0, roi.h - 1))
            patch = flow[max(0, cy - half) : cy + half + 1, max(0, cx - half) : cx + half + 1]
            flow_steps.append(np.median(patch.reshape(-1, 2), axis=0))
            lm_steps.append(elbow - prev_elbow)
        prev_gray, prev_elbow = gray, elbow

    if len(lm_steps) < 2:
        return _error("elbow not visible in enough frames")
    lm_arr, fl_arr = np.array(lm_steps), np.array(flow_steps)
    lm_path = float(np.linalg.norm(lm_arr, axis=1).sum())
    fl_path = float(np.linalg.norm(fl_arr, axis=1).sum())
    ratio = lm_path / max(fl_path, 1e-6)
    moving = (np.linalg.norm(fl_arr, axis=1) > 0.5) & (np.linalg.norm(lm_arr, axis=1) > 0.5)
    cos = [
        float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        for a, b in zip(lm_arr[moving], fl_arr[moving], strict=True)
    ]
    jitter = ratio > ex.tools.flow_jitter_ratio
    return {
        "ok": True,
        "rep_index": rep.rep_index,
        "frame_pairs": len(lm_steps),
        "roi_px": [roi.x, roi.y, roi.w, roi.h],
        "landmark_path_px": round(lm_path, 1),
        "flow_path_px": round(fl_path, 1),
        "jitter_ratio": round(ratio, 2),
        "direction_agreement": round(statistics.fmean(cos), 2) if cos else None,
        "landmark_reversals": _reversals(lm_arr[:, 1].tolist()),
        "flow_reversals": _reversals(fl_arr[:, 1].tolist()),
        "verdict": "landmark_jitter" if jitter else "real_motion",
        "assessment": "tracking_noise" if jitter else None,
    }


class CameraArgs(BaseModel):
    window_s: float = Field(3.0, gt=0, le=20, description="Seconds of recent video to check.")


def check_camera_setup(ctx: ToolContext, args: CameraArgs) -> ToolOutput:
    """Brightness, blur, framing, person count and subject size over the recent window."""
    ex = ctx.cfg.exercise
    poses = ctx.poses.window(args.window_s)
    if not poses:
        return _error("no recent frames")
    frames = _subsample(ctx.frames.frames_between(poses[0].t, poses[-1].t), 15)
    counts: dict[str, int] = {}
    if frames:  # re-measure image quality with OpenCV
        for f in frames:
            for flag in assess_frame(f.image, ctx.cfg.quality).flags:
                counts[flag] = counts.get(flag, 0) + 1
        image_n = len(frames)
    else:
        image_n = 0
    fractions: dict[str, float] = {}
    for flag in ("low_light", "overexposed", "low_contrast", "blurry"):
        if image_n:
            fractions[flag] = counts.get(flag, 0) / image_n
        else:
            fractions[flag] = sum(flag in p.quality_flags for p in poses) / len(poses)
    for flag in ("no_person", "multiple_people", "out_of_frame"):
        fractions[flag] = sum(flag in p.quality_flags for p in poses) / len(poses)

    torso: list[float] = []
    thr = ex.visibility_threshold
    for p in poses:
        pts = [
            p.landmarks.get(n) for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        ]
        if all(lm is not None and lm.visibility >= thr for lm in pts):
            ls, rs, lh, rh = pts
            assert ls and rs and lh and rh
            torso.append(abs((lh.y + rh.y) / 2 - (ls.y + rs.y) / 2))
    torso_frac = statistics.median(torso) if torso else None

    limit = ex.tools.camera_issue_fraction
    issues = [flag for flag in CAMERA_PRIORITY if fractions.get(flag, 0.0) > limit]
    lo, hi = ex.tools.subject_torso_frac
    if torso_frac is not None and torso_frac < lo:
        issues.append("too_far")
    if torso_frac is not None and torso_frac > hi:
        issues.append("too_close")
    issues = [i for i in CAMERA_PRIORITY if i in issues]
    return {
        "ok": True,
        "frames_checked": max(image_n, len(poses)),
        "fractions": {k: round(v, 2) for k, v in fractions.items() if v > 0},
        "torso_frac": round(torso_frac, 3) if torso_frac is not None else None,
        "issues": issues,
        "instruction": camera_instruction(issues) if issues else None,
        "assessment": "camera_issue" if issues else "camera_ok",
    }


def _annotate(
    ctx: ToolContext, frame: Frame, pose: PoseFrame | None, label: str, rep: RepMetrics
) -> np.ndarray:
    ex = ctx.cfg.exercise
    h, w = frame.image.shape[:2]
    view = frame.image
    lines = [
        f"rep {rep.rep_index} - {label} - t {frame.t:.1f}s",
        f"ROM {rep.rom:.0f} deg   rule {rep.rule_quality.value}",
    ]
    if pose is not None:
        lms = {
            n: Landmark(name=n, x=lm.x, y=lm.y, visibility=lm.visibility)
            for n, lm in pose.landmarks.items()
        }
        view = draw_skeleton(
            view,
            lms,
            COCO17_EDGES,
            visibility_threshold=ex.visibility_threshold,
            highlight_side=ex.side,
        )
        raw = raw_features(pose, ex.side, ex.visibility_threshold)
        names = (f"{ex.side}_hip", f"{ex.side}_shoulder", f"{ex.side}_elbow")
        if raw.abduction is not None and all(n in lms for n in names):
            hip, sh, el = ((lms[n].x * w, lms[n].y * h) for n in names)
            view = draw_angle_arc(view, sh, hip, el, raw.abduction)
            lines.append(f"abduction {raw.abduction:.0f} deg")
    return draw_hud(view, lines, None)


def render_evidence_snapshot(ctx: ToolContext, args: RepArgs) -> ToolOutput:
    """Annotated start / peak / end frames of a rep, saved for the human reviewer."""
    rep = _require_rep(ctx, args.rep_index)
    frames = ctx.frames.frames_between(rep.t_start, rep.t_end)
    if not frames:
        return _error("rep frames are no longer available")
    keys = []
    for label, t in (("start", rep.t_start), ("peak", rep.t_peak), ("end", rep.t_end)):
        frame = min(frames, key=lambda f: abs(f.t - t))
        image = _annotate(ctx, frame, ctx.poses.nearest(frame.t), label, rep)
        ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return _error("jpeg encoding failed")
        key = f"{ctx.session_id}/rep{rep.rep_index:04d}_{label}.jpg"
        keys.append(ctx.store.put_bytes(key, jpg.tobytes(), "image/jpeg"))
    ctx.snapshots[rep.rep_index] = keys
    return {"ok": True, "rep_index": rep.rep_index, "snapshot_keys": keys, "assessment": None}


# --- analysis tools -----------------------------------------------------------------------


def _rep_summary(r: RepMetrics) -> dict[str, Any]:
    return {
        "rep_index": r.rep_index,
        "rom": round(r.rom, 1),
        "duration_s": round(r.duration_s, 2),
        "confidence": round(r.confidence, 2),
        "smoothness": round(r.smoothness, 2),
        "max_trunk_lean": round(r.max_trunk_lean, 1),
        "max_elbow_flexion": round(r.max_elbow_flexion, 1),
        "quality": r.rule_quality.value,
        "reasons": r.rule_reasons,
    }


class RecentArgs(BaseModel):
    k: int = Field(3, ge=1, le=10, description="Number of most recent reps.")


def analyze_recent_repetitions(ctx: ToolContext, args: RecentArgs) -> ToolOutput:
    reps = ctx.session.state.reps[-args.k :]
    return {"ok": True, "reps": [_rep_summary(r) for r in reps], "assessment": None}


class NoArgs(BaseModel):
    pass


def check_movement_consistency(ctx: ToolContext, args: NoArgs) -> ToolOutput:
    s = ctx.session.state
    return {
        "ok": True,
        "consistency": round(s.consistency(), 3),
        "rom_trend": s.rom_trend(),
        "recent_roms": [round(r, 1) for r in s.recent_roms],
        "assessment": None,
    }


def calculate_movement_metrics(ctx: ToolContext, args: RepArgs) -> ToolOutput:
    return {"ok": True, **_rep_summary(_require_rep(ctx, args.rep_index)), "assessment": None}


def compare_with_session_baseline(ctx: ToolContext, args: NoArgs) -> ToolOutput:
    s = ctx.session.state
    latest = s.reps[-1] if s.reps else None
    ratio = latest.rom / s.baseline_rom if latest and s.baseline_rom else None
    return {
        "ok": True,
        "baseline_rom": s.baseline_rom,
        "latest_rom": round(latest.rom, 1) if latest else None,
        "latest_to_baseline": round(ratio, 3) if ratio is not None else None,
        "rom_trend": s.rom_trend(),
        "consecutive_deviations": s.consecutive_deviations,
        "assessment": None,
    }


class FeedbackArgs(BaseModel):
    category: str = Field(description=f"One of: {', '.join(FEEDBACK)}.")


def generate_feedback(ctx: ToolContext, args: FeedbackArgs) -> ToolOutput:
    if args.category not in FEEDBACK:
        return _error(f"unknown category; choose from {sorted(FEEDBACK)}")
    return {
        "ok": True,
        "category": args.category,
        "text": FEEDBACK[args.category],
        "assessment": None,
    }


class LogArgs(BaseModel):
    type: str = Field(description="Short event type, e.g. 'uncertain_rep'.")
    payload: dict[str, Any] = Field(default_factory=dict)


def log_event(ctx: ToolContext, args: LogArgs) -> ToolOutput:
    event = {
        "type": args.type,
        "payload": args.payload,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    ctx.logged_events.append(event)
    return {"ok": True, "logged": args.type, "assessment": None}


class ReviewArgs(BaseModel):
    reason: str = Field(description="Why a human should review (movement terms, numbers).")
    evidence: dict[str, Any] = Field(default_factory=dict)
    snapshot_keys: list[str] = Field(default_factory=list)


def request_human_review(ctx: ToolContext, args: ReviewArgs) -> ToolOutput:
    keys = list(args.snapshot_keys)
    rep_index = args.evidence.get("rep_index")
    if not keys and isinstance(rep_index, int):
        keys = ctx.snapshots.get(rep_index, [])
    event = ReviewEvent(
        event_id=uuid.uuid4().hex[:12],
        session_id=ctx.session_id,
        reason=args.reason,
        evidence=args.evidence,
        snapshot_keys=keys,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    ctx.review_events.append(event)
    return {"ok": True, "event_id": event.event_id, "snapshot_keys": keys, "assessment": None}


# --- registry -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[ToolContext, Any], ToolOutput]
    opencv: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model.model_json_schema(),
        }


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in (
        Tool(
            "reanalyze_segment_roi",
            "OpenCV 5: re-run DNN pose on a high-resolution arm ROI of a rep and re-measure "
            "ROM/posture. Use when a rep's confidence is borderline or a deviation might be "
            "tracking noise. Returns quality_after (confirmed or dismissed).",
            ReanalyzeArgs,
            reanalyze_segment_roi,
            opencv=True,
        ),
        Tool(
            "verify_motion_optical_flow",
            "OpenCV 5: dense optical flow in the arm ROI compared with the elbow landmark path. "
            "Separates real movement irregularity from landmark jitter.",
            RepArgs,
            verify_motion_optical_flow,
            opencv=True,
        ),
        Tool(
            "check_camera_setup",
            "OpenCV 5: check brightness, blur, framing, person count and subject size over the "
            "last seconds. Returns issues and one camera instruction.",
            CameraArgs,
            check_camera_setup,
            opencv=True,
        ),
        Tool(
            "render_evidence_snapshot",
            "OpenCV 5: render annotated start/peak/end frames of a rep for a human reviewer.",
            RepArgs,
            render_evidence_snapshot,
            opencv=True,
        ),
        Tool(
            "analyze_recent_repetitions",
            "Summaries of the last k repetitions.",
            RecentArgs,
            analyze_recent_repetitions,
        ),
        Tool(
            "check_movement_consistency",
            "ROM consistency and trend over recent reps.",
            NoArgs,
            check_movement_consistency,
        ),
        Tool(
            "calculate_movement_metrics",
            "All measured metrics of one repetition.",
            RepArgs,
            calculate_movement_metrics,
        ),
        Tool(
            "compare_with_session_baseline",
            "Latest ROM relative to the session baseline.",
            NoArgs,
            compare_with_session_baseline,
        ),
        Tool(
            "generate_feedback",
            "Approved feedback text for a category.",
            FeedbackArgs,
            generate_feedback,
        ),
        Tool("log_event", "Record an event in the session log.", LogArgs, log_event),
        Tool(
            "request_human_review",
            "Create a human-review event with evidence numbers and snapshot keys.",
            ReviewArgs,
            request_human_review,
        ),
    )
}


def tool_schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in TOOLS.values()]


def execute_tool(
    ctx: ToolContext,
    name: str,
    args: dict[str, Any],
    stubs: dict[str, ToolOutput] | None = None,
) -> tuple[ToolOutput, float]:
    """Validate args and run a tool (or its scripted stub). Returns (output, latency_ms)."""
    start = time.perf_counter()
    tool = TOOLS.get(name)
    if tool is None:
        return _error(f"unknown tool {name!r}"), 0.0
    try:
        parsed = tool.args_model.model_validate(args)
        if stubs is not None and name in stubs:
            out = dict(stubs[name])
            out.setdefault("ok", True)
            out.setdefault("assessment", None)
            if name == "request_human_review":  # still create the real review event
                out = {**tool.fn(ctx, parsed), **out}
        else:
            out = tool.fn(ctx, parsed)
    except ValidationError as exc:
        out = _error(f"invalid arguments: {exc.errors()[0]['msg']}")
    except LookupError as exc:
        out = _error(str(exc))
    return out, (time.perf_counter() - start) * 1000.0
