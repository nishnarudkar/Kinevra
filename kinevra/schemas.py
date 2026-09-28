"""Shared data contracts for Kinevra (PROJECT.md §6).

All data crossing module boundaries uses these models. Change a schema deliberately and
update every consumer and test in the same change.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

Side = Literal["left", "right"]


class Landmark(BaseModel):
    name: str
    x: float  # normalised 0..1
    y: float
    visibility: float = Field(ge=0, le=1)


class PoseFrame(BaseModel):
    session_id: str
    frame_idx: int
    t: float  # seconds since session start
    landmarks: dict[str, Landmark]
    person_count: int
    frame_quality: float  # 0..1
    quality_flags: list[str]  # low_light, blurry, out_of_frame, multiple_people


class FrameFeatures(BaseModel):
    frame_idx: int
    t: float
    side: Side
    shoulder_abduction_deg: float | None
    elbow_flexion_deg: float | None
    trunk_lean_deg: float | None
    angular_velocity_dps: float | None
    confidence: float


class RepQuality(str, Enum):
    GOOD = "GOOD"
    DEVIATION = "DEVIATION"
    UNCERTAIN = "UNCERTAIN"


class RepMetrics(BaseModel):
    rep_index: int
    side: Side
    t_start: float
    t_peak: float
    t_end: float
    min_angle: float
    max_angle: float
    rom: float
    duration_s: float
    mean_velocity_dps: float
    peak_velocity_dps: float
    smoothness: float
    max_elbow_flexion: float
    max_trunk_lean: float
    confidence: float
    rule_quality: RepQuality
    rule_reasons: list[str]


class SessionEvidence(BaseModel):
    session_id: str
    mode: Literal["live", "clip"]
    exercise: Literal["shoulder_abduction"]
    side: Side
    reps_completed: int
    reps_target: int | None
    latest_rep: RepMetrics | None
    baseline_rom: float | None
    recent_roms: list[float]
    rom_trend: Literal["increasing", "stable", "decreasing", "insufficient_data"]
    consecutive_deviations: int
    movement_consistency: float
    tracking_confidence: float
    data_quality_flags: list[str]


class AgentAction(str, Enum):
    CONTINUE = "CONTINUE_MONITORING"
    FEEDBACK = "PROVIDE_FEEDBACK"
    CAMERA_ADJUST = "REQUEST_CAMERA_ADJUSTMENT"
    LOG_EVENT = "LOG_EVENT"
    HUMAN_REVIEW = "REQUEST_HUMAN_REVIEW"


class TraceStep(BaseModel):  # one entry per node / tool call
    step: int
    node: str  # triage | reason | tool | guard | act
    tool: str | None
    tool_input: dict[str, Any] | None
    tool_output_summary: dict[str, Any] | None
    changed_assessment: bool  # did this OpenCV result change the working assessment?
    latency_ms: int


class AgentDecision(BaseModel):
    session_id: str
    rep_index: int | None
    action: AgentAction
    rationale: str  # short, cites evidence numbers, no diagnosis
    feedback_text: str | None
    evidence_refs: dict[str, Any]
    trace: list[TraceStep]
    guardrail_overrides: list[str] = Field(default_factory=list)
    llm_used: bool
    total_latency_ms: int


class ReviewEvent(BaseModel):
    event_id: str
    session_id: str
    reason: str
    evidence: dict[str, Any]
    snapshot_keys: list[str]  # S3 keys of annotated start/peak/end frames
    status: Literal["PENDING", "ACKNOWLEDGED", "DISMISSED"] = "PENDING"
    reviewer_note: str | None = None
    created_at: str


class ClipMetadata(BaseModel):  # sidecar JSON written next to each recorded clip
    name: str
    video_file: str
    created_at: str
    source: str  # "camera:0" or input path
    width: int
    height: int
    fps_nominal: float
    fps_measured: float
    frames: int
    duration_s: float
    side: Side
    consent: bool  # subject is the developer or a consenting volunteer
    lighting: Literal["bright", "normal", "dim", "backlit"] | None = None
    distance_m: float | None = None
    camera_angle_deg: float | None = None
    notes: str | None = None
    kinevra_version: str
    opencv_version: str
