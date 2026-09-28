import pytest
from pydantic import BaseModel, ValidationError

from kinevra.schemas import (
    AgentAction,
    AgentDecision,
    FrameFeatures,
    Landmark,
    PoseFrame,
    RepMetrics,
    RepQuality,
    ReviewEvent,
    SessionEvidence,
    TraceStep,
)


def _rep() -> RepMetrics:
    return RepMetrics(
        rep_index=3,
        side="right",
        t_start=10.0,
        t_peak=11.2,
        t_end=12.6,
        min_angle=12.5,
        max_angle=148.0,
        rom=135.5,
        duration_s=2.6,
        mean_velocity_dps=104.2,
        peak_velocity_dps=190.0,
        smoothness=0.87,
        max_elbow_flexion=14.0,
        max_trunk_lean=4.5,
        confidence=0.91,
        rule_quality=RepQuality.GOOD,
        rule_reasons=[],
    )


def _trace() -> TraceStep:
    return TraceStep(
        step=2,
        node="tool",
        tool="reanalyze_segment_roi",
        tool_input={"rep_index": 3, "scale": 2.0},
        tool_output_summary={"rom": 121.0, "confidence": 0.84},
        changed_assessment=True,
        latency_ms=310,
    )


SAMPLES: list[BaseModel] = [
    Landmark(name="right_shoulder", x=0.41, y=0.33, visibility=0.97),
    PoseFrame(
        session_id="s1",
        frame_idx=42,
        t=2.8,
        landmarks={"right_elbow": Landmark(name="right_elbow", x=0.3, y=0.5, visibility=0.8)},
        person_count=1,
        frame_quality=0.9,
        quality_flags=["low_light"],
    ),
    FrameFeatures(
        frame_idx=42,
        t=2.8,
        side="left",
        shoulder_abduction_deg=None,
        elbow_flexion_deg=12.0,
        trunk_lean_deg=3.1,
        angular_velocity_dps=None,
        confidence=0.3,
    ),
    _rep(),
    SessionEvidence(
        session_id="s1",
        mode="clip",
        exercise="shoulder_abduction",
        side="right",
        reps_completed=4,
        reps_target=10,
        latest_rep=_rep(),
        baseline_rom=140.0,
        recent_roms=[141.0, 138.5, 135.5],
        rom_trend="decreasing",
        consecutive_deviations=1,
        movement_consistency=0.92,
        tracking_confidence=0.88,
        data_quality_flags=[],
    ),
    _trace(),
    AgentDecision(
        session_id="s1",
        rep_index=3,
        action=AgentAction.FEEDBACK,
        rationale="ROI re-analysis confirmed ROM 121° (86% of baseline 140°).",
        feedback_text="Try lifting a little higher on the next rep.",
        evidence_refs={"rep_index": 3, "rom": 121.0},
        trace=[_trace()],
        llm_used=True,
        total_latency_ms=1450,
    ),
    ReviewEvent(
        event_id="e1",
        session_id="s1",
        reason="3 consecutive confirmed deviations",
        evidence={"roms": [110.0, 108.0, 105.0]},
        snapshot_keys=["s1/rep3_peak.jpg"],
        created_at="2026-09-28T10:00:00Z",
    ),
]


@pytest.mark.parametrize("obj", SAMPLES, ids=lambda o: type(o).__name__)
def test_json_round_trip(obj: BaseModel) -> None:
    restored = type(obj).model_validate_json(obj.model_dump_json())
    assert restored == obj


def test_enums_serialise_as_values() -> None:
    decision = SAMPLES[6]
    assert '"action":"PROVIDE_FEEDBACK"' in decision.model_dump_json()
    assert '"rule_quality":"GOOD"' in _rep().model_dump_json()


def test_defaults() -> None:
    event = SAMPLES[7]
    assert isinstance(event, ReviewEvent)
    assert event.status == "PENDING"
    assert event.reviewer_note is None
    decision = SAMPLES[6]
    assert isinstance(decision, AgentDecision)
    assert decision.guardrail_overrides == []


def test_guardrail_overrides_not_shared_between_instances() -> None:
    a = AgentDecision.model_validate_json(SAMPLES[6].model_dump_json())
    b = AgentDecision.model_validate_json(SAMPLES[6].model_dump_json())
    a.guardrail_overrides.append("x")
    assert b.guardrail_overrides == []


def test_visibility_bounds() -> None:
    with pytest.raises(ValidationError):
        Landmark(name="x", x=0.0, y=0.0, visibility=1.5)


def test_invalid_action_rejected() -> None:
    data = SAMPLES[6].model_dump(mode="json")
    data["action"] = "DIAGNOSE"
    with pytest.raises(ValidationError):
        AgentDecision.model_validate(data)
