"""Every guardrail rule (PROJECT.md §5.2)."""

from typing import Any

import pytest

from kinevra.agent.context import AgentMemory
from kinevra.agent.guardrails import apply_guardrails, blocked_terms
from kinevra.agent.state import DecisionContext, FinalAnswer, ToolResult
from kinevra.agent.templates import CAMERA, FEEDBACK
from kinevra.config import load_config
from kinevra.schemas import AgentAction, RepMetrics, RepQuality, SessionEvidence

CFG = load_config(env={}).exercise


def rep(
    i: int = 5,
    rom: float = 100.0,
    quality: RepQuality = RepQuality.DEVIATION,
    reasons: list[str] | None = None,
    confidence: float = 0.9,
) -> RepMetrics:
    return RepMetrics(
        rep_index=i,
        side="right",
        t_start=0,
        t_peak=1,
        t_end=2,
        min_angle=15,
        max_angle=15 + rom,
        rom=rom,
        duration_s=2.4,
        mean_velocity_dps=90,
        peak_velocity_dps=150,
        smoothness=0.95,
        max_elbow_flexion=5,
        max_trunk_lean=2,
        confidence=confidence,
        rule_quality=quality,
        rule_reasons=reasons if reasons is not None else ["reduced_rom: 100° < 80% ..."],
    )


def ctx(
    *,
    working: str = "DEVIATION",
    streak: int = 0,
    tracking: float = 0.9,
    memory: AgentMemory | None = None,
    r: RepMetrics | None = None,
    tools: list[ToolResult] | None = None,
    flags: list[str] | None = None,
) -> DecisionContext:
    r = r if r is not None else rep()
    ev = SessionEvidence(
        session_id="s",
        mode="clip",
        exercise="shoulder_abduction",
        side="right",
        reps_completed=r.rep_index,
        reps_target=None,
        latest_rep=r,
        baseline_rom=140.0,
        recent_roms=[140, 130, 100],
        rom_trend="decreasing",
        consecutive_deviations=1,
        movement_consistency=0.8,
        tracking_confidence=tracking,
        data_quality_flags=flags or [],
    )
    mem = memory or AgentMemory(confirmed_streak=streak)
    return DecisionContext(
        trigger="rep",
        evidence=ev,
        rep=r,
        triage_reasons=["x"],
        working_assessment=working,
        memory=mem,
        cfg=CFG,
        tool_results=tools or [],
    )


def fb(text: str | None = FEEDBACK["reduced_rom"], **kw: Any) -> FinalAnswer:
    return FinalAnswer(
        action="PROVIDE_FEEDBACK",
        rationale="ROM 100° vs 140°",
        feedback_text=text,
        feedback_category="reduced_rom",
        **kw,
    )


def test_valid_feedback_passes_unchanged() -> None:
    g = apply_guardrails(fb(), ctx())
    assert g.action is AgentAction.FEEDBACK and g.overrides == []
    assert g.feedback_text == FEEDBACK["reduced_rom"]


def test_close_paraphrase_allowed() -> None:
    text = "Try to lift your arm a bit higher, only as far as is comfortable."
    g = apply_guardrails(fb(text), ctx())
    assert g.feedback_text == text and g.overrides == []


@pytest.mark.parametrize(
    "text",
    [
        "You may have rotator cuff impingement, keep lifting.",
        "Push through the pain and lift higher!",
        "Your shoulder looks injured.",
    ],
)
def test_blocked_language_replaced(text: str) -> None:
    g = apply_guardrails(fb(text), ctx())
    assert g.feedback_text == FEEDBACK["reduced_rom"]
    assert g.overrides[0].startswith("feedback_text_replaced: blocked_language")


def test_free_text_not_template_replaced_and_length_limited() -> None:
    g = apply_guardrails(fb("Great job, you are a star athlete, amazing work today!"), ctx())
    assert "not_an_approved_template" in g.overrides[0]
    g = apply_guardrails(fb(FEEDBACK["reduced_rom"] + " " * 10 + "x" * 200), ctx())
    assert "too_long" in g.overrides[0]


def test_missing_text_filled_from_category_without_override() -> None:
    g = apply_guardrails(fb(None), ctx())
    assert g.feedback_text == FEEDBACK["reduced_rom"] and g.overrides == []


def test_invalid_action_uses_policy() -> None:
    g = apply_guardrails(FinalAnswer(action="DIAGNOSE", rationale="x"), ctx())
    assert g.overrides[0].startswith("invalid_action: 'DIAGNOSE'")
    assert g.action is AgentAction.FEEDBACK  # policy: confirmed deviation → feedback


def test_budget_exceeded_and_llm_error_use_policy() -> None:
    g = apply_guardrails(fb(), ctx(), budget_exceeded=True)
    assert g.overrides[0].startswith("tool_budget_exceeded")
    g = apply_guardrails(
        None, ctx(working="GOOD", r=rep(quality=RepQuality.GOOD, reasons=[])), llm_error="timeout"
    )
    assert g.overrides[0].startswith("llm_error: timeout") and g.action is AgentAction.CONTINUE


def test_forced_escalation_overrides_feedback() -> None:
    g = apply_guardrails(fb(), ctx(streak=2))  # this rep would be the 3rd confirmed
    assert g.action is AgentAction.HUMAN_REVIEW and g.forced_escalation
    assert g.overrides[0].startswith("forced_escalation: 3 confirmed deviations")
    assert g.evidence_refs["confirmed_streak"] == 3 and g.feedback_text is None


def test_no_forced_escalation_when_rom_ok_or_already_escalated() -> None:
    ok_rom = rep(rom=125.0, reasons=["trunk_lean: 14°"])  # 125/140 = 89%
    assert apply_guardrails(fb(), ctx(streak=2, r=ok_rom)).action is AgentAction.FEEDBACK
    done = AgentMemory(confirmed_streak=5, escalated=True)
    assert not apply_guardrails(fb(), ctx(memory=done)).forced_escalation


def test_unforced_escalation_needs_evidence() -> None:
    g = apply_guardrails(FinalAnswer(action="REQUEST_HUMAN_REVIEW", rationale="worried"), ctx())
    assert g.action is AgentAction.LOG_EVENT and "escalation_without_evidence" in g.overrides[0]
    g = apply_guardrails(
        FinalAnswer(
            action="REQUEST_HUMAN_REVIEW", rationale="ROM 100°", evidence_refs={"rep_index": 5}
        ),
        ctx(),
    )
    assert g.action is AgentAction.HUMAN_REVIEW and g.overrides == []


def test_low_tracking_confidence_forces_camera_adjustment() -> None:
    g = apply_guardrails(fb(), ctx(tracking=0.4, flags=["low_light"]))
    assert g.action is AgentAction.CAMERA_ADJUST
    assert g.feedback_text == CAMERA["low_light"] and g.overrides[0].startswith("poor_tracking")
    cam = ToolResult(
        "check_camera_setup",
        {},
        {"ok": True, "issues": ["too_far"], "instruction": CAMERA["too_far"]},
    )
    g = apply_guardrails(
        FinalAnswer(action="CONTINUE_MONITORING"), ctx(working="camera_issue", tools=[cam])
    )
    assert g.action is AgentAction.CAMERA_ADJUST and g.feedback_text == CAMERA["too_far"]


def test_no_feedback_on_uncertain_or_noise() -> None:
    for working in ("UNCERTAIN", "tracking_noise"):
        g = apply_guardrails(fb(), ctx(working=working))
        assert g.action is AgentAction.LOG_EVENT and g.feedback_text is None


def test_feedback_rate_limit() -> None:
    g = apply_guardrails(fb(), ctx(memory=AgentMemory(last_feedback_rep=4), r=rep(i=5)))
    assert g.action is AgentAction.LOG_EVENT and "feedback_rate_limited" in g.overrides[0]
    g = apply_guardrails(fb(), ctx(memory=AgentMemory(last_feedback_rep=3), r=rep(i=5)))
    assert g.action is AgentAction.FEEDBACK


def test_camera_text_must_be_template_and_rationale_sanitized() -> None:
    cam = ToolResult(
        "check_camera_setup",
        {},
        {"ok": True, "issues": ["low_light"], "instruction": CAMERA["low_light"]},
    )
    g = apply_guardrails(
        FinalAnswer(
            action="REQUEST_CAMERA_ADJUSTMENT",
            feedback_text="Buy a better camera!!",
            rationale="Your tendinitis makes this hard",
        ),
        ctx(working="camera_issue", tools=[cam]),
    )
    assert g.feedback_text == CAMERA["low_light"]
    assert any(o.startswith("camera_text_replaced") for o in g.overrides)
    assert any(o.startswith("rationale_sanitized") for o in g.overrides)
    assert "tendinitis" not in g.rationale


def test_non_feedback_actions_carry_no_text() -> None:
    g = apply_guardrails(
        FinalAnswer(action="CONTINUE_MONITORING", feedback_text="hi"),
        ctx(working="GOOD", r=rep(quality=RepQuality.GOOD, reasons=[])),
    )
    assert g.action is AgentAction.CONTINUE and g.feedback_text is None


def test_blocked_terms_word_boundaries() -> None:
    assert blocked_terms("Keep your elbow straight") == []
    assert blocked_terms("a small tear") == ["tear"]
    assert blocked_terms("retreat and treatment") == []  # no false hit on "treat"
