"""PolicyLLM decisions, and the agent end to end with real tools (no stubs)."""

from pathlib import Path

import pytest

from kinevra.agent.context import AgentMemory
from kinevra.agent.graph import KinevraAgent, triage_reasons
from kinevra.agent.llm import LLMRequest, PolicyLLM
from kinevra.agent.policy import policy_turn
from kinevra.agent.state import DecisionContext, LLMTurn, ToolResult, Trigger
from kinevra.config import load_config
from kinevra.schemas import AgentAction, RepQuality, SessionEvidence
from tests.conftest import make_pose
from tests.unit.test_agent_tools import (
    FPS,
    H,
    RoiOnlyEstimator,
    W,
    _blank_frames,
    _ctx,
    _rep,
    _truth_angles,
)

CFG = load_config(env={}).exercise


def evidence(**kw: object) -> SessionEvidence:
    base: dict[str, object] = dict(
        session_id="s",
        mode="clip",
        exercise="shoulder_abduction",
        side="right",
        reps_completed=4,
        reps_target=None,
        latest_rep=None,
        baseline_rom=135.0,
        recent_roms=[135.0],
        rom_trend="stable",
        consecutive_deviations=0,
        movement_consistency=0.9,
        tracking_confidence=0.9,
        data_quality_flags=[],
    )
    base.update(kw)
    return SessionEvidence.model_validate(base)


def dc(
    rep_kw: dict[str, object] | None = None,
    *,
    trigger: Trigger = "rep",
    working: str | None = None,
    results: list[ToolResult] | None = None,
    memory: AgentMemory | None = None,
    budget: int = 4,
    **ev: object,
) -> DecisionContext:
    rep = _rep(**rep_kw) if rep_kw is not None else None
    return DecisionContext(
        trigger=trigger,
        evidence=evidence(**ev),
        rep=rep,
        triage_reasons=[],
        working_assessment=working or (rep.rule_quality.value if rep else "quality_issue"),
        memory=memory or AgentMemory(),
        cfg=CFG,
        tool_results=results or [],
        tool_budget_left=budget,
    )


def test_uncertain_rep_reanalysed_then_continue() -> None:
    c = dc({})
    assert policy_turn(c).tool_calls[0].name == "reanalyze_segment_roi"
    c.tool_results.append(
        ToolResult("reanalyze_segment_roi", {}, {"ok": True, "rom_after": 132, "reasons_after": []})
    )
    c.working_assessment = "GOOD"
    final = policy_turn(c).final
    assert final is not None and final.action == AgentAction.CONTINUE.value


def test_confirmed_deviation_feedback_category() -> None:
    c = dc(
        {
            "rule_quality": RepQuality.DEVIATION,
            "confidence": 0.9,
            "rule_reasons": ["trunk_lean: 14°"],
        }
    )
    final = policy_turn(c).final  # confident deviation: no re-analysis needed
    assert final is not None and final.action == AgentAction.FEEDBACK.value
    assert final.feedback_category == "trunk_lean"


def test_quality_trigger_checks_camera_then_asks_for_adjustment() -> None:
    c = dc(None, trigger="quality", tracking_confidence=0.4, data_quality_flags=["low_light"])
    assert policy_turn(c).tool_calls[0].name == "check_camera_setup"
    c.tool_results.append(
        ToolResult(
            "check_camera_setup",
            {},
            {
                "ok": True,
                "issues": ["low_light"],
                "instruction": "Please turn on more light or face a light source.",
            },
        )
    )
    c.working_assessment = "camera_issue"
    final = policy_turn(c).final
    assert final is not None and final.action == AgentAction.CAMERA_ADJUST.value
    assert "light" in (final.feedback_text or "")


def test_escalation_renders_snapshot_first() -> None:
    c = dc(
        {
            "rule_quality": RepQuality.DEVIATION,
            "confidence": 0.9,
            "rom": 95,
            "rule_reasons": ["reduced_rom: 95°"],
        },
        memory=AgentMemory(confirmed_streak=2),
    )
    assert policy_turn(c).tool_calls[0].name == "render_evidence_snapshot"
    c.tool_results.append(ToolResult("render_evidence_snapshot", {}, {"ok": True}))
    final = policy_turn(c).final
    assert final is not None and final.action == AgentAction.HUMAN_REVIEW.value
    assert final.evidence_refs["confirmed_streak"] == 3


def test_irregular_trajectory_uses_flow_and_budget_zero_decides() -> None:
    c = dc(
        {"rule_quality": RepQuality.GOOD, "confidence": 0.9, "smoothness": 0.6, "rule_reasons": []},
        working="irregular_trajectory",
    )
    assert policy_turn(c).tool_calls[0].name == "verify_motion_optical_flow"
    c.tool_budget_left = 0
    assert policy_turn(c).final is not None


def test_triage_reasons() -> None:
    good = dc({"rule_quality": RepQuality.GOOD, "confidence": 0.9, "rule_reasons": []})
    assert triage_reasons(good) == []
    assert triage_reasons(
        dc({"confidence": 0.7, "rule_quality": RepQuality.GOOD, "rule_reasons": []})
    )[0].startswith("borderline_confidence")
    assert "rom_trend_decreasing" in triage_reasons(
        dc(
            {"rule_quality": RepQuality.GOOD, "confidence": 0.9, "rule_reasons": []},
            rom_trend="decreasing",
        )
    )


def test_agent_end_to_end_real_roi_tool_changes_decision(tmp_path: Path) -> None:
    """No stubs: PolicyLLM calls the real OpenCV re-analysis tool, which flips UNCERTAIN → GOOD."""
    truth = _truth_angles(66, 150.0)
    poses = [
        make_pose(a, t=i / FPS, frame_idx=i, width=W, height=H, visibility=0.55)
        for i, a in enumerate(truth)
    ]
    ctx = _ctx(tmp_path, _blank_frames(66), poses, _rep(), estimator=RoiOnlyEstimator(truth))
    agent = KinevraAgent(ctx, PolicyLLM())
    decision = agent.decide("rep", ctx.session.state.reps[0], evidence())
    assert decision.action is AgentAction.CONTINUE and decision.llm_used
    tool_steps = [s for s in decision.trace if s.node == "tool"]
    assert [s.tool for s in tool_steps] == ["reanalyze_segment_roi"]
    assert tool_steps[0].changed_assessment
    assert tool_steps[0].tool_output_summary is not None
    assert tool_steps[0].tool_output_summary["quality_after"] == "GOOD"
    assert decision.guardrail_overrides == []


def test_agent_end_to_end_confirmed_deviation_escalates_with_real_snapshots(
    tmp_path: Path,
) -> None:
    truth = _truth_angles(66, 100.0)  # ROM 85° < 80% of baseline 135°
    poses = [
        make_pose(a, t=i / FPS, frame_idx=i, width=W, height=H, visibility=0.55)
        for i, a in enumerate(truth)
    ]
    ctx = _ctx(tmp_path, _blank_frames(66), poses, _rep(), estimator=RoiOnlyEstimator(truth))
    ctx.memory.confirmed_streak = 2
    decision = KinevraAgent(ctx, PolicyLLM()).decide("rep", ctx.session.state.reps[0], evidence())
    assert decision.action is AgentAction.HUMAN_REVIEW
    assert [s.tool for s in decision.trace if s.node == "tool"] == [
        "reanalyze_segment_roi",
        "render_evidence_snapshot",
    ]
    review = ctx.review_events[0]
    assert len(review.snapshot_keys) == 3
    assert all((tmp_path / k).is_file() for k in review.snapshot_keys)
    assert ctx.memory.confirmed_streak == 3 and ctx.memory.escalated


@pytest.mark.parametrize("bad", [ValueError("boom"), TimeoutError("slow")])
def test_llm_failure_falls_back_to_policy(tmp_path: Path, bad: Exception) -> None:
    class Broken:
        name = "broken"

        def next_turn(self, request: LLMRequest) -> LLMTurn:
            raise bad

    ctx = _ctx(
        tmp_path,
        [],
        [],
        _rep(rule_quality=RepQuality.DEVIATION, confidence=0.9, rule_reasons=["trunk_lean: 14°"]),
    )
    decision = KinevraAgent(ctx, Broken()).decide("rep", ctx.session.state.reps[0], evidence())
    assert decision.action is AgentAction.FEEDBACK
    assert decision.guardrail_overrides[0].startswith(f"llm_error: {type(bad).__name__}")
