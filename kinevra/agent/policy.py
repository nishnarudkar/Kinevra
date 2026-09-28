"""Deterministic agent policy: which OpenCV tool to call next, and what to decide.

Used by PolicyLLM (local replay without AWS; Bedrock fallback in Phase 6) and by the
guardrails when the LLM gives no usable proposal. Mirrors what the system prompt asks the LLM
to do, so the two can be compared in evaluation.
"""

from __future__ import annotations

from kinevra.agent.state import DecisionContext, FinalAnswer, LLMTurn, ToolCall
from kinevra.agent.templates import CAMERA, FEEDBACK, feedback_category_for
from kinevra.schemas import AgentAction, RepQuality


def needs_camera_check(c: DecisionContext) -> bool:
    if c.trigger == "quality" or c.memory.awaiting_camera_fix or c.evidence.data_quality_flags:
        return True
    return bool(c.rep and any(r.startswith("quality_flag") for r in c.rep.rule_reasons))


def needs_reanalysis(c: DecisionContext) -> bool:
    if c.rep is None:
        return False
    lo, hi = c.cfg.tools.reanalyze_trigger_confidence
    if c.rep.rule_quality is RepQuality.UNCERTAIN:
        return True
    # borderline confidence: a deviation (or a GOOD verdict) could be tracking noise
    borderline = c.rep.confidence <= hi
    return borderline and (c.rep.rule_quality is RepQuality.DEVIATION or c.rep.confidence >= lo)


def needs_flow_check(c: DecisionContext) -> bool:
    return bool(c.rep and c.rep.smoothness < c.cfg.agent.triage_min_smoothness)


def next_tool(c: DecisionContext) -> ToolCall | None:
    if c.tool_budget_left <= 0:
        return None
    done = {r.name for r in c.tool_results}
    idx = c.rep.rep_index if c.rep else None
    if needs_camera_check(c) and "check_camera_setup" not in done:
        return ToolCall(name="check_camera_setup", args={"window_s": 3.0})
    if c.working_assessment == "camera_issue":
        return None  # no point re-measuring movement on bad video
    if idx is not None and needs_reanalysis(c) and "reanalyze_segment_roi" not in done:
        return ToolCall(name="reanalyze_segment_roi", args={"rep_index": idx})
    if (
        idx is not None
        and needs_flow_check(c)
        and c.working_assessment not in ("tracking_noise",)
        and "verify_motion_optical_flow" not in done
    ):
        return ToolCall(name="verify_motion_optical_flow", args={"rep_index": idx})
    if idx is not None and c.escalation_due() and "render_evidence_snapshot" not in done:
        return ToolCall(name="render_evidence_snapshot", args={"rep_index": idx})
    return None


def decide(c: DecisionContext) -> FinalAnswer:
    """Final proposal from the facts gathered so far (no tool calls)."""
    rep, ev, wa = c.rep, c.evidence, c.working_assessment
    refs: dict[str, object] = {}
    if rep is not None:
        refs = {
            "rep_index": rep.rep_index,
            "rom": round(c.current_rom() or rep.rom, 1),
            "confidence": round(rep.confidence, 2),
        }
    if ev.baseline_rom is not None:
        refs["baseline_rom"] = round(ev.baseline_rom, 1)

    if wa == "camera_issue":
        cam = c.last("check_camera_setup") or {}
        return FinalAnswer(
            action=AgentAction.CAMERA_ADJUST.value,
            rationale=f"Camera check found {', '.join(cam.get('issues', [])) or 'poor video'}; "
            "movement feedback paused until the view improves.",
            feedback_text=cam.get("instruction") or CAMERA["generic"],
            evidence_refs={**refs, "issues": cam.get("issues", [])},
        )
    if wa in ("camera_ok",) and rep is None:
        return FinalAnswer(
            action=AgentAction.CONTINUE.value,
            rationale="Camera check passed; monitoring resumes.",
            evidence_refs={"camera": "ok"},
        )
    if wa == "tracking_noise":
        flow = c.last("verify_motion_optical_flow") or {}
        return FinalAnswer(
            action=AgentAction.CONTINUE.value,
            rationale=f"Optical flow shows the irregularity is landmark jitter "
            f"(jitter ratio {flow.get('jitter_ratio', '?')}), not a movement deviation.",
            evidence_refs={**refs, "jitter_ratio": flow.get("jitter_ratio")},
        )
    if wa == "DEVIATION" and rep is not None:
        if c.escalation_due():
            streak = c.memory.confirmed_streak + 1
            return FinalAnswer(
                action=AgentAction.HUMAN_REVIEW.value,
                rationale=f"{streak} consecutive confirmed deviations; ROM "
                f"{refs.get('rom')}° vs baseline {refs.get('baseline_rom')}°.",
                evidence_refs={**refs, "confirmed_streak": streak},
            )
        category = feedback_category_for(c.current_reasons()) or "encouragement"
        return FinalAnswer(
            action=AgentAction.FEEDBACK.value,
            rationale=f"Deviation confirmed: {', '.join(c.current_reasons()) or 'rule flag'}.",
            feedback_text=FEEDBACK[category],
            feedback_category=category,
            evidence_refs=refs,
        )
    if wa == "UNCERTAIN":
        return FinalAnswer(
            action=AgentAction.LOG_EVENT.value,
            rationale="Evidence is still uncertain after re-checking; no feedback on weak data.",
            evidence_refs=refs,
        )
    rom_text = f" (ROM {refs['rom']}°)" if "rom" in refs else ""
    return FinalAnswer(
        action=AgentAction.CONTINUE.value,
        rationale=f"Movement within expected range{rom_text}.",
        evidence_refs=refs,
    )


def policy_turn(c: DecisionContext) -> LLMTurn:
    call = next_tool(c)
    return LLMTurn(tool_calls=[call]) if call else LLMTurn(final=decide(c))
