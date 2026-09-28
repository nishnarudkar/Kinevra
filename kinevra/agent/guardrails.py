"""Guardrails between the agent's proposal and what the system does (PROJECT.md §5.2, §12).

Applied in order; every change is recorded as an override string:
1. no usable proposal (LLM error, tool budget exceeded, invalid action) → deterministic policy
2. deterministic escalation rule forces REQUEST_HUMAN_REVIEW (cannot be suppressed)
3. an unforced escalation needs evidence refs, otherwise LOG_EVENT
4. bad data (low tracking confidence / camera issue) → REQUEST_CAMERA_ADJUSTMENT instead of
   feedback or silent monitoring; no movement feedback on uncertain or noise-dismissed reps
5. feedback text must be an approved template (or a close paraphrase), short, and free of
   diagnostic / "push through pain" language; otherwise it is replaced
6. feedback is rate-limited (at most once every N reps)
7. camera instructions must come from the camera templates
8. the rationale must be free of blocked language
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from kinevra.agent.policy import decide
from kinevra.agent.state import DecisionContext, FinalAnswer
from kinevra.agent.templates import CAMERA, FEEDBACK, camera_instruction, feedback_category_for
from kinevra.schemas import AgentAction

MAX_FEEDBACK_CHARS = 160
MAX_RATIONALE_CHARS = 400
TEMPLATE_SIMILARITY = 0.75

BLOCKED = re.compile(
    r"\b(diagnos\w*|injur\w*|tear|torn|tendin\w*|bursitis|impinge\w*|arthriti\w*|"
    r"frozen shoulder|rotator cuff|dislocat\w*|fractur\w*|sprain\w*|strain\w*|"
    r"disease|disorder|patholog\w*|syndrome|inflam\w*|lesion|you have|you suffer|"
    r"push through|through the pain|ignore the pain|no pain,? no gain|work through the pain)\b",
    re.IGNORECASE,
)
SAFE_RATIONALE = "Decision based on the measured movement evidence."


@dataclass
class GuardResult:
    action: AgentAction
    rationale: str
    feedback_text: str | None
    evidence_refs: dict[str, Any]
    overrides: list[str] = field(default_factory=list)
    forced_escalation: bool = False


def blocked_terms(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in BLOCKED.finditer(text)})


def closest_template(text: str, templates: dict[str, str]) -> tuple[str | None, float]:
    best, score = None, 0.0
    for key, tpl in templates.items():
        s = difflib.SequenceMatcher(None, text.lower().strip(), tpl.lower()).ratio()
        if s > score:
            best, score = key, s
    return best, score


def _parse_action(value: str) -> AgentAction | None:
    try:
        return AgentAction(value)
    except ValueError:
        return None


def apply_guardrails(
    proposal: FinalAnswer | None,
    ctx: DecisionContext,
    *,
    budget_exceeded: bool = False,
    llm_error: str | None = None,
) -> GuardResult:
    overrides: list[str] = []
    rep = ctx.rep

    # 1. usable proposal?
    if llm_error:
        overrides.append(f"llm_error: {llm_error} → deterministic policy")
        proposal = None
    if budget_exceeded:
        overrides.append(
            f"tool_budget_exceeded: > {ctx.cfg.agent.max_tool_calls} tool calls → "
            "deterministic policy"
        )
        proposal = None
    action = _parse_action(proposal.action) if proposal else None
    if proposal is not None and action is None:
        overrides.append(f"invalid_action: {proposal.action!r} → deterministic policy")
        proposal = None
    if proposal is None:
        proposal = decide(ctx)
        action = AgentAction(proposal.action)
        if not overrides:
            overrides.append("no_proposal → deterministic policy")
    assert action is not None
    rationale = proposal.rationale or SAFE_RATIONALE
    text = proposal.feedback_text
    refs = dict(proposal.evidence_refs)

    # 2. deterministic escalation (cannot be suppressed)
    forced = ctx.escalation_due()
    if forced and action is not AgentAction.HUMAN_REVIEW:
        esc = ctx.cfg.agent.escalation
        overrides.append(
            f"forced_escalation: {ctx.memory.confirmed_streak + 1} confirmed deviations, "
            f"ROM < {esc.rom_ratio_below:.0%} of baseline → {AgentAction.HUMAN_REVIEW.value}"
        )
        action, text = AgentAction.HUMAN_REVIEW, None
        rationale = (
            f"{ctx.memory.confirmed_streak + 1} consecutive confirmed deviations with ROM "
            f"{ctx.current_rom():.1f}° below {esc.rom_ratio_below:.0%} of baseline "
            f"{ctx.evidence.baseline_rom:.1f}°."
        )
        refs.update(
            confirmed_streak=ctx.memory.confirmed_streak + 1,
            rom=ctx.current_rom(),
            baseline_rom=ctx.evidence.baseline_rom,
        )
        if rep is not None:
            refs.setdefault("rep_index", rep.rep_index)

    # 3. unforced escalation needs evidence
    if action is AgentAction.HUMAN_REVIEW and not forced and not refs:
        overrides.append("escalation_without_evidence → LOG_EVENT")
        action = AgentAction.LOG_EVENT

    # 4. bad data: never movement feedback on it
    min_conf = ctx.cfg.rules.min_confidence
    bad_video = ctx.working_assessment == "camera_issue" or (
        ctx.evidence.tracking_confidence < min_conf and ctx.working_assessment != "camera_ok"
    )
    if bad_video and action in (AgentAction.FEEDBACK, AgentAction.CONTINUE):
        overrides.append(
            f"poor_tracking (confidence {ctx.evidence.tracking_confidence:.2f} < {min_conf:g} "
            f"or camera issue): {action.value} → {AgentAction.CAMERA_ADJUST.value}"
        )
        action = AgentAction.CAMERA_ADJUST
        cam = ctx.last("check_camera_setup") or {}
        text = cam.get("instruction") or camera_instruction(list(ctx.evidence.data_quality_flags))
    if action is AgentAction.FEEDBACK and ctx.working_assessment in ("UNCERTAIN", "tracking_noise"):
        overrides.append(f"feedback_on_{ctx.working_assessment.lower()}_data → LOG_EVENT")
        action, text = AgentAction.LOG_EVENT, None

    # 5. feedback text
    if action is AgentAction.FEEDBACK:
        category = proposal.feedback_category
        if category not in FEEDBACK:
            category = feedback_category_for(ctx.current_reasons()) or "encouragement"
        replace_reason = None
        if not text:
            replace_reason = "missing"
        elif blocked_terms(text):
            replace_reason = f"blocked_language {blocked_terms(text)}"
        elif len(text) > MAX_FEEDBACK_CHARS:
            replace_reason = f"too_long ({len(text)} chars)"
        else:
            _, score = closest_template(text, FEEDBACK)
            if score < TEMPLATE_SIMILARITY:
                replace_reason = f"not_an_approved_template (similarity {score:.2f})"
        if replace_reason:
            if text:  # a missing text is just filled in, not an override
                overrides.append(f"feedback_text_replaced: {replace_reason}")
            text = FEEDBACK[category]

        # 6. rate limit
        last, gap = ctx.memory.last_feedback_rep, ctx.cfg.agent.feedback_min_reps_between
        if rep is not None and last is not None and rep.rep_index - last < gap:
            overrides.append(
                f"feedback_rate_limited: last feedback on rep {last}, now rep {rep.rep_index} "
                f"(min {gap} reps apart) → LOG_EVENT"
            )
            action, text = AgentAction.LOG_EVENT, None

    # 7. camera instruction text
    if action is AgentAction.CAMERA_ADJUST:
        cam = ctx.last("check_camera_setup") or {}
        allowed = cam.get("instruction") or CAMERA["generic"]
        if (
            not text
            or blocked_terms(text)
            or closest_template(text, CAMERA)[1] < TEMPLATE_SIMILARITY
        ):
            if text and text != allowed:
                overrides.append("camera_text_replaced: not an approved instruction")
            text = allowed

    if action not in (AgentAction.FEEDBACK, AgentAction.CAMERA_ADJUST):
        text = None

    # 8. rationale
    if blocked_terms(rationale):
        overrides.append(f"rationale_sanitized: {blocked_terms(rationale)}")
        rationale = SAFE_RATIONALE
    rationale = rationale[:MAX_RATIONALE_CHARS]

    return GuardResult(action, rationale, text, refs, overrides, forced)
