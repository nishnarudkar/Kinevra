"""Agent scenario suite (PROJECT.md §5.3): tests/agent_scenarios/*.yaml with ScriptedLLM.

Each step is one agent decision with scripted LLM turns and stubbed tool outputs. The suite
checks the final action AND which tools ran, and whether an OpenCV result changed the
working assessment.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from kinevra.agent.context import BufferFrameProvider, PoseHistory, ToolContext
from kinevra.agent.graph import KinevraAgent
from kinevra.agent.llm import ScriptedLLM
from kinevra.agent.state import LLMTurn
from kinevra.config import load_config
from kinevra.movement.session import MovementSession
from kinevra.schemas import AgentDecision, RepMetrics, SessionEvidence
from kinevra.storage.local import LocalStore
from kinevra.vision.buffer import FrameBuffer

SCENARIOS = sorted((Path(__file__).resolve().parents[1] / "agent_scenarios").glob("*.yaml"))

REP_DEFAULTS: dict[str, Any] = dict(
    side="right",
    min_angle=15.0,
    max_angle=150.0,
    rom=135.0,
    duration_s=2.4,
    mean_velocity_dps=100.0,
    peak_velocity_dps=180.0,
    smoothness=0.95,
    max_elbow_flexion=5.0,
    max_trunk_lean=2.0,
    confidence=0.9,
    rule_quality="GOOD",
    rule_reasons=[],
)
EVIDENCE_DEFAULTS: dict[str, Any] = dict(
    session_id="s1",
    mode="clip",
    exercise="shoulder_abduction",
    side="right",
    reps_target=None,
    baseline_rom=135.0,
    recent_roms=[135.0, 136.0, 134.0],
    rom_trend="stable",
    consecutive_deviations=0,
    movement_consistency=0.95,
    tracking_confidence=0.9,
    data_quality_flags=[],
)


def build_rep(spec: dict[str, Any]) -> RepMetrics:
    data = {**REP_DEFAULTS, **spec}
    i = data["rep_index"]
    data.setdefault("t_start", 3.0 * i)
    data.setdefault("t_peak", 3.0 * i + 1.2)
    data.setdefault("t_end", 3.0 * i + 2.4)
    data["max_angle"] = data["min_angle"] + data["rom"]
    return RepMetrics.model_validate(data)


def check(
    decision: AgentDecision, expect: dict[str, Any], agent: KinevraAgent, llm: ScriptedLLM
) -> None:
    ctx = agent.ctx
    assert decision.action.value == expect["action"], (
        decision.action,
        decision.rationale,
        decision.guardrail_overrides,
    )
    tools = [s.tool for s in decision.trace if s.node == "tool"]
    if "tools" in expect:
        assert tools == expect["tools"]
    if "llm_used" in expect:
        assert decision.llm_used is expect["llm_used"]
    if "changed_assessment" in expect:
        changed = any(s.changed_assessment for s in decision.trace)
        assert changed is expect["changed_assessment"]
    if "overrides" in expect:
        got = [o.split(":")[0].split(" ")[0] for o in decision.guardrail_overrides]
        assert got == expect["overrides"], decision.guardrail_overrides
    if "feedback_contains" in expect:
        assert expect["feedback_contains"] in (decision.feedback_text or "")
    if "feedback_equals" in expect:
        assert decision.feedback_text == expect["feedback_equals"]
    if "review_events" in expect:
        assert len(ctx.review_events) == expect["review_events"]
    if "review_snapshots" in expect:
        assert len(ctx.review_events[-1].snapshot_keys) == expect["review_snapshots"]
    if "confirmed_streak" in expect:
        assert ctx.memory.confirmed_streak == expect["confirmed_streak"]
    if "awaiting_camera_fix" in expect:
        assert ctx.memory.awaiting_camera_fix is expect["awaiting_camera_fix"]
    assert llm.remaining == 0, "not all scripted LLM turns were used"


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_scenario(path: Path, tmp_path: Path) -> None:
    scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = load_config(env={})
    session = MovementSession(cfg.exercise, "s1", "clip")
    ctx = ToolContext(
        cfg=cfg,
        session=session,
        frames=BufferFrameProvider(FrameBuffer(20.0)),
        poses=PoseHistory(),
        store=LocalStore(tmp_path),
    )
    llm = ScriptedLLM()
    agent = KinevraAgent(ctx, llm)
    assert len(scenario["steps"]) >= 1
    for step in scenario["steps"]:
        rep = build_rep(step["rep"]) if "rep" in step else None
        if rep is not None:
            session.state.reps.append(rep)
        evidence = SessionEvidence.model_validate(
            {
                **EVIDENCE_DEFAULTS,
                "reps_completed": len(session.state.reps),
                "latest_rep": rep,
                **step.get("evidence", {}),
            }
        )
        llm.push(LLMTurn.model_validate(t) for t in step.get("llm", []))
        agent.tool_stubs = step.get("tool_outputs", {})
        decision = agent.decide(step.get("trigger", "rep"), rep, evidence)
        check(decision, step["expect"], agent, llm)
        # every decision is complete and serialisable
        assert AgentDecision.model_validate_json(decision.model_dump_json()) == decision
        assert decision.trace[0].node == "observe" and decision.trace[-1].node == "act"


def test_all_nine_scenarios_present() -> None:
    assert len(SCENARIOS) == 9
