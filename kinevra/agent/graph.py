"""The Kinevra agent: a LangGraph perception → decision → action loop (PROJECT.md §3.2, §5.2).

    observe → triage ─(normal)──────────────────────────► decide → guard → act
                 └─(unusual)─► reason (LLM) ⇄ tools (≤ max_tool_calls) ─┘

- triage is deterministic: normal reps never call the LLM.
- tools re-measure the video with OpenCV 5; a result that changes the working assessment is
  marked `changed_assessment` in the trace — the evidence that vision output changed the plan.
- guard applies the guardrails; act performs the action and updates the agent's memory.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from kinevra.agent.context import ToolContext
from kinevra.agent.guardrails import GuardResult, apply_guardrails
from kinevra.agent.llm import LLMClient, LLMRequest
from kinevra.agent.policy import decide as policy_decide
from kinevra.agent.state import AgentState, DecisionContext, ToolResult, Trigger
from kinevra.agent.tools import execute_tool, tool_schemas
from kinevra.agent.trace import TraceRecorder
from kinevra.schemas import AgentAction, AgentDecision, RepMetrics, RepQuality, SessionEvidence

PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.is_file() else ""


def initial_assessment(trigger: Trigger, rep: RepMetrics | None, min_smoothness: float) -> str:
    if rep is None:
        return "quality_issue"
    if rep.rule_quality is RepQuality.GOOD and rep.smoothness < min_smoothness:
        return "irregular_trajectory"
    return rep.rule_quality.value


def triage_reasons(dc: DecisionContext) -> list[str]:
    """Deterministic triage: why (if at all) this moment needs reasoning and tools."""
    out: list[str] = []
    if dc.trigger == "quality":
        out.append("quality_check")
    if dc.memory.awaiting_camera_fix:
        out.append("camera_recheck")
    if dc.evidence.data_quality_flags:
        out.append("data_quality: " + ", ".join(dc.evidence.data_quality_flags))
    rep = dc.rep
    if rep is not None:
        if rep.rule_quality is not RepQuality.GOOD:
            out.append(f"rule_{rep.rule_quality.value.lower()}")
        lo, hi = dc.cfg.tools.reanalyze_trigger_confidence
        if lo <= rep.confidence <= hi:
            out.append(f"borderline_confidence ({rep.confidence:.2f})")
        if rep.smoothness < dc.cfg.agent.triage_min_smoothness:
            out.append(f"irregular_trajectory (smoothness {rep.smoothness:.2f})")
    if dc.evidence.rom_trend == "decreasing":
        out.append("rom_trend_decreasing")
    return out


class KinevraAgent:
    def __init__(
        self,
        ctx: ToolContext,
        llm: LLMClient,
        *,
        tool_stubs: dict[str, dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.ctx = ctx
        self.llm = llm
        self.tool_stubs = tool_stubs or {}
        self.system_prompt = system_prompt if system_prompt is not None else load_system_prompt()
        self.decisions: list[AgentDecision] = []
        self.graph = self._build()

    # --- public API -----------------------------------------------------------------------

    def decide(
        self, trigger: Trigger, rep: RepMetrics | None, evidence: SessionEvidence
    ) -> AgentDecision:
        ex = self.ctx.cfg.exercise
        dc = DecisionContext(
            trigger=trigger,
            evidence=evidence,
            rep=rep,
            triage_reasons=[],
            working_assessment=initial_assessment(trigger, rep, ex.agent.triage_min_smoothness),
            memory=self.ctx.memory,
            cfg=ex,
            tool_budget_left=ex.agent.max_tool_calls,
        )
        state: AgentState = {
            "ctx": dc,
            "recorder": TraceRecorder(),
            "messages": [],
            "pending_calls": [],
            "tool_calls_made": 0,
            "proposal": None,
            "budget_exceeded": False,
            "llm_used": False,
            "llm_error": None,
            "started": time.perf_counter(),
        }
        final = self.graph.invoke(state, {"recursion_limit": 4 * ex.agent.max_tool_calls + 20})
        decision: AgentDecision = final["decision"]
        self.decisions.append(decision)
        return decision

    # --- graph ----------------------------------------------------------------------------

    def _build(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("observe", self._observe)
        g.add_node("triage", self._triage)
        g.add_node("reason", self._reason)
        g.add_node("tools", self._tools)
        g.add_node("decide", self._decide)
        g.add_node("guard", self._guard)
        g.add_node("act", self._act)
        g.add_edge(START, "observe")
        g.add_edge("observe", "triage")
        g.add_conditional_edges(
            "triage",
            lambda state: "decide" if state["normal"] else "reason",
            {"decide": "decide", "reason": "reason"},
        )
        g.add_conditional_edges(
            "reason",
            lambda state: "tools" if state["pending_calls"] else "decide",
            {"tools": "tools", "decide": "decide"},
        )
        g.add_edge("tools", "reason")
        g.add_edge("decide", "guard")
        g.add_edge("guard", "act")
        g.add_edge("act", END)
        return g.compile()

    def _observe(self, state: AgentState) -> dict[str, Any]:
        dc = state["ctx"]
        rep, ev = dc.rep, dc.evidence
        state["recorder"].add(
            "observe",
            output={
                "trigger": dc.trigger,
                "rep_index": rep.rep_index if rep else None,
                "rule_quality": rep.rule_quality.value if rep else None,
                "rom": rep.rom if rep else None,
                "confidence": rep.confidence if rep else None,
                "tracking_confidence": ev.tracking_confidence,
                "baseline_rom": ev.baseline_rom,
                "rom_trend": ev.rom_trend,
                "working_assessment": dc.working_assessment,
            },
        )
        return {}

    def _triage(self, state: AgentState) -> dict[str, Any]:
        dc = state["ctx"]
        dc.triage_reasons = triage_reasons(dc)
        normal = not dc.triage_reasons
        state["recorder"].add(
            "triage",
            output={"route": "decide" if normal else "reason", "reasons": dc.triage_reasons},
        )
        messages = [] if normal else [self._facts_message(dc)]
        return {"normal": normal, "messages": messages}

    def _facts_message(self, dc: DecisionContext) -> dict[str, Any]:
        facts = {
            "trigger": dc.trigger,
            "triage_reasons": dc.triage_reasons,
            "working_assessment": dc.working_assessment,
            "latest_rep": dc.rep.model_dump(mode="json") if dc.rep else None,
            "evidence": dc.evidence.model_dump(mode="json", exclude={"latest_rep"}),
            "memory": asdict(dc.memory),
            "tool_budget": dc.tool_budget_left,
        }
        return {"role": "user", "content": json.dumps(facts)}

    def _reason(self, state: AgentState) -> dict[str, Any]:
        dc, rec = state["ctx"], state["recorder"]
        max_calls = dc.cfg.agent.max_tool_calls
        dc.tool_budget_left = max_calls - state["tool_calls_made"]
        request = LLMRequest(self.system_prompt, state["messages"], tool_schemas(), dc)
        start = time.perf_counter()
        try:
            turn = self.llm.next_turn(request)
        except AssertionError:
            raise  # test doubles signal misuse loudly
        except Exception as exc:  # network, timeout, parsing (Phase 6)
            rec.add(
                "reason",
                output={"error": str(exc)[:200]},
                latency_ms=(time.perf_counter() - start) * 1000,
            )
            return {"llm_used": True, "llm_error": type(exc).__name__, "pending_calls": []}
        ms = (time.perf_counter() - start) * 1000
        calls = turn.tool_calls
        rec.add(
            "reason",
            output={
                "llm": getattr(self.llm, "name", "llm"),
                "tool_calls": [c.name for c in calls],
                "final_action": turn.final.action if turn.final else None,
            },
            latency_ms=ms,
        )
        messages = [*state["messages"], {"role": "assistant", "content": turn.model_dump_json()}]
        if calls and state["tool_calls_made"] + len(calls) > max_calls:
            return {
                "llm_used": True,
                "budget_exceeded": True,
                "pending_calls": [],
                "messages": messages,
            }
        return {
            "llm_used": True,
            "pending_calls": list(calls),
            "proposal": turn.final,
            "messages": messages,
        }

    def _tools(self, state: AgentState) -> dict[str, Any]:
        dc, rec = state["ctx"], state["recorder"]
        messages = list(state["messages"])
        for call in state["pending_calls"]:
            out, ms = execute_tool(self.ctx, call.name, call.args, self.tool_stubs)
            new = out.get("assessment")
            changed = new is not None and new != dc.working_assessment
            if changed:
                dc.working_assessment = str(new)
            dc.tool_results.append(ToolResult(call.name, call.args, out))
            rec.add(
                "tool",
                tool=call.name,
                tool_input=call.args,
                output=out,
                changed_assessment=changed,
                latency_ms=ms,
            )
            messages.append(
                {"role": "tool", "name": call.name, "content": json.dumps(out, default=str)}
            )
        return {
            "pending_calls": [],
            "messages": messages,
            "tool_calls_made": state["tool_calls_made"] + len(state["pending_calls"]),
        }

    def _decide(self, state: AgentState) -> dict[str, Any]:
        dc = state["ctx"]
        proposal = policy_decide(dc) if state["normal"] else state.get("proposal")
        state["recorder"].add(
            "decide",
            output={
                "source": "triage" if state["normal"] else "llm",
                "proposal": proposal.action if proposal else None,
                "working_assessment": dc.working_assessment,
            },
        )
        return {"proposal": proposal}

    def _guard(self, state: AgentState) -> dict[str, Any]:
        dc = state["ctx"]
        g = apply_guardrails(
            state.get("proposal"),
            dc,
            budget_exceeded=state["budget_exceeded"],
            llm_error=state.get("llm_error"),
        )
        state["recorder"].add("guard", output={"action": g.action.value, "overrides": g.overrides})
        return {"guard": g}

    def _act(self, state: AgentState) -> dict[str, Any]:
        dc, rec = state["ctx"], state["recorder"]
        g: GuardResult = state["guard"]
        mem, rep = self.ctx.memory, dc.rep

        if g.action is AgentAction.HUMAN_REVIEW:
            self._ensure_review(dc, g, rec)
            mem.escalated = True
        if g.action is AgentAction.FEEDBACK and rep is not None:
            mem.last_feedback_rep = rep.rep_index
        if dc.working_assessment == "camera_ok":
            mem.awaiting_camera_fix = False
        if g.action is AgentAction.CAMERA_ADJUST:
            mem.awaiting_camera_fix = True
            mem.last_camera_instruction = g.feedback_text
        if g.action is AgentAction.LOG_EVENT:
            self.ctx.logged_events.append(
                {
                    "type": "agent_log",
                    "rep_index": rep.rep_index if rep else None,
                    "rationale": g.rationale,
                    "overrides": g.overrides,
                }
            )
        if rep is not None:
            if dc.working_assessment == "DEVIATION":
                mem.confirmed_streak += 1
            elif dc.working_assessment in ("GOOD", "tracking_noise"):
                mem.confirmed_streak = 0
                mem.escalated = False

        total_ms = (time.perf_counter() - state["started"]) * 1000
        rec.add(
            "act",
            output={
                "action": g.action.value,
                "feedback_text": g.feedback_text,
                "confirmed_streak": mem.confirmed_streak,
            },
        )
        decision = AgentDecision(
            session_id=self.ctx.session_id,
            rep_index=rep.rep_index if rep else None,
            action=g.action,
            rationale=g.rationale,
            feedback_text=g.feedback_text,
            evidence_refs=g.evidence_refs,
            trace=list(rec.steps),
            guardrail_overrides=g.overrides,
            llm_used=state["llm_used"],
            total_latency_ms=round(total_ms),
        )
        return {"decision": decision}

    def _ensure_review(self, dc: DecisionContext, g: GuardResult, rec: TraceRecorder) -> None:
        """Every escalation carries annotated evidence and a review event (PROJECT.md §12)."""
        rep = dc.rep
        keys: list[str] = list((dc.last("render_evidence_snapshot") or {}).get("snapshot_keys", []))
        if not keys and rep is not None:
            keys = self.ctx.snapshots.get(rep.rep_index, [])
        if not keys and rep is not None:
            out, ms = execute_tool(
                self.ctx, "render_evidence_snapshot", {"rep_index": rep.rep_index}, self.tool_stubs
            )
            rec.add(
                "act",
                tool="render_evidence_snapshot",
                tool_input={"rep_index": rep.rep_index},
                output=out,
                latency_ms=ms,
            )
            keys = list(out.get("snapshot_keys", []))
        already = any(r.name == "request_human_review" for r in dc.tool_results)
        if not already:
            args = {"reason": g.rationale, "evidence": g.evidence_refs, "snapshot_keys": keys}
            out, ms = execute_tool(self.ctx, "request_human_review", args)
            rec.add("act", tool="request_human_review", tool_input=args, output=out, latency_ms=ms)
