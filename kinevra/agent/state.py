"""Agent state: the LangGraph state dict and the structured view given to LLM clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from kinevra.agent.context import AgentMemory
from kinevra.config import ExerciseConfig
from kinevra.schemas import AgentDecision, RepMetrics, SessionEvidence

Trigger = Literal["rep", "quality"]


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class FinalAnswer(BaseModel):
    """What the LLM (or policy) proposes; guardrails turn it into an AgentDecision."""

    action: str
    rationale: str = ""
    feedback_text: str | None = None
    feedback_category: str | None = None
    evidence_refs: dict[str, Any] = Field(default_factory=dict)


class LLMTurn(BaseModel):
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final: FinalAnswer | None = None


@dataclass
class ToolResult:
    name: str
    args: dict[str, Any]
    output: dict[str, Any]


@dataclass
class DecisionContext:
    """Structured facts for one decision (all numbers come from OpenCV / deterministic code)."""

    trigger: Trigger
    evidence: SessionEvidence
    rep: RepMetrics | None
    triage_reasons: list[str]
    working_assessment: str
    memory: AgentMemory
    cfg: ExerciseConfig
    tool_results: list[ToolResult] = field(default_factory=list)
    tool_budget_left: int = 0

    def last(self, tool: str) -> dict[str, Any] | None:
        for r in reversed(self.tool_results):
            if r.name == tool and r.output.get("ok"):
                return r.output
        return None

    def current_rom(self) -> float | None:
        re = self.last("reanalyze_segment_roi")
        if re is not None and re.get("rom_after") is not None:
            return float(re["rom_after"])
        return self.rep.rom if self.rep else None

    def current_reasons(self) -> list[str]:
        re = self.last("reanalyze_segment_roi")
        if re is not None and re.get("reasons_after") is not None:
            return list(re["reasons_after"])
        return list(self.rep.rule_reasons) if self.rep else []

    def escalation_due(self) -> bool:
        """Deterministic escalation rule (PROJECT.md §5.2): N confirmed deviations in a row with
        ROM below the ratio of baseline. Uses the streak INCLUDING this rep if it is confirmed."""
        if self.working_assessment != "DEVIATION" or self.memory.escalated:
            return False
        esc = self.cfg.agent.escalation
        if self.memory.confirmed_streak + 1 < esc.consecutive_confirmed_deviations:
            return False
        rom, baseline = self.current_rom(), self.evidence.baseline_rom
        return rom is not None and baseline is not None and rom < esc.rom_ratio_below * baseline


class AgentState(TypedDict, total=False):
    """LangGraph state for one decision (objects are mutated in place by the nodes)."""

    ctx: DecisionContext
    recorder: Any  # TraceRecorder (Any avoids a circular import)
    normal: bool  # triage verdict: normal rep → no LLM
    messages: list[dict[str, Any]]
    pending_calls: list[ToolCall]
    tool_calls_made: int
    proposal: FinalAnswer | None
    budget_exceeded: bool
    llm_used: bool
    llm_error: str | None
    guard: Any  # GuardResult
    decision: AgentDecision
    started: float
