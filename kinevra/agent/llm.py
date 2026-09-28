"""LLM clients behind one interface. Bedrock (Phase 6) implements the same protocol.

A client sees the chat `messages` (for a real model) and a structured `DecisionContext`
(for deterministic clients). It returns either tool calls or a final proposal. It never
computes measurements: every number it can cite comes from the tools.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from kinevra.agent.policy import policy_turn
from kinevra.agent.state import DecisionContext, LLMTurn


@dataclass
class LLMRequest:
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    context: DecisionContext


class LLMClient(Protocol):
    name: str

    def next_turn(self, request: LLMRequest) -> LLMTurn: ...


class ScriptedLLM:
    """Test double: returns pre-scripted turns in order; fails loudly if called unexpectedly."""

    name = "scripted"

    def __init__(self, turns: Iterable[LLMTurn] = ()) -> None:
        self._turns: deque[LLMTurn] = deque(turns)
        self.calls = 0
        self.requests: list[LLMRequest] = []

    def push(self, turns: Iterable[LLMTurn]) -> None:
        self._turns.extend(turns)

    @property
    def remaining(self) -> int:
        return len(self._turns)

    def next_turn(self, request: LLMRequest) -> LLMTurn:
        self.calls += 1
        self.requests.append(request)
        if not self._turns:
            raise AssertionError("ScriptedLLM called but no scripted turn is left")
        return self._turns.popleft()


class PolicyLLM:
    """Deterministic stand-in for the LLM: a small tool-choosing policy (see policy.py).

    Used for local replay without AWS and as the fallback when Bedrock fails (Phase 6).
    """

    name = "policy"

    def next_turn(self, request: LLMRequest) -> LLMTurn:
        return policy_turn(request.context)
