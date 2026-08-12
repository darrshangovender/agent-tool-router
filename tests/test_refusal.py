"""Tests for the refusal taxonomy and Router → Refusal mapping."""

from __future__ import annotations

import json

import pytest

from agent_tool_router import RefusalReason, Router, Tool
from agent_tool_router.embedding_prefilter import StubEncoder
from agent_tool_router.llm_judge import EchoBackend, LLMJudge, LLMBackend
from agent_tool_router.refusal import Refusal


class ScriptedBackend:
    """Returns a pre-baked JSON string regardless of input — used to force refusal paths."""

    model = "scripted"

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def complete(self, system: str, user: str, *, max_tokens: int = 400) -> str:
        return json.dumps(self._payload)


def _router_with(backend: LLMBackend, *, threshold: float = 0.5) -> Router:
    tools = [
        Tool(name="refund", description="issue a refund to a customer"),
        Tool(name="ticket", description="open a support ticket"),
        Tool(name="lookup", description="look up a customer account"),
    ]
    return Router(
        tools,
        encoder=StubEncoder(dim=64),
        judge=LLMJudge(backend),
        confidence_threshold=threshold,
    )


@pytest.mark.parametrize(
    "reason",
    [
        RefusalReason.NO_MATCHING_TOOL,
        RefusalReason.AMBIGUOUS_MATCH,
        RefusalReason.OUT_OF_SCOPE,
        RefusalReason.LOW_CONFIDENCE,
    ],
)
def test_refusal_round_trips_through_router(reason):
    backend = ScriptedBackend(
        {
            "tool": None,
            "confidence": 0.1,
            "reasoning": f"forcing {reason.value}",
            "refusal_reason": reason.value,
        }
    )
    r = _router_with(backend)
    d = r.route("any query")
    assert d.is_refusal
    assert d.refusal.reason == reason
    assert d.refusal.message
    assert d.refusal.judge_confidence == pytest.approx(0.1)


def test_low_confidence_refusal_from_threshold_overrides_route():
    # Judge picks a tool with 0.3 confidence; router threshold of 0.8 promotes to refusal.
    backend = ScriptedBackend(
        {"tool": "refund", "confidence": 0.3, "reasoning": "guess", "refusal_reason": None}
    )
    r = _router_with(backend, threshold=0.8)
    d = r.route("any query")
    assert d.is_refusal
    assert d.refusal.reason == RefusalReason.LOW_CONFIDENCE
    assert "below threshold" in d.refusal.message


def test_refusal_includes_candidate_diagnostics():
    backend = ScriptedBackend(
        {"tool": None, "confidence": 0.0, "reasoning": "no match", "refusal_reason": "no_matching_tool"}
    )
    r = _router_with(backend)
    d = r.route("anything")
    assert d.refusal.candidates  # prefilter still ran and produced a shortlist


def test_refusal_str_is_informative():
    ref = Refusal(reason=RefusalReason.OUT_OF_SCOPE, message="policy")
    assert "out_of_scope" in str(ref)
    assert "policy" in str(ref)


def test_echo_backend_low_floor_produces_no_match_refusal():
    backend = EchoBackend(decisive_margin=0.1, low_floor=0.9)  # nothing will clear floor
    r = _router_with(backend)
    d = r.route("refund my order please")
    assert d.is_refusal
    assert d.refusal.reason == RefusalReason.NO_MATCHING_TOOL
