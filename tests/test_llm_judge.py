"""Tests for the LLM judge — schema validation, refusal mapping, parser robustness."""

from __future__ import annotations

import json

import pytest

from agent_tool_router.embedding_prefilter import Candidate
from agent_tool_router.llm_judge import (
    EchoBackend,
    JudgeDecision,
    LLMJudge,
    parse_judge_response,
)
from agent_tool_router.refusal import RefusalReason


# ---- Schema validation ----------------------------------------------------


def test_decision_requires_refusal_reason_when_tool_is_null():
    with pytest.raises(ValueError, match="refusal_reason"):
        JudgeDecision(tool=None, confidence=0.2, reasoning="no clear match", refusal_reason=None)


def test_decision_rejects_refusal_reason_when_tool_set():
    with pytest.raises(ValueError, match="refusal_reason"):
        JudgeDecision(
            tool="refund",
            confidence=0.9,
            reasoning="ok",
            refusal_reason=RefusalReason.AMBIGUOUS_MATCH,
        )


def test_decision_confidence_bounds():
    with pytest.raises(ValueError):
        JudgeDecision(tool="refund", confidence=1.5, reasoning="x", refusal_reason=None)
    with pytest.raises(ValueError):
        JudgeDecision(tool="refund", confidence=-0.1, reasoning="x", refusal_reason=None)


# ---- Parser ---------------------------------------------------------------


def test_parser_extracts_json_from_chatter():
    raw = "Sure, here is my decision:\n```json\n" + json.dumps(
        {"tool": "refund", "confidence": 0.88, "reasoning": "clear refund query", "refusal_reason": None}
    ) + "\n```"
    d = parse_judge_response(raw, allowed_tools={"refund", "ticket"})
    assert d.tool == "refund"
    assert d.confidence == 0.88


def test_parser_rejects_unknown_tool_as_low_confidence():
    raw = json.dumps({"tool": "hallucinated", "confidence": 0.9, "reasoning": "x", "refusal_reason": None})
    d = parse_judge_response(raw, allowed_tools={"refund"})
    assert d.tool is None
    assert d.refusal_reason == RefusalReason.LOW_CONFIDENCE


def test_parser_handles_garbage_as_low_confidence():
    d = parse_judge_response("totally not json", allowed_tools={"refund"})
    assert d.tool is None
    assert d.refusal_reason == RefusalReason.LOW_CONFIDENCE


def test_parser_handles_empty_response():
    d = parse_judge_response("", allowed_tools={"refund"})
    assert d.tool is None
    assert d.refusal_reason == RefusalReason.LOW_CONFIDENCE


# ---- EchoBackend judge (deterministic) ------------------------------------


def test_echo_judge_picks_clear_winner():
    judge = LLMJudge(EchoBackend(decisive_margin=0.1, low_floor=0.2))
    cands = [Candidate("refund", 0.85), Candidate("ticket", 0.30)]
    d = judge.decide("I want a refund", cands, tool_descriptions={"refund": "issue refund", "ticket": "open ticket"})
    assert d.tool == "refund"
    assert d.refusal_reason is None
    assert d.confidence > 0.5


def test_echo_judge_refuses_ambiguous():
    judge = LLMJudge(EchoBackend(decisive_margin=0.2, low_floor=0.0))
    cands = [Candidate("account_lookup", 0.71), Candidate("billing_query", 0.68)]
    d = judge.decide(
        "look up customer 4421",
        cands,
        tool_descriptions={"account_lookup": "a", "billing_query": "b"},
    )
    assert d.tool is None
    assert d.refusal_reason == RefusalReason.AMBIGUOUS_MATCH


def test_echo_judge_refuses_no_match():
    judge = LLMJudge(EchoBackend(decisive_margin=0.1, low_floor=0.4))
    cands = [Candidate("refund", 0.1), Candidate("ticket", 0.05)]
    d = judge.decide("what's the capital of France", cands, tool_descriptions={"refund": "", "ticket": ""})
    assert d.tool is None
    assert d.refusal_reason == RefusalReason.NO_MATCHING_TOOL


def test_echo_judge_empty_candidates_refuses():
    judge = LLMJudge(EchoBackend())
    d = judge.decide("anything", [], tool_descriptions={})
    assert d.tool is None
