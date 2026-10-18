"""Shared test doubles. Nothing here touches the network or a model."""

from __future__ import annotations

import json

import numpy as np

from agent_tool_router import Router, Tool
from agent_tool_router.embedding_prefilter import HashingEncoder
from agent_tool_router.llm_judge import LLMJudge

CATALOGUE = [
    Tool(name="issue_refund", description="Issue a refund to a customer for an order.",
         args_schema={"order_id": "string"}, tags=["money", "refund"]),
    Tool(name="create_ticket", description="Open a support ticket for human follow up."),
    Tool(name="kb_search", description="Search the knowledge base for an existing answer."),
    Tool(name="schedule_call", description="Book a callback appointment with the customer."),
]


def verdict(tool: str | None, confidence: float = 0.9, reason: str | None = None,
            reasoning: str = "because") -> dict:
    return {"tool": tool, "confidence": confidence, "reasoning": reasoning,
            "refusal_reason": reason}


class ScriptedBackend:
    """Returns the scripted replies in order (the last one repeats) and records every call."""

    model = "scripted"

    def __init__(self, *replies: str | dict | Exception) -> None:
        self.replies = list(replies) or [json.dumps(verdict("issue_refund"))]
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, *, max_tokens: int = 400) -> str:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


class CountingEncoder(HashingEncoder):
    """HashingEncoder that counts how many texts it was asked to encode."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.encoded: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.encoded.extend(texts)
        return super().encode(texts)


def make_router(backend: ScriptedBackend | None = None, **kw) -> Router:
    judge = kw.pop("judge", None)
    if judge is None and backend is not None:
        judge = LLMJudge(backend)
    kw.setdefault("encoder", HashingEncoder())
    return Router(CATALOGUE, judge=judge, **kw)
