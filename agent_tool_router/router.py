"""Top-level Router: orchestrates prefilter, judge, and refusal handling.

Typical flow per query:

    1.  EmbeddingPrefilter.top_k(query, k=top_k)         O(N) cosine
    2.  LLMJudge.decide(query, candidates)               1 LLM call
    3.  Apply confidence_threshold; convert LOW_CONFIDENCE
        responses or below-threshold picks into Refusal
    4.  Return RouteDecision

Everything is held by value — Router instances are cheap to construct but the
prefilter holds the encoded corpus, so reuse one Router per tool catalogue.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from pydantic import BaseModel, Field

from .cache import EmbeddingCache
from .embedding_prefilter import Candidate, EmbeddingPrefilter, Encoder, make_encoder
from .llm_judge import JudgeDecision, LLMJudge
from .refusal import Refusal, RefusalReason


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class Tool(BaseModel):
    """A tool the agent can route to.

    `description` is what the prefilter and judge see — invest in it. Two or
    three sentences covering what the tool does, what inputs it needs, and
    what kind of question it is the right answer to is the right shape.

    `args_schema` is optional metadata about argument keys; the router itself
    does not extract arguments (that's the next layer's job), but downstream
    callers often want to display the expected signature.
    """

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    args_schema: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    def embed_text(self) -> str:
        """Text fed to the encoder. Name + description + tags reads best in tests."""
        parts = [self.name, self.description]
        if self.tags:
            parts.append(" ".join(self.tags))
        return " — ".join(parts)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class RouteDecision:
    """End-to-end routing outcome.

    Exactly one of `tool` or `refusal` is set. Callers should branch on
    `decision.is_route` (or `decision.tool is not None`); the other fields
    are diagnostic.
    """

    tool: str | None
    refusal: Refusal | None
    confidence: float
    reasoning: str
    candidates: list[Candidate] = field(default_factory=list)
    latency_ms: int = 0

    @property
    def is_route(self) -> bool:
        return self.tool is not None

    @property
    def is_refusal(self) -> bool:
        return self.refusal is not None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class Router:
    """Two-stage tool router.

    Parameters
    ----------
    tools:
        The catalogue. Order does not matter; names must be unique.
    encoder:
        Optional pre-built encoder. If None, a sentence-transformers MiniLM
        model is loaded. Pass `make_encoder("openai")` to use the OpenAI
        embeddings API, or `make_encoder("stub")` for offline tests.
    judge:
        Optional pre-built `LLMJudge`. If None, the EchoBackend stub is used —
        good for tests, NOT for production routing quality.
    top_k:
        How many candidates the judge sees. 3-5 is the sweet spot.
    confidence_threshold:
        Picks with judge confidence below this become LOW_CONFIDENCE refusals.
    cache:
        Optional `EmbeddingCache` for query-side embedding reuse.
    """

    def __init__(
        self,
        tools: Iterable[Tool],
        *,
        encoder: Encoder | None = None,
        judge: LLMJudge | None = None,
        top_k: int = 3,
        confidence_threshold: float = 0.5,
        cache: EmbeddingCache | None = None,
    ) -> None:
        tool_list = list(tools)
        if not tool_list:
            raise ValueError("Router requires at least one tool")
        names = [t.name for t in tool_list]
        if len(set(names)) != len(names):
            raise ValueError(f"Tool names must be unique; got {names}")

        self.tools: dict[str, Tool] = {t.name: t for t in tool_list}
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.prefilter = EmbeddingPrefilter(
            tool_names=names,
            tool_texts=[t.embed_text() for t in tool_list],
            encoder=encoder or make_encoder("sentence-transformers"),
            cache=cache,
        )
        self.judge = judge or LLMJudge()

    # ---- public API ----------------------------------------------------

    def route(self, query: str) -> RouteDecision:
        t0 = time.perf_counter()
        candidates = self.prefilter.top_k(query, k=self.top_k)
        if not candidates:
            return self._refuse(
                RefusalReason.NO_MATCHING_TOOL,
                "Tool catalogue is empty.",
                [],
                t0,
                judge_conf=None,
            )

        descriptions = {name: tool.description for name, tool in self.tools.items()}
        decision: JudgeDecision = self.judge.decide(query, candidates, tool_descriptions=descriptions)

        if decision.tool is None:
            assert decision.refusal_reason is not None  # invariant from JudgeDecision validator
            return self._refuse(
                decision.refusal_reason,
                decision.reasoning,
                candidates,
                t0,
                judge_conf=decision.confidence,
            )

        if decision.confidence < self.confidence_threshold:
            return self._refuse(
                RefusalReason.LOW_CONFIDENCE,
                f"Judge picked {decision.tool} but confidence {decision.confidence:.2f} "
                f"is below threshold {self.confidence_threshold:.2f}.",
                candidates,
                t0,
                judge_conf=decision.confidence,
            )

        return RouteDecision(
            tool=decision.tool,
            refusal=None,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            candidates=candidates,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    def __len__(self) -> int:
        return len(self.tools)

    # ---- internal ------------------------------------------------------

    def _refuse(
        self,
        reason: RefusalReason,
        message: str,
        candidates: list[Candidate],
        t0: float,
        judge_conf: float | None,
    ) -> RouteDecision:
        refusal = Refusal(
            reason=reason,
            message=message,
            candidates=[c.name for c in candidates],
            judge_confidence=judge_conf,
        )
        return RouteDecision(
            tool=None,
            refusal=refusal,
            confidence=judge_conf or 0.0,
            reasoning=message,
            candidates=candidates,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
