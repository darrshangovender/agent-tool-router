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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

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
    #: Set when the refusal was caused by the judge backend failing (`on_judge_error="refuse"`),
    #: not by the judge's own verdict: lets you alert on outages instead of reading them as taste.
    error: str | None = None

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
        Optional pre-built encoder. If None, the offline lexical `HashingEncoder` is used:
        deterministic and dependency-free, but it matches shared words, not meaning. Pass
        `make_encoder("sentence-transformers")` or `make_encoder("openai")` for semantic matching.
    judge:
        Optional pre-built `LLMJudge`. If None, the EchoBackend stub is used —
        good for tests, NOT for production routing quality.
    top_k:
        How many candidates the judge sees. 3-5 is the sweet spot.
    confidence_threshold:
        Picks with judge confidence below this become LOW_CONFIDENCE refusals.
    cache:
        Optional `EmbeddingCache` for query-side embedding reuse.
    policy:
        Optional `policy(query) -> str | None`, run BEFORE any embedding or model call. Return a
        message to refuse the query as OUT_OF_SCOPE (blocked capability, PII export, ...), or None
        to let it through. Scores cannot tell "allowed topic" from "forbidden request", and an LLM
        judge can be talked round; an explicit rule is the dependable backstop.
    on_judge_error:
        "raise" (default) lets a failing judge backend (rate limit, outage) propagate. "refuse"
        returns a LOW_CONFIDENCE refusal with `RouteDecision.error` set instead.
    max_query_chars:
        `route()` raises `ValueError` for a longer query (default 4,000). Policy regular
        expressions and the judge prompt are written for chat-length input, and a user-supplied
        pattern with two open-ended gaps takes time cubic in the query length.
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
        policy: Callable[[str], str | None] | None = None,
        on_judge_error: Literal["raise", "refuse"] = "raise",
        max_query_chars: int = 4000,
    ) -> None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be an integer >= 1")
        if (not isinstance(confidence_threshold, int | float)
                or not 0.0 <= confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0 and 1")
        if on_judge_error not in ("raise", "refuse"):
            raise ValueError("on_judge_error must be 'raise' or 'refuse'")
        if not isinstance(max_query_chars, int) or isinstance(max_query_chars, bool) \
                or max_query_chars < 1:
            raise ValueError("max_query_chars must be an integer >= 1")
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
            encoder=encoder or make_encoder("hashing"),
            cache=cache,
        )
        self.judge = judge or LLMJudge()
        self.policy = policy
        self.on_judge_error = on_judge_error
        self.max_query_chars = max_query_chars

    # ---- public API ----------------------------------------------------

    def route(self, query: str) -> RouteDecision:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if len(query) > self.max_query_chars:
            raise ValueError(
                f"query is {len(query)} characters; the limit is {self.max_query_chars} "
                "(max_query_chars). Policy rules and the prompt are sized for chat-length input."
            )
        t0 = time.perf_counter()
        if self.policy is not None:
            blocked = self.policy(query)
            if blocked:
                return self._refuse(RefusalReason.OUT_OF_SCOPE, blocked, [], t0, judge_conf=None)
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
        args = {name: tool.args_schema for name, tool in self.tools.items() if tool.args_schema}
        try:
            decision: JudgeDecision = self.judge.decide(
                query, candidates, tool_descriptions=descriptions, tool_args=args,
            )
        except Exception as exc:
            if self.on_judge_error == "raise":
                raise
            message = f"Judge backend failed ({type(exc).__name__}); not routing."
            out = self._refuse(RefusalReason.LOW_CONFIDENCE, message, candidates, t0, None)
            out.error = f"{type(exc).__name__}: {exc}"[:300]
            return out

        if decision.tool is None:
            if decision.refusal_reason is None:       # unreachable: JudgeDecision validates this
                raise RuntimeError("judge returned neither a tool nor a refusal reason")
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
