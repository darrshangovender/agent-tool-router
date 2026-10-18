"""Stage 2: an LLM picks THE tool from the shortlist (or refuses).

The judge sees only the candidates the prefilter shortlisted, plus the raw
query. Its output is strictly typed via Pydantic — any model response that
fails schema validation is treated as a low-confidence refusal rather than
silently passed through.

A provider-portable `LLMBackend` protocol lets callers plug in Anthropic /
OpenAI / a custom mock. The default `EchoBackend` is a deterministic stub
used by tests and offline demos — it inspects the candidate scores and
returns a structured pick without any network call.

The prompt itself lives in `docs/prompt-design.md`; the function that
materialises it (`build_judge_prompt`) is kept here so that prompt and parser
stay in lockstep.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .embedding_prefilter import Candidate
from .refusal import RefusalReason

# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------


REASONING_MAX = 600


class JudgeDecision(BaseModel):
    """Strictly-validated output of the LLM judge.

    `tool` is the name of the chosen tool, or None when the judge wants to
    refuse. `refusal_reason` must be set iff `tool` is None.
    """

    tool: str | None = Field(default=None, description="Chosen tool name, or null to refuse.")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-reported confidence in the pick.")
    reasoning: str = Field(
        min_length=1, max_length=REASONING_MAX, description="One-sentence justification.",
    )
    refusal_reason: RefusalReason | None = Field(
        default=None,
        description="Required when tool is null; must be unset when a tool is chosen.",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _numeric_confidence(cls, v: Any) -> Any:
        # Pydantic's lax mode turns `true` into 1.0 and "0.9" into 0.9. A judge that replied
        # {"confidence": true} would have cleared every threshold with a perfect score.
        if isinstance(v, bool) or isinstance(v, str):
            raise ValueError("confidence must be a JSON number")
        return v

    @field_validator("reasoning", mode="before")
    @classmethod
    def _trim_reasoning(cls, v: Any) -> Any:
        # The justification is for logs. A decision with a valid tool is not thrown away because
        # the model was wordy, but a blank one is still an error (min_length applies after strip).
        if isinstance(v, str):
            return v.strip()[:REASONING_MAX]
        return v

    @model_validator(mode="after")
    def _refusal_consistent(self) -> JudgeDecision:
        # A field validator on `refusal_reason` never runs when the field is OMITTED (pydantic does
        # not validate defaults), so {"tool": null, "confidence": 0.5, "reasoning": "x"} produced a
        # decision with neither a tool nor a reason and crashed the router's invariant check.
        if self.tool is None and self.refusal_reason is None:
            raise ValueError("refusal_reason is required when tool is null")
        if self.tool is not None and self.refusal_reason is not None:
            raise ValueError("refusal_reason must be null when a tool is chosen")
        return self


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    """Anything that takes a (system, user) prompt and returns the raw text."""

    model: str

    def complete(self, system: str, user: str, *, max_tokens: int = 400) -> str: ...


class EchoBackend:
    """Deterministic stub. Picks the top-scoring candidate iff its score is
    meaningfully ahead of the second; otherwise refuses with a reason that
    matches the gap pattern.

    Used by tests, the default benchmark run, and any environment without API
    keys. Not a recommendation for production — swap in `AnthropicBackend` or
    `OpenAIBackend` for real routing quality.
    """

    model = "echo-stub"

    #: Chosen on the dev half of benchmarks/routing_corpus.yml for the offline HashingEncoder
    #: (`python benchmarks/run.py --tune --sweep`). Cosine scores differ by encoder, so with another
    #: encoder re-tune, or ignore this stub and use a real backend.
    DEFAULT_LOW_FLOOR = 0.15
    DEFAULT_MARGIN = 0.10

    def __init__(self, decisive_margin: float = DEFAULT_MARGIN,
                 low_floor: float = DEFAULT_LOW_FLOOR) -> None:
        if decisive_margin < 0 or low_floor < 0:
            raise ValueError("decisive_margin and low_floor must be >= 0")
        self.decisive_margin = decisive_margin
        self.low_floor = low_floor

    def complete(self, system: str, user: str, *, max_tokens: int = 400) -> str:
        # The user prompt embeds a JSON candidate block; parse it back out.
        candidates = _extract_candidate_block(user)
        candidates = [c for c in candidates if isinstance(c, dict) and "score" in c and "name" in c]
        if not candidates:
            return json.dumps(
                {
                    "tool": None,
                    "confidence": 0.0,
                    "reasoning": "No candidates available.",
                    "refusal_reason": RefusalReason.NO_MATCHING_TOOL.value,
                }
            )
        top = candidates[0]
        second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
        gap = top["score"] - second_score
        if top["score"] < self.low_floor:
            return json.dumps(
                {
                    "tool": None,
                    "confidence": round(top["score"], 2),
                    "reasoning": f"Top candidate {top['name']} only scored {top['score']:.2f}.",
                    "refusal_reason": RefusalReason.NO_MATCHING_TOOL.value,
                }
            )
        # The scores arrive rounded to 4 places, so 0.30 - 0.20 is 0.09999999999999998: a gap
        # equal to the margin must count as decisive, not as a float-rounding ambiguity.
        if gap < self.decisive_margin - 1e-9 and len(candidates) > 1:
            return json.dumps(
                {
                    "tool": None,
                    "confidence": round(0.5 + gap, 2),
                    "reasoning": (
                        f"{top['name']} and {candidates[1]['name']} are within "
                        f"{gap:.2f} similarity."
                    ),
                    "refusal_reason": RefusalReason.AMBIGUOUS_MATCH.value,
                }
            )
        return json.dumps(
            {
                "tool": top["name"],
                "confidence": round(min(0.99, 0.5 + top["score"] / 2), 2),
                "reasoning": (f"{top['name']} is the clear match "
                              f"(score {top['score']:.2f})."),
                "refusal_reason": None,
            }
        )


class RetryingBackend:
    """Retry transient failures of another backend with exponential backoff.

    Wrap a real backend: `RetryingBackend(AnthropicBackend(), retries=2)`. Every exception is
    retried (the SDKs' rate-limit and overload errors are plain exceptions with no common base);
    after the last attempt the original exception propagates, and `Router(on_judge_error="refuse")`
    decides what the caller sees. `sleep` is injectable for tests.
    """

    def __init__(self, inner: LLMBackend, *, retries: int = 2, base_delay: float = 0.5,
                 retry_on: tuple[type[Exception], ...] = (Exception,),
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if retries < 0:
            raise ValueError("retries must be >= 0")
        if base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        self.inner, self.retries, self.base_delay, self._sleep = inner, retries, base_delay, sleep
        self.retry_on = retry_on
        self.model = inner.model

    def complete(self, system: str, user: str, *, max_tokens: int = 400) -> str:
        for attempt in range(self.retries + 1):
            try:
                return self.inner.complete(system, user, max_tokens=max_tokens)
            except self.retry_on:
                if attempt == self.retries:
                    raise
                self._sleep(self.base_delay * (2 ** attempt))
        raise AssertionError("unreachable")  # pragma: no cover


class AnthropicBackend:
    """Real Claude backend. Requires `anthropic` + ANTHROPIC_API_KEY (or pass `client=`).

    `temperature` is sent as given (default 0.0: routing is classification). Newer Claude models
    reject sampling parameters outright (Claude Haiku 5.5 answers 400 to any `temperature` other
    than 1), so pass `temperature=None` to leave it out. Only `text` blocks of the reply are
    read, so thinking blocks never end up in the JSON the parser sees.
    """

    def __init__(self, model: str = "claude-haiku-4-5", *, temperature: float | None = 0.0,
                 client: Any = None) -> None:
        self.model = model
        self.temperature = temperature
        if client is None:
            from anthropic import Anthropic  # noqa: WPS433

            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._client = client

    def complete(self, system: str, user: str, *, max_tokens: int = 400) -> str:
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            # Anthropic 1.15.0 removed `temperature` from the typed Messages.create signature;
            # pass it through `extra_body` so the field reaches the HTTP request either way, and
            # the backend keeps working across SDK versions. A None skips it entirely for models
            # that reject the parameter.
            kwargs["extra_body"] = {"temperature": self.temperature}
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        return "".join(
            b.text for b in resp.content
            if getattr(b, "type", None) == "text" and isinstance(getattr(b, "text", None), str)
        )


class OpenAIBackend:
    """Real OpenAI backend. Requires `openai` + OPENAI_API_KEY (or pass `client=`).

    `temperature=None` leaves the parameter out, for models that do not accept it.
    """

    def __init__(self, model: str = "gpt-4o-mini", *, temperature: float | None = 0.0,
                 client: Any = None) -> None:
        self.model = model
        self.temperature = temperature
        if client is None:
            from openai import OpenAI  # noqa: WPS433

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._client = client

    def complete(self, system: str, user: str, *, max_tokens: int = 400) -> str:
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are a tool routing judge.

You receive a user query and a shortlist of candidate tools (name, description, prefilter similarity score). \
Your job is to choose AT MOST one tool, or refuse.

Refuse with one of these reasons when appropriate:
  - no_matching_tool: none of the candidates plausibly answers the query.
  - ambiguous_match:  two or more candidates are equally plausible.
  - out_of_scope:     the query asks for policy-blocked or non-routable capability.
  - low_confidence:   you would guess, but the choice is not defensible.

Return STRICT JSON matching this schema:
{
  "tool": string | null,
  "confidence": number in [0, 1],
  "reasoning": string (<= 1 sentence, <= 600 chars),
  "refusal_reason": "no_matching_tool" | "ambiguous_match" | "out_of_scope" | "low_confidence" | null
}

Candidates may list the arguments each tool takes ("args"); prefer a tool whose arguments the query can \
actually supply.

Rules:
  - If "tool" is null, "refusal_reason" MUST be set.
  - If "tool" is set, "refusal_reason" MUST be null.
  - Do not invent tool names — only pick from the candidate list.
"""  # noqa: E501  (the prompt's own line breaks are part of the text; wrapping would change it)


def build_judge_prompt(query: str, candidates: list[dict]) -> str:
    """Render the user-side of the judge prompt. Pure function, no side effects."""
    cand_json = json.dumps(candidates, indent=2)
    return (
        f"USER QUERY:\n{query}\n\n"
        f"CANDIDATE TOOLS (ranked by embedding similarity, highest first):\n{cand_json}\n\n"
        f"Return your decision as JSON only."
    )


class LLMJudge:
    """Wrap a backend with prompt building, JSON parsing, and Pydantic validation."""

    def __init__(self, backend: LLMBackend | None = None, *, max_tokens: int = 400) -> None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer >= 1")
        self.backend: LLMBackend = backend if backend is not None else EchoBackend()
        self.max_tokens = max_tokens

    def decide(
        self,
        query: str,
        candidates: list[Candidate],
        *,
        tool_descriptions: dict[str, str] | None = None,
        tool_args: dict[str, dict] | None = None,
    ) -> JudgeDecision:
        cand_blocks = []
        for c in candidates:
            block: dict = {
                "name": c.name,
                "score": round(c.score, 4),
                "description": (tool_descriptions or {}).get(c.name, ""),
            }
            args = (tool_args or {}).get(c.name)
            if args:
                block["args"] = args          # can the query supply them? a judge weighs that
            cand_blocks.append(block)
        user_prompt = build_judge_prompt(query, cand_blocks)
        raw = self.backend.complete(SYSTEM_PROMPT, user_prompt, max_tokens=self.max_tokens)
        return parse_judge_response(raw, allowed_tools={c.name for c in candidates})


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_MAX_OBJECTS = 20


def _json_objects(text: str) -> list[dict]:
    """Every top-level JSON object in `text`, in order (at most 20; nested objects are not listed
    separately). A greedy brace-to-brace regex used to swallow trailing prose containing another
    brace and then fail to parse a perfectly good leading object."""
    decoder = json.JSONDecoder()
    found: list[dict] = []
    idx, tries = text.find("{"), 0
    while idx != -1 and tries < 200 and len(found) < _MAX_OBJECTS:
        try:
            obj, end = decoder.raw_decode(text, idx)
        except (json.JSONDecodeError, RecursionError):
            idx, tries = text.find("{", idx + 1), tries + 1
            continue
        if isinstance(obj, dict):
            found.append(obj)
            idx = text.find("{", end)
        else:
            idx = text.find("{", idx + 1)
    return found


def _is_verdict(obj: dict) -> bool:
    return "tool" in obj and "confidence" in obj


def parse_judge_response(raw: str, allowed_tools: set[str]) -> JudgeDecision:
    """Extract JSON from the raw model output and validate via Pydantic.

    A malformed response, a tool name not in `allowed_tools`, or a reply that contains two
    different decisions is converted to a `LOW_CONFIDENCE` refusal: the caller never has to handle
    parse errors, and never has to guess which of two verdicts the model meant. (Taking "the first
    object" let a reply that quoted the user's text, `{"tool": "rotate_secret", "confidence": 1}`,
    before its own verdict route to the quoted tool.)
    """
    if not isinstance(raw, str):
        return _refusal(RefusalReason.LOW_CONFIDENCE, "Judge backend returned no text.")
    text = raw.strip()
    if not text:
        return _refusal(RefusalReason.LOW_CONFIDENCE, "Empty response from judge.")

    objects = _json_objects(text)
    if not objects:
        return _refusal(RefusalReason.LOW_CONFIDENCE, "No JSON object in judge response.")
    verdicts = [o for o in objects if _is_verdict(o)]
    if any(v != verdicts[0] for v in verdicts[1:]):
        return _refusal(RefusalReason.LOW_CONFIDENCE,
                        "Judge reply contains conflicting decisions; not routing.")
    data = verdicts[0] if verdicts else objects[0]

    try:
        decision = JudgeDecision.model_validate(data)
    except ValidationError as exc:
        return _refusal(RefusalReason.LOW_CONFIDENCE,
                        f"Judge output failed validation: {exc}"[:500])

    if decision.tool is not None and decision.tool not in allowed_tools:
        return _refusal(
            RefusalReason.LOW_CONFIDENCE,
            f"Judge picked '{decision.tool}' which is not in the candidate set.",
        )
    return decision


def _refusal(reason: RefusalReason, message: str) -> JudgeDecision:
    return JudgeDecision(tool=None, confidence=0.0, reasoning=message, refusal_reason=reason)


_CANDIDATE_MARKER = "CANDIDATE TOOLS (ranked by embedding similarity, highest first):"


def _extract_candidate_block(prompt: str) -> list[dict]:
    """Helper for EchoBackend: pull the candidate JSON list back out of the prompt.

    It reads the array that follows the candidate header. Searching the whole prompt for the first
    `[` and the last `]` broke on any query containing brackets ("refund order [9921]"), which
    then looked like a catalogue with no candidates.
    """
    at = prompt.rfind(_CANDIDATE_MARKER)       # the real block is last; the query comes before it
    if at == -1:
        return []
    body = prompt[at + len(_CANDIDATE_MARKER):]
    start = body.find("[")
    if start == -1:
        return []
    try:
        data, _ = json.JSONDecoder().raw_decode(body, start)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []
