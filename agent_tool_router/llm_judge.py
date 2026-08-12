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
import re
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator

from .embedding_prefilter import Candidate
from .refusal import RefusalReason


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------


class JudgeDecision(BaseModel):
    """Strictly-validated output of the LLM judge.

    `tool` is the name of the chosen tool, or None when the judge wants to
    refuse. `refusal_reason` must be set iff `tool` is None.
    """

    tool: str | None = Field(default=None, description="Chosen tool name, or null to refuse.")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-reported confidence in the pick.")
    reasoning: str = Field(min_length=1, max_length=600, description="One-sentence justification.")
    refusal_reason: RefusalReason | None = Field(
        default=None,
        description="Required when tool is null; must be unset when a tool is chosen.",
    )

    @field_validator("refusal_reason")
    @classmethod
    def _refusal_consistent(cls, v: RefusalReason | None, info) -> RefusalReason | None:
        tool = info.data.get("tool")
        if tool is None and v is None:
            raise ValueError("refusal_reason is required when tool is null")
        if tool is not None and v is not None:
            raise ValueError("refusal_reason must be null when a tool is chosen")
        return v


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

    def __init__(self, decisive_margin: float = 0.15, low_floor: float = 0.25) -> None:
        self.decisive_margin = decisive_margin
        self.low_floor = low_floor

    def complete(self, system: str, user: str, *, max_tokens: int = 400) -> str:
        # The user prompt embeds a JSON candidate block; parse it back out.
        candidates = _extract_candidate_block(user)
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
        if gap < self.decisive_margin and len(candidates) > 1:
            return json.dumps(
                {
                    "tool": None,
                    "confidence": round(0.5 + gap, 2),
                    "reasoning": (
                        f"{top['name']} and {candidates[1]['name']} are within {gap:.2f} similarity."
                    ),
                    "refusal_reason": RefusalReason.AMBIGUOUS_MATCH.value,
                }
            )
        return json.dumps(
            {
                "tool": top["name"],
                "confidence": round(min(0.99, 0.5 + top["score"] / 2), 2),
                "reasoning": f"{top['name']} is the clear semantic match (score {top['score']:.2f}).",
                "refusal_reason": None,
            }
        )


class AnthropicBackend:
    """Real Claude backend. Requires `anthropic` + ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        from anthropic import Anthropic  # noqa: WPS433

        self.model = model
        self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str, *, max_tokens: int = 400) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text  # type: ignore[union-attr]


class OpenAIBackend:
    """Real OpenAI backend. Requires `openai` + OPENAI_API_KEY."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI  # noqa: WPS433

        self.model = model
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def complete(self, system: str, user: str, *, max_tokens: int = 400) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
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

Rules:
  - If "tool" is null, "refusal_reason" MUST be set.
  - If "tool" is set, "refusal_reason" MUST be null.
  - Do not invent tool names — only pick from the candidate list.
"""


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

    def __init__(self, backend: LLMBackend | None = None) -> None:
        self.backend: LLMBackend = backend or EchoBackend()

    def decide(
        self,
        query: str,
        candidates: list[Candidate],
        *,
        tool_descriptions: dict[str, str] | None = None,
    ) -> JudgeDecision:
        cand_blocks = [
            {
                "name": c.name,
                "score": round(c.score, 4),
                "description": (tool_descriptions or {}).get(c.name, ""),
            }
            for c in candidates
        ]
        user_prompt = build_judge_prompt(query, cand_blocks)
        raw = self.backend.complete(SYSTEM_PROMPT, user_prompt)
        return parse_judge_response(raw, allowed_tools={c.name for c in candidates})


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_response(raw: str, allowed_tools: set[str]) -> JudgeDecision:
    """Extract JSON from the raw model output and validate via Pydantic.

    A malformed response or a tool name not in `allowed_tools` is converted to
    a `LOW_CONFIDENCE` refusal — the caller never has to handle parse errors.
    """
    text = raw.strip()
    if not text:
        return _refusal(RefusalReason.LOW_CONFIDENCE, "Empty response from judge.")

    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return _refusal(RefusalReason.LOW_CONFIDENCE, "No JSON object in judge response.")

    try:
        data = json.loads(match.group(0))
        decision = JudgeDecision.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        return _refusal(RefusalReason.LOW_CONFIDENCE, f"Judge output failed validation: {exc}")

    if decision.tool is not None and decision.tool not in allowed_tools:
        return _refusal(
            RefusalReason.LOW_CONFIDENCE,
            f"Judge picked '{decision.tool}' which is not in the candidate set.",
        )
    return decision


def _refusal(reason: RefusalReason, message: str) -> JudgeDecision:
    return JudgeDecision(tool=None, confidence=0.0, reasoning=message, refusal_reason=reason)


def _extract_candidate_block(prompt: str) -> list[dict]:
    """Helper for EchoBackend: pull the candidate JSON list back out of the prompt."""
    start = prompt.find("[")
    end = prompt.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        return json.loads(prompt[start : end + 1])
    except json.JSONDecodeError:
        return []
