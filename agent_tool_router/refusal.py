"""Explicit refusal categories.

Routing systems silently picking "the closest tool" is one of the most common
failure modes in production agent stacks. We treat refusal as a first-class
outcome — not as an error — and force callers to handle it.

The four reasons are mutually exclusive and cover every "do not route" case:

    no_matching_tool   The top candidates clearly do not solve the query.
                       Example: query "what's the weather in Rome" against
                       a catalogue of SQL / refund / ticket tools.

    ambiguous_match    Two or more candidates are equally plausible and the
                       judge cannot disambiguate without more user context.
                       Example: "look up customer 4421" against both
                       account_lookup and billing_query.

    low_confidence     The judge picked a tool but its self-reported
                       confidence is below the router's threshold. Surfaces
                       borderline routes for human review instead of acting
                       on a coin-flip.

    out_of_scope       The query is well-formed but asks for capability that
                       the agent is not allowed to perform (PII export,
                       financial advice, anything policy-blocked). Distinct
                       from no_matching_tool — the agent could in principle
                       answer, but shouldn't.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RefusalReason(str, Enum):
    NO_MATCHING_TOOL = "no_matching_tool"
    AMBIGUOUS_MATCH = "ambiguous_match"
    LOW_CONFIDENCE = "low_confidence"
    OUT_OF_SCOPE = "out_of_scope"


class Refusal(BaseModel):
    """Structured refusal returned to the caller.

    `candidates` lists what the prefilter considered so a human reviewer can
    diagnose whether the corpus is missing a tool or the judge mis-fired.
    """

    reason: RefusalReason
    message: str = Field(description="Human-readable explanation; safe to surface to end users.")
    candidates: list[str] = Field(default_factory=list, description="Tool names the prefilter shortlisted.")
    judge_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    def __str__(self) -> str:
        return f"Refusal({self.reason.value}: {self.message})"
