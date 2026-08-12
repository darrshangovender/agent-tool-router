"""agent-tool-router — two-stage tool routing with explicit refusal.

Public surface:
    Router           — orchestrates embedding prefilter + LLM judge
    Tool             — a registered tool description
    RouteDecision    — what the router returns (either a tool pick or a refusal)
    RefusalReason    — enum of the four refusal categories
"""

from .embedding_prefilter import EmbeddingPrefilter
from .llm_judge import JudgeDecision, LLMJudge
from .refusal import RefusalReason
from .router import RouteDecision, Router, Tool

__version__ = "0.1.0"
__all__ = [
    "Router",
    "Tool",
    "RouteDecision",
    "RefusalReason",
    "EmbeddingPrefilter",
    "LLMJudge",
    "JudgeDecision",
]
