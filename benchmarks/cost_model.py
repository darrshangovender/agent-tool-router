"""Illustrative arithmetic: what does the prefilter save on the judge call?

    python benchmarks/cost_model.py

This is a CALCULATION, not a measurement. The token counts below are ASSUMPTIONS (roughly what the
prompt in `agent_tool_router/llm_judge.py` costs); change them to match your own prompts, and measure
real usage before quoting a figure. Prices are list prices per million tokens read from the
providers' pricing pages on 2026-10-14:

    claude-haiku-4-5   input $1.00   output $5.00    (platform.claude.com/docs/en/about-claude/pricing)
    gpt-4o-mini        input $0.15   output $0.60    (developers.openai.com/api/docs/pricing)

The comparison is "show the judge every tool" against "show it the top-k the prefilter shortlisted".
The saving is only as good as the prefilter's recall@k: when the right tool is not in the shortlist the
judge cannot pick it (see the README's measured recall).
"""

from __future__ import annotations

SYSTEM_TOKENS = 230        # system prompt (rules and schema): assumption
QUERY_TOKENS = 20          # the user's query: assumption
TOKENS_PER_TOOL = 40       # one candidate block (name, description, score): assumption
OUTPUT_TOKENS = 80         # the JSON decision: assumption
PRICES = {"claude-haiku-4-5": (1.00, 5.00), "gpt-4o-mini": (0.15, 0.60)}
TOOL_COUNTS = (5, 20, 50)
TOP_K = 3


def cost_per_route(model: str, shown_tools: int) -> float:
    p_in, p_out = PRICES[model]
    tokens_in = SYSTEM_TOKENS + QUERY_TOKENS + shown_tools * TOKENS_PER_TOOL
    return (tokens_in * p_in + OUTPUT_TOKENS * p_out) / 1_000_000


def table() -> str:
    lines = [
        f"| judge model | tools in catalogue | judge sees all tools (USD / 1,000 routes) "
        f"| judge sees top-{TOP_K} (USD / 1,000 routes) | saving |",
        "|---|---|---|---|---|",
    ]
    for model in PRICES:
        for n in TOOL_COUNTS:
            full = cost_per_route(model, n) * 1000
            short = cost_per_route(model, min(TOP_K, n)) * 1000
            lines.append(f"| {model} | {n} | ${full:.3f} | ${short:.3f} | {1 - short / full:.0%} |")
    return "\n".join(lines)


def main() -> int:
    print(f"assumed tokens: system {SYSTEM_TOKENS}, query {QUERY_TOKENS}, {TOKENS_PER_TOOL} per tool "
          f"shown, output {OUTPUT_TOKENS}\n")
    print(table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
