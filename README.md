# agent-tool-router — A small, well-evaluated tool-use router

> Given a user question, decide which of N tools to call (or none), parse arguments, and return a structured ToolCall.

**Why this exists.** Every "AI agent" framework hand-waves tool selection. In practice this single decision is where ~40% of agent failures originate — picking the wrong tool, hallucinating arguments, or invoking a tool when the model should have just answered.

This repo is a focused router with explicit evaluation. No agent loop, no memory, no rabbit holes — just the routing decision, done well.

---

## What it does

```python
from agent_tool_router import Router, ToolSpec

tools = [
    ToolSpec(name="search_web", description="Fetch live information from the public internet.", schema={...}),
    ToolSpec(name="run_sql", description="Run a read-only SELECT against the warehouse.", schema={...}),
    ToolSpec(name="email", description="Send an email to a verified contact.", schema={...}),
]

router = Router(tools=tools, model="claude-sonnet-4-5")
decision = router.route("What was our top-selling SKU last month?")

# decision.tool        → "run_sql"
# decision.arguments   → {"query": "SELECT sku, SUM(qty) ... last_month"}
# decision.confidence  → 0.92
# decision.reasoning   → "User asks for analytics on internal data; SQL is appropriate."
```

If confidence is below a threshold, the router falls back to **"no tool — answer directly"** rather than guessing.

## Architecture

```
User question
   │
   ▼
┌─────────────────────────┐
│ Tool description encoder │ ── embeddings (one-time)
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ Top-3 candidate filter  │ ── embedding similarity over tool descriptions
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ LLM final decision      │ ── strict JSON schema, structured output
└──────────┬──────────────┘
           ▼
   ToolCall (or "no tool")
```

The two-stage filter (embeddings → LLM) keeps cost down: you only pay for the LLM call when the top candidates are close. For routine queries with one clear winner, the embedding step is decisive.

## Why structured output, not function calling

Provider-native function calling (OpenAI tools, Anthropic tool_use) ties you to one vendor's quirks. Structured JSON schema output is the same idea but portable; you can swap models without rewriting the agent.

## Evaluation

The repo ships with an eval set: 200 labelled (question, correct_tool, correct_arguments) triples covering:

- Correct tool, correct args (positive cases)
- Should answer directly, no tool needed
- Ambiguous between two tools (test the fallback)
- Out-of-scope (should refuse)

Metrics:
- **Tool accuracy** — % of cases where the right tool is chosen
- **Argument F1** — fuzzy match on extracted argument values
- **Refusal recall** — % of out-of-scope cases correctly refused

## Status (design)

- [ ] Tool registration + embedding index
- [ ] Two-stage routing pipeline
- [ ] Eval set + harness (200 labelled cases)
- [ ] Provider abstraction (Anthropic, OpenAI)

## Author

Darrshan Govender · [Agulhas Code](https://agulhascode.co.za)
