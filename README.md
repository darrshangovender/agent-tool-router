# agent-tool-router

> Given a user query and a catalogue of tools, pick the right one — or refuse, with a typed reason.

Every agent framework hand-waves tool selection. In practice, this single decision is where most agent failures originate: picking the wrong tool, hallucinating arguments, or invoking a tool when the model should have refused. This repo is a focused two-stage router with explicit evaluation. No agent loop, no memory, no rabbit holes — just the routing decision, done well.

---

## Architecture

```
              ┌─────────────────────────────────────────┐
              │              User query                 │
              └─────────────────────┬───────────────────┘
                                    │
              ┌─────────────────────▼───────────────────┐
              │  Stage 1: Embedding prefilter           │
              │  - tools encoded once at startup        │
              │  - query encoded per call (cached)      │
              │  - cosine similarity → top-K candidates │
              └─────────────────────┬───────────────────┘
                                    │   K candidates (default 3)
              ┌─────────────────────▼───────────────────┐
              │  Stage 2: LLM judge                     │
              │  - sees shortlist + query               │
              │  - returns Pydantic-validated JSON      │
              │  - picks a tool OR refuses with reason  │
              └─────────────────────┬───────────────────┘
                                    │
              ┌─────────────────────▼───────────────────┐
              │  RouteDecision                          │
              │  • tool + confidence + reasoning        │
              │  OR                                     │
              │  • Refusal(reason, message, candidates) │
              └─────────────────────────────────────────┘
```

Refusal is a first-class outcome with four reasons: `no_matching_tool`, `ambiguous_match`, `low_confidence`, `out_of_scope`. The router never silently picks the closest tool. See [`agent_tool_router/refusal.py`](agent_tool_router/refusal.py) and [`docs/architecture.md`](docs/architecture.md) for the rationale.

## Quick start

```python
from agent_tool_router import Router, Tool

tools = [
    Tool(name="issue_refund",   description="Issue a refund for a specific order."),
    Tool(name="kb_search",      description="Search the help docs for an answer."),
    Tool(name="escalate",       description="Hand off to a live human agent."),
]

router = Router(tools)  # local sentence-transformers + EchoBackend stub by default

d = router.route("I want my money back for order 9921")
if d.is_route:
    print(d.tool, d.confidence)      # "issue_refund" 0.92
else:
    print(d.refusal.reason, d.refusal.message)
```

Swap in a real judge with two lines:

```python
from agent_tool_router.llm_judge import LLMJudge, AnthropicBackend
router = Router(tools, judge=LLMJudge(AnthropicBackend(model="claude-haiku-4-5")))
```

Run the worked examples:

```
python examples/customer_support_bot.py
python examples/dev_assistant.py
```

## Benchmarks

50-case routing corpus, customer-support catalogue (8 tools). Latencies measured on a 2024 laptop CPU.

| Configuration | Routing acc. | Refusal precision | Refusal recall | Median latency | $ / 1k routes |
|---|---:|---:|---:|---:|---:|
| MiniLM + EchoBackend (no API)         | 78% | 92% | 71% | 9 ms   | $0.000 |
| MiniLM + Haiku judge                  | 96% | 98% | 89% | 240 ms | $0.012 |
| MiniLM + GPT-4o-mini judge            | 94% | 96% | 87% | 310 ms | $0.001 |
| OpenAI embeddings + Haiku judge       | 96% | 98% | 91% | 380 ms | $0.015 |

Numbers reproducible via [`benchmarks/run.py`](benchmarks/run.py); corpus at [`benchmarks/routing_corpus.yml`](benchmarks/routing_corpus.yml).

```
python benchmarks/run.py --backend anthropic --judge-model claude-haiku-4-5
```

## When to use this vs a single LLM call

| Use this router | Use a single judge call directly |
|---|---|
| 10+ tools | <5 tools |
| You want a small judge model (Haiku, 4o-mini) to do the work | You're already paying for a frontier model |
| You need typed refusal reasons for human review | "I don't know" is a fine fallback |
| Per-route cost matters (high traffic) | Latency floor is sub-50ms |
| Tool descriptions are distinct enough for embeddings to separate them | All tool descriptions are short and near-identical |

The cost arithmetic is worked out in [`docs/architecture.md`](docs/architecture.md). The judge prompt and the reasoning behind each line of it is in [`docs/prompt-design.md`](docs/prompt-design.md).

## What's in the box

| Path | What it is |
|---|---|
| [`agent_tool_router/router.py`](agent_tool_router/router.py) | Top-level `Router` that orchestrates the two stages |
| [`agent_tool_router/embedding_prefilter.py`](agent_tool_router/embedding_prefilter.py) | Stage 1 — encodes the corpus, ranks queries; local + OpenAI + stub backends |
| [`agent_tool_router/llm_judge.py`](agent_tool_router/llm_judge.py) | Stage 2 — Pydantic-validated judge with Anthropic / OpenAI / Echo backends |
| [`agent_tool_router/refusal.py`](agent_tool_router/refusal.py) | The four refusal reasons + the structured `Refusal` type |
| [`agent_tool_router/cache.py`](agent_tool_router/cache.py) | In-memory + SQLite cache for query embeddings |
| [`examples/customer_support_bot.py`](examples/customer_support_bot.py) | 8 tools, 20 demo queries — runs with no API key |
| [`examples/dev_assistant.py`](examples/dev_assistant.py) | 10 tools, 15 demo queries — ChatOps surface |
| [`benchmarks/run.py`](benchmarks/run.py) | Accuracy, refusal correctness, latency, cost |
| [`tests/`](tests/) | `pytest` suite — runs with no API key, no model download |

## Install

```
pip install -e .                     # base install: numpy, pydantic, pyyaml
pip install -e ".[dev]"              # adds sentence-transformers + pytest + ruff
pip install -e ".[anthropic,openai]" # adds real judge backends
```

## Status

- [x] Tool registration + embedding index
- [x] Two-stage routing pipeline
- [x] Refusal taxonomy with structured reasons
- [x] Eval set (50 cases) + harness with accuracy/refusal/latency/cost
- [x] Provider abstraction (Anthropic, OpenAI, Echo)
- [x] Query-embedding cache (in-memory + SQLite)

## Author

Darrshan Govender · [Agulhas Code](https://agulhascode.co.za)
