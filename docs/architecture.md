# Architecture — why two stages

The router has two stages on purpose. The cost trade-off only makes sense once you do the arithmetic, so here it is.

## The shape of the problem

A production agent typically registers between 5 and 50 tools. The naive approach — show every tool description to the LLM on every query — looks like this per route:

- Prompt overhead: ~80 tokens (system) + ~30 tokens (rules)
- Per-tool description: ~40 tokens on average
- Query: ~20 tokens
- Output: ~80 tokens

At **20 tools**, that's roughly `80 + 30 + 20*40 + 20 + 80 ≈ 1010` input tokens and 80 output tokens per route.

Using `claude-haiku-4-5` at $0.80/$4.00 per 1M tokens:

```
(1010 / 1_000_000) * 0.80 + (80 / 1_000_000) * 4.00 = $0.0011 per route
```

That's `$1.10 per 1000 routes` and `$1100 per 1M routes`. For a chat product running 10M routes/month, you're paying ~$11k/month just to pick a tool — before any actual tool work runs.

## What the two-stage filter changes

Stage 1 (embedding prefilter):

- Tool descriptions are encoded **once** at startup, not per query.
- Query embedding: 1 forward pass through a 22M-param MiniLM (~5ms on CPU). $0 if local; ~$0.00001 per route on the OpenAI embedding API.
- Cosine similarity over a small matrix: <1ms.

Stage 2 (LLM judge):

- Sees only the top **K=3** candidates instead of all 20.
- Prompt size drops from ~1010 to roughly `80 + 30 + 3*40 + 20 + 80 ≈ 330` input tokens.

New cost per route on Haiku:

```
(330 / 1_000_000) * 0.80 + (80 / 1_000_000) * 4.00 = $0.00059 per route
```

That's **~46% cheaper** at 20 tools, and the saving grows linearly with catalogue size. At 50 tools the naive route is $0.0019 vs the two-stage $0.00059 — **a 69% reduction**.

Plus the embedding step is deterministic and fully cacheable for templated queries (`show order #X`, `refund order #Y`), which knocks another chunk off in practice.

## When the two-stage filter does NOT pay off

The trade-off inverts when:

| Situation | Why |
|---|---|
| **Fewer than ~5 tools** | The prefilter overhead (the model load, the always-on 5ms embed) costs more than just sending all descriptions to a cheap judge. |
| **All tool descriptions are short and near-identical** | Embeddings can't separate "refund order" from "refund subscription" reliably; you'd be paying for an extra stage that adds noise. |
| **Latency floor is sub-50ms** | The local embedding adds 5-15ms; the OpenAI embedding API adds 80-200ms over the network. If your SLO is tight, route directly with a fast judge. |
| **You already paid for a frontier model** | A single Sonnet/GPT-4o call at 1000 tokens is fine. The two-stage matters most when you want to use a *small* judge (Haiku/4o-mini) and trust the prefilter to set it up for success. |

## Why the judge stays — even though the embedding could "just pick top-1"

Embeddings tell you "this query is in the neighbourhood of this tool description". They cannot tell you:

- Whether the query *fully* matches the tool's intent or only partially overlaps. (`refund and cancel` is half `issue_refund`, half `cancel_subscription`.)
- Whether two tools are equally plausible — which is the **ambiguous_match** refusal case.
- Whether the query is asking for something the agent *shouldn't* do (the **out_of_scope** refusal).
- Whether the judge would have to guess between two close candidates — the **low_confidence** path.

Picking top-1 by cosine is right ~70% of the time on the bundled benchmark. The LLM judge raises that to ~95% *and* gives you structured refusal data when it can't. The cost of that lift is one small-model call, gated by the prefilter so it stays cheap.

## Why structured output, not tool-calling

OpenAI tools and Anthropic `tool_use` are convenient but they:

1. Couple your agent to a specific provider's tool schema and parsing quirks.
2. Don't compose well with a refusal taxonomy — you'd have to add a `refuse_with_reason` synthetic tool to every catalogue.
3. Hide the model's reasoning (most providers don't surface why a tool was chosen).

The judge here returns plain JSON validated by Pydantic. Same idea, portable, easier to test, easier to log.
