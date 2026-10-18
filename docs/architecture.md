# Architecture: why two stages

The router has two stages on purpose. The trade-off is easier to see with the arithmetic in front of you, so it is a script: `python benchmarks/cost_model.py`. It is a calculation from ASSUMED token counts and list prices read from the providers' pricing pages on 2026-10-14, not a measurement; change the assumptions to match your prompts and measure real usage before quoting a figure.

## The shape of the problem

A production agent typically registers somewhere between a handful and a few dozen tools. The naive design shows every tool description to the LLM on every query, so the judge's input grows linearly with the catalogue. The two-stage design shows it only the shortlist: a cheap, local, deterministic ranking picks the top-K candidates, and the judge decides among those.

Under the assumptions in `benchmarks/cost_model.py` the saving is small for a catalogue of five tools and grows with catalogue size, because the shortlist costs the same however many tools exist:

```
| judge model | tools in catalogue | judge sees all tools (USD / 1,000 routes) | judge sees top-3 (USD / 1,000 routes) | saving |
|---|---|---|---|---|
| claude-haiku-4-5 | 5 | $0.850 | $0.770 | 9% |
| claude-haiku-4-5 | 20 | $1.450 | $0.770 | 47% |
| claude-haiku-4-5 | 50 | $2.650 | $0.770 | 71% |
| gpt-4o-mini | 5 | $0.116 | $0.103 | 10% |
| gpt-4o-mini | 20 | $0.206 | $0.103 | 50% |
| gpt-4o-mini | 50 | $0.386 | $0.103 | 73% |
```

## What the prefilter costs you: recall

The saving is only real if the right tool survives the shortlist. When it does not, the judge never sees it, and the outcome (`no_matching_tool`, or a wrong pick among the candidates) is indistinguishable from a catalogue that lacks the tool. That recall ceiling is measurable without any judge: `benchmarks/run.py` prints the prefilter's recall@1, recall@3 and recall@5 on the labelled queries. With the offline lexical encoder it is recall@1 / @3 / @5 of 72% / 80% / 88% on the 40 test-half queries that should route (see the README for the table and for what a semantic encoder does on the same queries).

## When the two-stage filter does NOT pay off

| Situation | Why |
|---|---|
| **A handful of tools** | The cost model above shows the saving is small, and the prefilter is one more moving part. Send all descriptions to a cheap judge. |
| **Near-identical descriptions** | Embeddings cannot separate "refund an order" from "refund a subscription" reliably; the extra stage adds noise, and recall@K suffers. |
| **A tight latency floor** | A local encoder adds a forward pass per query and a hosted encoder adds a network round trip. Measure both on your hardware (this repo does not publish a latency number it has not measured for your setup). |
| **You already run a frontier model per request** | One long prompt is simpler. The two stages matter most when you want a *small* judge and a prefilter that sets it up for success. |

## Why a judge at all, when the prefilter could "just pick top-1"

Cosine similarity tells you "this query is in the neighbourhood of this tool description". It cannot tell you:

- whether the query *fully* matches the tool's intent or only partly overlaps;
- whether two tools are equally plausible (the **ambiguous_match** refusal);
- whether the query asks for something the agent *shouldn't* do (the **out_of_scope** refusal);
- whether any pick would be a guess (the **low_confidence** path).

The benchmark makes the first point concrete without a model: "always take the prefilter's top-1 tool" is a baseline that never refuses, so on the queries that should be refused it routes every one of them to some tool (a 100% false-route rate). Whether a real LLM judge closes that gap is exactly what this repository could not measure offline: see the README's Limitations.

## Why a policy hook as well

An allow/deny decision should not depend on a similarity score or on a model that can be talked round. `Router(policy=...)` runs a deterministic function over the query *before* any embedding or model call and refuses `out_of_scope` when it fires. The benchmark reports the router with and without the example rules: they catch what they name and nothing else, and the README says so. They reduce risk; they do not replace authorisation on the tools themselves.

## Why structured output, not provider tool-calling

OpenAI tools and Anthropic `tool_use` are convenient, but they:

1. couple the agent to one provider's tool schema and parsing quirks;
2. compose awkwardly with a refusal taxonomy (you would add a synthetic `refuse_with_reason` tool to every catalogue);
3. surface little about why a tool was chosen.

The judge here returns plain JSON validated by Pydantic: portable, easy to test, easy to log. A malformed reply, or a tool outside the shortlist, becomes a `low_confidence` refusal instead of an exception.
