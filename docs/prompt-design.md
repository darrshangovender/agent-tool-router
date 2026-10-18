# Prompt design: the judge

The judge prompt has one job: pick a tool from the shortlist or refuse, with a typed reason. It runs once per query and its output is parsed by Pydantic, so any deviation from the schema is treated as a failure rather than silently passed through.

This document explains the choices. It makes no accuracy claims about the prompt: the repository has no way to run a real model offline, so nothing here is backed by a measurement of one. If you change the prompt, run `benchmarks/run.py` with your backend and keep your own numbers.

## The system prompt

The authoritative text is `SYSTEM_PROMPT` in `agent_tool_router/llm_judge.py` (a test checks that every refusal reason and rule below appears in it):

```
You are a tool routing judge.

You receive a user query and a shortlist of candidate tools (name, description, prefilter
similarity score). Your job is to choose AT MOST one tool, or refuse.

Refuse with one of these reasons when appropriate:
  - no_matching_tool: none of the candidates plausibly answers the query.
  - ambiguous_match:  two or more candidates are equally plausible.
  - out_of_scope:     the query asks for policy-blocked or non-routable capability.
  - low_confidence:   you would guess, but the choice is not defensible.

Return STRICT JSON matching this schema:
{ "tool": string | null, "confidence": number in [0, 1],
  "reasoning": string (<= 1 sentence, <= 600 chars),
  "refusal_reason": "no_matching_tool" | "ambiguous_match" | "out_of_scope" | "low_confidence" | null }

Candidates may list the arguments each tool takes ("args"); prefer a tool whose arguments the query
can actually supply.

Rules:
  - If "tool" is null, "refusal_reason" MUST be set.
  - If "tool" is set, "refusal_reason" MUST be null.
  - Do not invent tool names - only pick from the candidate list.
```

## Why each piece is there

**A tight role ("tool routing judge").** A generic assistant framing invites conversational answers; "judge" keeps the model in classification mode.

**The refusal taxonomy, one line each.** Without it a model tends to use "I'm not sure" as its only refusal. The four reasons map to what a human would do next: extend the catalogue (`no_matching_tool`), sharpen descriptions (`ambiguous_match`), apply policy (`out_of_scope`), or review (`low_confidence`).

**The schema inlined in the prompt.** Provider JSON modes help, but inlining the schema lets the same prompt run on any backend. Pydantic validates the result; a reply that fails validation, omits the refusal reason when `tool` is null, names a tool outside the shortlist, or contains no JSON becomes a `low_confidence` refusal, so the caller never sees a parse exception.

**"Do not invent tool names."** A model can invent a plausible tool that does not exist. The parser enforces this as well (a pick outside the shortlist is rejected), so the instruction is belt and braces.

**`reasoning` capped at one sentence.** A short justification is enough to debug a bad route afterwards and cheap enough to log on every request.

**`args` in each candidate block.** `Tool.args_schema` used to be collected and never shown to the judge, although "can the query supply this tool's arguments" is a main reason to have a judge at all. It is now part of the candidate block when a tool declares one.

## The user message

```
USER QUERY:
{the raw query}

CANDIDATE TOOLS (ranked by embedding similarity, highest first):
[
  {"name": "...", "score": 0.81, "description": "...", "args": {"order_id": "string"}},
  ...
]

Return your decision as JSON only.
```

- **The similarity score is shown to the judge** as evidence. (The offline `EchoBackend` stub uses it as its only input.)
- **The query is sent verbatim.** Casing, typos and tone carry signal; the prompt does no cleaning. Note the query is untrusted text inside the prompt: the candidate block is delimited by a fixed header and the stub reads the array after that header only, so brackets in a query cannot be mistaken for the block. A real LLM can still be steered by instructions inside a query; that is what `Router(policy=...)` and authorisation on the tools are for.

## Decoding

The provider backends send `temperature=0.0` by default: routing is classification and variability is noise. Pass `temperature=None` to leave it out: newer models reject sampling parameters (Claude Haiku 5.5 answers 400 to any `temperature` other than 1), and with adaptive thinking on by default, thinking tokens count toward `max_tokens`, so raise `LLMJudge(max_tokens=...)` for such a model or the reply can end before any text. Only `text` blocks are read, so thinking blocks never reach the JSON parser. None of this has been run against a live model here; the tests drive the real SDK client code over a mock transport.

## What was left out, and why

Few-shot examples, a "think step by step" preamble and a confidence calibration table were not added. This repository has no model-in-the-loop measurement of their effect, so it makes no claim in either direction; `confidence_threshold` on the router is the operational knob for self-reported confidence.
