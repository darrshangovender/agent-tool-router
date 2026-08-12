# Prompt design — the judge

The judge prompt has one job: pick a tool from the shortlist or refuse, with a typed reason. It runs once per query and its output is parsed by Pydantic, so any deviation from schema is treated as a failure rather than silently passed through.

## The system prompt

Verbatim from `agent_tool_router/llm_judge.py`:

```
You are a tool routing judge.

You receive a user query and a shortlist of candidate tools
(name, description, prefilter similarity score). Your job is to
choose AT MOST one tool, or refuse.

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
  "refusal_reason": ... | null
}

Rules:
  - If "tool" is null, "refusal_reason" MUST be set.
  - If "tool" is set, "refusal_reason" MUST be null.
  - Do not invent tool names — only pick from the candidate list.
```

## Why each piece is there

**"You are a tool routing judge."** Sets a tight role. Generic "you are a helpful assistant" framings push the model toward conversational answers; the explicit "judge" framing keeps it in classification mode.

**Refusal taxonomy spelled out with one-line definitions.** Without this, models lean hard on "I'm not sure" as the only refusal. The four explicit reasons map to operational decisions a human reviewer would make: extend the catalogue (`no_matching_tool`), tighten the descriptions (`ambiguous_match`), apply policy (`out_of_scope`), or surface for review (`low_confidence`).

**Strict JSON schema in the prompt.** Provider-side JSON modes (OpenAI's `response_format`, Anthropic's tool-use) work, but inlining the schema lets the same prompt run on any backend without conditional code. Pydantic validates the result; anything that fails validation becomes a `low_confidence` refusal so the caller never sees a parse exception.

**"Do not invent tool names — only pick from the candidate list."** The single most common failure in early experiments was the judge inventing a plausible-sounding tool name that didn't exist. The parser also enforces this — a chosen tool not in the shortlist is rejected — but the explicit instruction cuts the failure rate by an order of magnitude in practice.

**`"reasoning"` capped at one sentence / 600 chars.** Long chain-of-thought before the verdict slows the route and rarely changes the answer at this scale (3-5 candidates). A short justification is enough to debug a bad route after the fact and cheap enough to log on every request.

## The user message

```
USER QUERY:
{the raw query}

CANDIDATE TOOLS (ranked by embedding similarity, highest first):
[
  {"name": "...", "score": 0.81, "description": "..."},
  {"name": "...", "score": 0.42, "description": "..."},
  ...
]

Return your decision as JSON only.
```

Two notable choices:

- **The similarity score is shown to the judge.** This is evidence — when the top candidate is 0.81 and the next is 0.30, the judge has a strong prior even before reading the descriptions. When both are 0.55, the prompt itself signals "ambiguous, consider refusing".
- **The query is sent verbatim, with no preprocessing.** Casing, typos, and tone all carry signal. We rely on the encoder's robustness rather than trying to clean the query before routing.

## Temperature and decoding

The judge runs at `temperature=0.0`. Routing is a classification task — variability is noise here, not exploration. The provider backends both set this explicitly.

## What we deliberately did NOT add

- **Few-shot examples.** They roughly double the prompt size for ~2 percentage points of accuracy on the bundled benchmark. Not worth it for a per-request cost.
- **A "think step by step" preamble.** Empirically slows the route by ~30% with no measurable accuracy lift at this prompt size.
- **A confidence calibration table.** Models are not well-calibrated on self-reported confidence anyway; the router's `confidence_threshold` is the operational knob.

If you change the prompt, re-run `benchmarks/run.py` and update `benchmarks/results.json` — both routing accuracy and refusal recall are sensitive to small wording changes.
