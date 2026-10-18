# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Breaking
- The default encoder is now the offline, lexical `HashingEncoder` (it was the `sentence-transformers`
  model, which pulled in torch). Pass `make_encoder("sentence-transformers")` or `"openai"` for
  semantic matching. `StubEncoder` is kept but is no longer the offline default.
- `Router.route` raises `ValueError` for a query over `max_query_chars` (default 4,000).
- `AnthropicBackend` / `OpenAIBackend` take `temperature=` (default `0.0`; `None` leaves it out) and
  `client=`. `JudgeDecision.confidence` must be a JSON number: `true` and `"0.9"` are refused.
- `pyyaml` moved from the runtime dependencies to the `dev` extra (only the benchmark reads YAML).

### Added
- A labelled, SYNTHETIC routing and refusal benchmark (`benchmarks/run.py`, `benchmarks/routing_corpus.yml`):
  135 queries, 80 that should route and 55 that should be refused with a named reason, a dev/test split
  with the stub judge tuned on dev only, a "never refuse" baseline, prefilter recall@k, 95% Wilson intervals.
- `PatternPolicy` and `Router(policy=...)`: deterministic rules that refuse `out_of_scope` before any
  embedding or model call.
- `Router(on_judge_error="refuse")` turns a judge outage into a typed `low_confidence` refusal with
  `RouteDecision.error` set; `RetryingBackend` retries transient failures with backoff (`retry_on=` filters).
- `Tool.args_schema` now reaches the judge (it used to be collected and never shown).
- `EmbeddingCache`: bounded (LRU, `max_entries`), optional SQLite file with the same bound, read-only
  cached vectors, hit/miss `stats()`, a pluggable query normaliser, a lock for threads.
- `LLMJudge(max_tokens=...)`; `benchmarks/cost_model.py` (arithmetic from stated assumptions).
- Offline examples, a CHANGELOG, a Makefile with `demo`, `bench`, `cost`; CI runs lint (a real gate), the
  tests, the demo and the benchmark.

### Fixed
- **A policy rule with two open-ended gaps could hold a worker indefinitely.** The shipped example rule
  took 1.3 s to scan 1,300 characters of `export all ` and never finished on 5,000. Every gap is now
  bounded (`.{0,80}`) and `Router(max_query_chars=...)` caps the input as a second line of defence.
- **A judge reply that quoted the user's text could route to the quoted tool**: the parser took the first
  JSON object. A reply containing two different decisions is now refused as `low_confidence`.
- `{"confidence": true}` was read as 1.0 and `"0.9"` as 0.9; a boolean cleared every threshold.
- A wordy `reasoning` (over 600 characters) discarded an otherwise valid decision; it is now trimmed.
  A blank one is still refused.
- The offline `EchoBackend` read the candidate block from the first marker in the prompt, so a query that
  imitated the block replaced it (the router then refused, but only because the parser caught the unknown
  tool). It reads the last marker. A gap equal to the margin is decisive (0.30 - 0.20 is 0.0999...).
- `AnthropicBackend` read every content block with a `text` attribute; it now reads `text` blocks only, and
  can omit `temperature` (Claude Haiku 5.5 answers 400 to any value other than 1).
- A decision with neither a tool nor a refusal reason (`refusal_reason` omitted) crashed the router's
  invariant check; the model validator now refuses it at parse time.
- Brackets in a query ("refund order [9921]") broke the stub judge's candidate extraction.
- The stub encoder used the salted builtin `hash()`: rankings changed from run to run and a persisted
  cache was meaningless in the next process. It uses blake2b.
- Tied scores came out in reverse insertion order (an ascending sort was reversed); ties keep insertion order.
- The greedy brace-to-brace regex that extracted judge JSON failed on trailing prose containing a brace.
- The embedding cache was unbounded, loaded the whole table at construction and opened a new connection
  per write; `StubEncoder`-era caches could be read against another encoder's vectors (now a width check).
- Lint: `ruff check .` is a required CI step with an explicit rule set (it was `continue-on-error`).

### Removed
- The README's accuracy, latency and cost-per-1k-routes table. None of it was reproducible (the results
  file was not committed, the API rows needed keys, the dollar figures were constants in the harness).

## [0.1.0] - 2026-10-09

### Added
- Two-stage tool router: embedding prefilter, LLM judge with Pydantic-validated output, four typed
  refusal reasons, Anthropic and OpenAI backends, `EchoBackend` stub, in-memory and SQLite embedding cache.
