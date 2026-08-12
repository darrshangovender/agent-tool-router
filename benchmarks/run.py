"""Reproducible benchmark for the two-stage router.

Run:
    python benchmarks/run.py
    python benchmarks/run.py --openai-embeddings
    python benchmarks/run.py --backend anthropic --judge-model claude-haiku-4-5

Reports:
    routing accuracy        — % of expected-route cases where the correct tool was chosen
    refusal precision       — of cases the router refused, % that were *expected* to refuse
    refusal recall          — of expected-refusal cases, % the router actually refused
    median latency (ms)     — wall-clock per route()
    median $ / 1000 routes  — using the prices table in llm_judge / hosted-embedding estimates

Writes the full result table to benchmarks/results.json. The README benchmark
section is generated from that file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_tool_router import RefusalReason, Router  # noqa: E402
from agent_tool_router.embedding_prefilter import make_encoder  # noqa: E402
from agent_tool_router.llm_judge import AnthropicBackend, EchoBackend, LLMJudge, OpenAIBackend  # noqa: E402

# Re-import the catalogue from the example so corpus and demo stay in sync.
from examples.customer_support_bot import TOOLS  # noqa: E402


# Rough cost-per-1000-routes estimates ($USD).
# Embeddings: local sentence-transformers is $0; openai text-embedding-3-small is $0.02/1M tokens.
# Judge: depends on backend and prompt size. We approximate from observed token counts.
COST_PER_1K_ROUTES = {
    ("local", "echo"): 0.000,
    ("local", "claude-haiku-4-5"): 0.012,  # ~300 in + 80 out per route
    ("local", "gpt-4o-mini"): 0.001,
    ("openai", "echo"): 0.003,
    ("openai", "claude-haiku-4-5"): 0.015,
    ("openai", "gpt-4o-mini"): 0.004,
}


def build_judge(backend: str, judge_model: str) -> LLMJudge:
    if backend == "echo":
        return LLMJudge(EchoBackend())
    if backend == "anthropic":
        return LLMJudge(AnthropicBackend(model=judge_model))
    if backend == "openai":
        return LLMJudge(OpenAIBackend(model=judge_model))
    raise ValueError(f"Unknown judge backend: {backend}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openai-embeddings", action="store_true", help="Use OpenAI embedding API instead of local model.")
    p.add_argument("--backend", choices=["echo", "anthropic", "openai"], default="echo")
    p.add_argument("--judge-model", default="claude-haiku-4-5")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default="results.json")
    args = p.parse_args()

    corpus_path = Path(__file__).parent / "routing_corpus.yml"
    cases: list[dict] = yaml.safe_load(corpus_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(cases)} cases from {corpus_path.name}")

    encoder_backend = "openai" if args.openai_embeddings else "sentence-transformers"
    encoder = make_encoder(encoder_backend)
    judge = build_judge(args.backend, args.judge_model)
    router = Router(
        TOOLS,
        encoder=encoder,
        judge=judge,
        top_k=args.top_k,
        confidence_threshold=args.threshold,
    )
    print(f"Encoder: {encoder.model_name}    Judge backend: {args.backend} ({args.judge_model})\n")

    route_hits = route_total = 0
    refuse_correct_reason = refuse_correct_any = refuse_total = 0
    routed_when_refusal_expected = 0
    refused_when_route_expected = 0
    latencies: list[int] = []

    for case in cases:
        t0 = time.perf_counter()
        d = router.route(case["query"])
        latencies.append(int((time.perf_counter() - t0) * 1000))

        expected_tool = case.get("expected_tool")
        expected_refusal = case.get("expected_refusal")

        marker = " "
        if expected_tool:
            route_total += 1
            if d.is_route and d.tool == expected_tool:
                route_hits += 1
                marker = "✓"
            elif d.is_refusal:
                refused_when_route_expected += 1
                marker = "✗R"
            else:
                marker = "✗"
        else:
            refuse_total += 1
            if d.is_refusal:
                refuse_correct_any += 1
                if d.refusal.reason.value == expected_refusal:
                    refuse_correct_reason += 1
                    marker = "✓"
                else:
                    marker = "≈"  # refused but wrong reason
            else:
                routed_when_refusal_expected += 1
                marker = "✗→tool"

        outcome = d.tool or (d.refusal.reason.value if d.refusal else "?")
        print(f"  {marker:<5} {case['id']:<5} {outcome:<22} | {case['query'][:70]}")

    refuse_precision = (
        refuse_correct_any / (refuse_correct_any + refused_when_route_expected)
        if (refuse_correct_any + refused_when_route_expected)
        else 0.0
    )
    refuse_recall = refuse_correct_any / refuse_total if refuse_total else 0.0
    refuse_reason_acc = refuse_correct_reason / refuse_total if refuse_total else 0.0

    cost_key = (encoder_backend if args.openai_embeddings else "local",
                args.judge_model if args.backend != "echo" else "echo")
    cost_per_1k = COST_PER_1K_ROUTES.get(cost_key, 0.0)

    summary = {
        "n_cases": len(cases),
        "encoder": encoder.model_name,
        "judge_backend": args.backend,
        "judge_model": args.judge_model if args.backend != "echo" else "echo-stub",
        "top_k": args.top_k,
        "threshold": args.threshold,
        "routing_accuracy": route_hits / route_total if route_total else 0.0,
        "refusal_precision": refuse_precision,
        "refusal_recall": refuse_recall,
        "refusal_reason_accuracy": refuse_reason_acc,
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0,
        "approx_cost_usd_per_1k_routes": cost_per_1k,
    }
    out_path = Path(__file__).parent / args.out
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"  Routing accuracy        : {summary['routing_accuracy']:.1%}  ({route_hits}/{route_total})")
    print(f"  Refusal precision       : {summary['refusal_precision']:.1%}")
    print(f"  Refusal recall          : {summary['refusal_recall']:.1%}")
    print(f"  Refusal reason accuracy : {summary['refusal_reason_accuracy']:.1%}")
    print(f"  Median latency          : {summary['median_latency_ms']} ms")
    print(f"  p95 latency             : {summary['p95_latency_ms']} ms")
    print(f"  Approx $ / 1k routes    : ${summary['approx_cost_usd_per_1k_routes']:.4f}")
    print(f"\n  Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
