"""Routing and refusal accuracy on a bundled, labelled, SYNTHETIC query set.

    python benchmarks/run.py                       # offline: HashingEncoder + EchoBackend stub judge
    python benchmarks/run.py --encoder st          # MiniLM (needs sentence-transformers)
    python benchmarks/run.py --backend anthropic --judge-model claude-haiku-4-5   # needs a key

Data: `benchmarks/routing_corpus.yml`, 135 hand-written queries against the 8-tool catalogue in
`examples/customer_support_bot.py`: 80 that should route (10 per tool) and 55 that should be refused
(no_matching_tool 20, out_of_scope 15, ambiguous_match 12, low_confidence 8). Labels are the author's
judgement. Offline, the "judge" is `EchoBackend`, a score-threshold heuristic and NOT a language
model, so the numbers describe the prefilter plus that heuristic, not an LLM's routing quality.

Dev / test: the cases are split by position (even = dev, odd = test). The EchoBackend's two numbers
(`low_floor`, `decisive_margin`) are chosen on the DEV half only (`--tune`), and every table below is
reported on the TEST half, so the parameters were not fitted to the cases they are scored on. Both
halves are small (68 and 67 cases): treat differences of a few points as noise.

Reported: routing accuracy, wrong-tool and over-refusal rates on queries that should route; refusal
recall, false-route rate and reason accuracy on queries that should be refused; prefilter recall@k
(independent of the judge); and the "always take the top-1 tool" baseline that never refuses.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_tool_router import PatternPolicy, Router  # noqa: E402
from agent_tool_router.embedding_prefilter import make_encoder  # noqa: E402
from agent_tool_router.llm_judge import (  # noqa: E402
    AnthropicBackend,
    EchoBackend,
    LLMJudge,
    OpenAIBackend,
)
from examples.customer_support_bot import POLICY_RULES, TOOLS  # noqa: E402

CORPUS = Path(__file__).parent / "routing_corpus.yml"
REASONS = ("no_matching_tool", "out_of_scope", "ambiguous_match", "low_confidence")


def load_cases(path: Path = CORPUS) -> list[dict]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids")
    for c in cases:
        if ("expected_tool" in c) == ("expected_refusal" in c):
            raise ValueError(f"{c['id']}: exactly one of expected_tool / expected_refusal")
    return cases


@dataclass
class Results:
    label: str
    n_route: int = 0
    n_refuse: int = 0
    correct: int = 0                      # should route, routed to the right tool
    wrong_tool: int = 0                   # should route, routed to another tool
    over_refused: int = 0                 # should route, refused
    refused: int = 0                      # should refuse, refused (any reason)
    right_reason: int = 0                 # ... with the expected reason
    false_routes: int = 0                 # should refuse, routed
    by_reason: dict[str, dict[str, int]] = field(default_factory=dict)
    top1_hits: int = 0                    # prefilter's first choice is the expected tool
    recall_at: dict[int, int] = field(default_factory=dict)
    median_latency_us: float = 0.0
    failures: list[str] = field(default_factory=list)
    params: tuple[float, float] | None = None

    # ---- derived ----
    @property
    def routing_accuracy(self) -> float:
        return self.correct / self.n_route

    @property
    def wrong_tool_rate(self) -> float:
        return self.wrong_tool / self.n_route

    @property
    def over_refusal_rate(self) -> float:
        return self.over_refused / self.n_route

    @property
    def refusal_recall(self) -> float:
        return self.refused / self.n_refuse

    @property
    def false_route_rate(self) -> float:
        return self.false_routes / self.n_refuse

    @property
    def reason_accuracy(self) -> float:
        return self.right_reason / self.n_refuse

    @property
    def right_decision(self) -> float:
        """Share of ALL cases where route-vs-refuse was right (a correct tool, or any refusal when
        refusal was expected). A baseline that never refuses can only score the routing share."""
        return (self.correct + self.refused) / (self.n_route + self.n_refuse)

    @property
    def exactly_right(self) -> float:
        """Share of all cases with the exact expected outcome (the tool, or the refusal reason)."""
        return (self.correct + self.right_reason) / (self.n_route + self.n_refuse)

    @property
    def refusal_precision(self) -> float:
        refusals = self.refused + self.over_refused
        return self.refused / refusals if refusals else 0.0


def evaluate(router: Router, cases: list[dict], label: str = "") -> Results:
    r = Results(label=label)
    r.by_reason = {k: {"n": 0, "refused": 0, "right_reason": 0} for k in REASONS}
    r.recall_at = {1: 0, 3: 0, 5: 0}
    times: list[float] = []
    for case in cases:
        t0 = time.perf_counter()
        d = router.route(case["query"])
        times.append((time.perf_counter() - t0) * 1e6)
        expected_tool = case.get("expected_tool")
        if expected_tool:
            r.n_route += 1
            ranked = [c.name for c in router.prefilter.top_k(case["query"], k=max(r.recall_at))]
            for k in r.recall_at:
                r.recall_at[k] += expected_tool in ranked[:k]
            r.top1_hits += ranked[0] == expected_tool
            if d.is_route and d.tool == expected_tool:
                r.correct += 1
            elif d.is_route:
                r.wrong_tool += 1
                r.failures.append(f"{case['id']} routed to {d.tool}, expected {expected_tool}")
            else:
                r.over_refused += 1
                r.failures.append(f"{case['id']} refused ({d.refusal.reason.value}), "
                                  f"expected {expected_tool}")
        else:
            reason = case["expected_refusal"]
            r.n_refuse += 1
            r.by_reason[reason]["n"] += 1
            if d.is_refusal:
                r.refused += 1
                r.by_reason[reason]["refused"] += 1
                if d.refusal.reason.value == reason:
                    r.right_reason += 1
                    r.by_reason[reason]["right_reason"] += 1
            else:
                r.false_routes += 1
                r.failures.append(f"{case['id']} routed to {d.tool}, expected refusal ({reason})")
    r.median_latency_us = statistics.median(times)
    return r


class MemoEncoder:
    """Encode each distinct text once: the parameter sweep re-scores the same queries many times."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.model_name = inner.model_name
        self._memo: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._memo]
        if missing:
            for t, v in zip(missing, self._inner.encode(missing), strict=True):
                self._memo[t] = v
        return np.stack([self._memo[t] for t in texts])


_ENCODERS: dict[str, MemoEncoder] = {}


def get_encoder(name: str) -> MemoEncoder:
    if name not in _ENCODERS:
        _ENCODERS[name] = MemoEncoder(make_encoder(name))
    return _ENCODERS[name]


def build_judge(backend: str, judge_model: str, low_floor: float, margin: float) -> LLMJudge:
    if backend == "echo":
        return LLMJudge(EchoBackend(decisive_margin=margin, low_floor=low_floor))
    if backend == "anthropic":
        return LLMJudge(AnthropicBackend(model=judge_model))
    if backend == "openai":
        return LLMJudge(OpenAIBackend(model=judge_model))
    raise ValueError(f"Unknown judge backend: {backend}")


def build_router(encoder_name="hashing", backend="echo", judge_model="claude-haiku-4-5", top_k=3,
                 threshold=0.5, policy=False, low_floor=None, margin=None) -> Router:
    echo_defaults = EchoBackend()
    low_floor = echo_defaults.low_floor if low_floor is None else low_floor
    margin = echo_defaults.decisive_margin if margin is None else margin
    return Router(
        TOOLS,
        encoder=get_encoder(encoder_name),
        judge=build_judge(backend, judge_model, low_floor, margin),
        top_k=top_k,
        confidence_threshold=threshold,
        policy=PatternPolicy(POLICY_RULES) if policy else None,
    )


def pct(x: float) -> str:
    return f"{x:.0%}"


def summary_table(rows: list[Results]) -> str:
    n_r, n_f = rows[0].n_route, rows[0].n_refuse
    head = (f"| configuration | routing accuracy (of {n_r}) | wrong tool | over-refusal "
            f"| refusal recall (of {n_f}) | false route | refusal reason correct | refusal precision "
            f"| right route-or-refuse | exactly right |")
    lines = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r.label} | {pct(r.routing_accuracy)} ({r.correct}) | {pct(r.wrong_tool_rate)} "
            f"({r.wrong_tool}) | {pct(r.over_refusal_rate)} ({r.over_refused}) "
            f"| {pct(r.refusal_recall)} ({r.refused}) | {pct(r.false_route_rate)} "
            f"({r.false_routes}) | {pct(r.reason_accuracy)} ({r.right_reason}) "
            f"| {pct(r.refusal_precision)} | {pct(r.right_decision)} | {pct(r.exactly_right)} |"
        )
    return "\n".join(lines)


def reason_table(rows: list[Results]) -> str:
    lines = ["| configuration | " + " | ".join(
        f"{k} (refused / right reason, of n)" for k in REASONS) + " |",
        "|---|" + "---|" * len(REASONS)]
    for r in rows:
        cells = [f"{v['refused']} / {v['right_reason']} of {v['n']}"
                 for v in (r.by_reason[k] for k in REASONS)]
        lines.append(f"| {r.label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials (n=0 gives the vacuous [0, 1])."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def with_ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {pct(k / n)} [{lo:.0%}-{hi:.0%}]"


def interval_table(plain: Results, guarded: Results) -> str:
    """Every headline rate with its 95% Wilson interval: with 40 and 27 cases they are wide."""
    rows = [
        ("routing accuracy", "correct", "n_route"),
        ("wrong tool", "wrong_tool", "n_route"),
        ("over-refusal", "over_refused", "n_route"),
        ("refusal recall", "refused", "n_refuse"),
        ("false route", "false_routes", "n_refuse"),
        ("refusal reason correct", "right_reason", "n_refuse"),
    ]
    lines = ["| rate | router, no policy | router + policy rules |", "|---|---|---|"]
    for name, num, den in rows:
        cells = [with_ci(getattr(r, num), getattr(r, den)) for r in (plain, guarded)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def prefilter_table(r: Results) -> str:
    n = r.n_route
    return "\n".join([
        f"| prefilter, queries that should route (n={n}) | recall@1 | recall@3 | recall@5 |",
        "|---|---|---|---|",
        f"| {r.label} | " + " | ".join(with_ci(r.recall_at[k], n) for k in (1, 3, 5)) + " |",
    ])


def baseline(r: Results) -> Results:
    """'Always take the prefilter's top-1 tool; never refuse.'"""
    b = Results(label="top-1 baseline (never refuses)", n_route=r.n_route, n_refuse=r.n_refuse)
    b.correct, b.wrong_tool = r.top1_hits, r.n_route - r.top1_hits
    b.false_routes = r.n_refuse
    b.by_reason = {k: {"n": v["n"], "refused": 0, "right_reason": 0} for k, v in r.by_reason.items()}
    return b


LOW_FLOORS = tuple(round(0.05 * i, 2) for i in range(1, 9))     # 0.05 .. 0.40
MARGINS = tuple(round(0.05 * i, 2) for i in range(0, 7))        # 0.00 .. 0.30


def split_cases(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """(dev, test): even positions are dev, odd positions are test. Each tool and each refusal
    reason appears in both halves (10 per tool and 8 to 20 per reason, interleaved)."""
    return cases[0::2], cases[1::2]


def objective(r: Results) -> float:
    """What tuning maximises: the mean of routing accuracy and refusal recall."""
    return (r.routing_accuracy + r.refusal_recall) / 2


def sweep(encoder_name: str, cases: list[dict]) -> list[Results]:
    rows = []
    for low_floor in LOW_FLOORS:
        for margin in MARGINS:
            router = build_router(encoder_name, low_floor=low_floor, margin=margin)
            rows.append(evaluate(router, cases, f"low_floor {low_floor:.2f}, margin {margin:.2f}"))
            rows[-1].params = (low_floor, margin)
    return rows


def tune(encoder_name: str, dev: list[dict]) -> tuple[float, float]:
    """The (low_floor, margin) that maximises `objective` on the dev half. Ties go to the smaller
    low_floor, then the smaller margin (the grid is walked in that order and only a strictly better
    score replaces the incumbent)."""
    best, best_score = (LOW_FLOORS[0], MARGINS[0]), -1.0
    for r in sweep(encoder_name, dev):
        if objective(r) > best_score + 1e-12:
            best, best_score = r.params, objective(r)
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", choices=["hashing", "st", "openai"], default="hashing")
    ap.add_argument("--backend", choices=["echo", "anthropic", "openai"], default="echo")
    ap.add_argument("--judge-model", default="claude-haiku-4-5")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tune", action="store_true",
                    help="choose the Echo judge's parameters on the dev half for this encoder")
    ap.add_argument("--sweep", action="store_true", help="also print the dev-half sweep")
    ap.add_argument("--all", action="store_true", help="score all 135 cases instead of the test half")
    ap.add_argument("--json", type=Path, help="write the results as JSON")
    ap.add_argument("--list-failures", action="store_true")
    args = ap.parse_args(argv)

    cases = load_cases()
    dev, test = split_cases(cases)
    scored = cases if args.all else test
    low_floor = margin = None
    if args.tune and args.backend == "echo":
        low_floor, margin = tune(args.encoder, dev)
    kw = dict(encoder_name=args.encoder, backend=args.backend, judge_model=args.judge_model,
              top_k=args.top_k, threshold=args.threshold, low_floor=low_floor, margin=margin)
    plain = evaluate(build_router(**kw, policy=False), scored, "router, no policy")
    guarded = evaluate(build_router(**kw, policy=True), scored, "router + policy rules")
    base = baseline(plain)
    rows = [base, plain, guarded]

    which = "all" if args.all else f"{len(test)} test"
    echo = build_router(**kw).judge.backend
    params = (f", low_floor={echo.low_floor}, decisive_margin={echo.decisive_margin}"
              if isinstance(echo, EchoBackend) else "")
    print(f"{which} synthetic cases ({plain.n_route} should route, {plain.n_refuse} should be "
          f"refused); encoder={args.encoder}; judge={args.backend}"
          f"{'' if args.backend != 'echo' else ' (EchoBackend: a score heuristic, not an LLM' + params + ')'}\n")
    print(summary_table(rows))
    print()
    print(reason_table([plain, guarded]))
    print()
    print(interval_table(plain, guarded))
    print()
    print(prefilter_table(plain))
    if args.sweep:
        print("\nEchoBackend parameter sweep on the DEV half (no policy); best by "
              "(routing accuracy + refusal recall) / 2:")
        print(summary_table(sweep(args.encoder, dev)))
    if args.list_failures:
        print("\nfailures (no policy):")
        for line in plain.failures:
            print("  " + line)
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in rows], indent=2, default=str),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
