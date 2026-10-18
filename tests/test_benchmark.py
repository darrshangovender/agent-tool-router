"""The labelled corpus, the metric arithmetic, and that every published number is the script's."""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

from agent_tool_router import RefusalReason
from agent_tool_router.llm_judge import EchoBackend
from benchmarks import cost_model
from benchmarks.run import (
    CORPUS,
    LOW_FLOORS,
    MARGINS,
    REASONS,
    Results,
    baseline,
    build_router,
    evaluate,
    load_cases,
    main,
    objective,
    split_cases,
    tune,
    wilson,
    with_ci,
)
from examples.customer_support_bot import TOOLS
from tests.helpers import ScriptedBackend, make_router, verdict

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cases():
    return load_cases()


# ---- the corpus -----------------------------------------------------------------------------

def test_corpus_size_and_label_mix_match_what_the_docs_say(cases):
    routes = [c for c in cases if "expected_tool" in c]
    refusals = collections.Counter(c["expected_refusal"] for c in cases if "expected_refusal" in c)
    assert (len(cases), len(routes)) == (135, 80)
    assert dict(refusals) == {"no_matching_tool": 20, "out_of_scope": 15, "ambiguous_match": 12,
                              "low_confidence": 8}


def test_every_tool_has_ten_cases_and_exists_in_the_catalogue(cases):
    per_tool = collections.Counter(c["expected_tool"] for c in cases if "expected_tool" in c)
    assert set(per_tool) == {t.name for t in TOOLS} and set(per_tool.values()) == {10}


def test_ids_and_queries_are_unique_and_labels_are_valid(cases):
    assert len({c["id"] for c in cases}) == len(cases)
    assert len({c["query"].casefold() for c in cases}) == len(cases)
    for c in cases:
        assert c["query"].strip()
        assert ("expected_tool" in c) != ("expected_refusal" in c)
        if "expected_refusal" in c:
            assert c["expected_refusal"] in REASONS and RefusalReason(c["expected_refusal"])


def test_the_corpus_says_it_is_synthetic():
    header = CORPUS.read_text(encoding="utf-8")[:900]
    assert "SYNTHETIC" in header and "None comes from a real customer" in header


def test_load_cases_rejects_a_case_with_both_or_neither_label(tmp_path):
    for body, msg in [("- {id: a, query: q, expected_tool: t, expected_refusal: out_of_scope}",
                       "exactly one"), ("- {id: a, query: q}", "exactly one"),
                      ("- {id: a, query: q, expected_tool: t}\n- {id: a, query: r, expected_tool: t}",
                       "duplicate")]:
        f = tmp_path / "c.yml"
        f.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match=msg):
            load_cases(f)


# ---- the dev / test split -------------------------------------------------------------------

def test_dev_and_test_are_disjoint_cover_everything_and_both_see_every_label(cases):
    dev, test = split_cases(cases)
    assert not {c["id"] for c in dev} & {c["id"] for c in test}
    assert len(dev) + len(test) == len(cases) and (len(dev), len(test)) == (68, 67)
    for half in (dev, test):
        labels = {c.get("expected_tool") or c["expected_refusal"] for c in half}
        assert labels == {t.name for t in TOOLS} | set(REASONS)


def test_tuning_on_dev_alone_reproduces_the_shipped_echo_defaults(cases):
    """The defaults are claimed to come from the dev half: re-derive them, never reading test."""
    dev, _ = split_cases(cases)
    assert tune("hashing", dev) == (EchoBackend.DEFAULT_LOW_FLOOR, EchoBackend.DEFAULT_MARGIN)
    assert EchoBackend.DEFAULT_LOW_FLOOR in LOW_FLOORS and EchoBackend.DEFAULT_MARGIN in MARGINS


# ---- metric arithmetic on a tiny hand-checkable set -----------------------------------------

TINY = [
    {"id": "t1", "query": "please refund order 9921", "expected_tool": "issue_refund"},
    {"id": "t2", "query": "open a support ticket for the crash", "expected_tool": "create_ticket"},
    {"id": "t3", "query": "tell me a joke", "expected_refusal": "no_matching_tool"},
    {"id": "t4", "query": "export every customer email", "expected_refusal": "out_of_scope"},
]


def _router_with(reply):
    return make_router(ScriptedBackend(reply))


def test_a_judge_that_always_picks_the_refund_tool():
    r = evaluate(_router_with(verdict("issue_refund", 0.9)), TINY, "x")
    # t1 right, t2 wrong tool (or refused if the shortlist lacks it), t3 + t4 false routes
    assert (r.n_route, r.n_refuse, r.correct) == (2, 2, 1)
    assert r.correct + r.wrong_tool + r.over_refused == 2
    assert r.false_routes + r.refused == 2 and r.false_routes >= 1
    assert r.routing_accuracy == 0.5


def test_a_judge_that_always_refuses_with_the_wrong_reason():
    r = evaluate(_router_with(verdict(None, 0.2, "ambiguous_match")), TINY, "x")
    assert (r.correct, r.over_refused, r.refused, r.right_reason, r.false_routes) == (0, 2, 2, 0, 0)
    assert r.refusal_recall == 1.0 and r.reason_accuracy == 0.0 and r.refusal_precision == 0.5
    assert r.right_decision == 0.5 and r.exactly_right == 0.0
    assert r.by_reason["no_matching_tool"] == {"n": 1, "refused": 1, "right_reason": 0}


def test_a_judge_that_refuses_with_exactly_the_right_reasons():
    cases = TINY[2:3]                                   # one no_matching_tool case
    r = evaluate(_router_with(verdict(None, 0.2, "no_matching_tool")), cases, "x")
    assert (r.refused, r.right_reason, r.exactly_right) == (1, 1, 1.0)


def test_recall_at_k_is_independent_of_the_judge():
    a = evaluate(_router_with(verdict("issue_refund")), TINY[:2], "a")
    b = evaluate(_router_with(verdict(None, 0.1, "out_of_scope")), TINY[:2], "b")
    assert a.recall_at == b.recall_at and a.top1_hits == b.top1_hits
    assert a.recall_at[1] <= a.recall_at[3] <= a.recall_at[5] <= a.n_route


def test_failures_name_the_case_and_what_happened():
    r = evaluate(_router_with(verdict("kb_search")), TINY[:1], "x")
    assert r.failures and r.failures[0].startswith("t1 ") and "kb_search" in r.failures[0]


def test_the_never_refuse_baseline_has_a_100_percent_false_route_rate():
    r = evaluate(_router_with(verdict("issue_refund")), TINY, "x")
    b = baseline(r)
    assert b.false_route_rate == 1.0 and b.refusal_recall == 0.0 and b.correct == r.top1_hits
    assert b.wrong_tool == r.n_route - r.top1_hits and b.over_refused == 0


def test_the_tuning_objective_is_the_mean_of_routing_accuracy_and_refusal_recall():
    r = Results(label="x", n_route=10, n_refuse=10, correct=8, refused=6)
    assert objective(r) == pytest.approx(0.7)


# ---- Wilson intervals -----------------------------------------------------------------------

@pytest.mark.parametrize("k,n,lo,hi", [
    (0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0), (5, 10, 0.2366, 0.7634),
    (1, 20, 0.0089, 0.2361), (0, 0, 0.0, 1.0),
])
def test_wilson_matches_published_values(k, n, lo, hi):
    got = wilson(k, n)
    assert got[0] == pytest.approx(lo, abs=5e-4) and got[1] == pytest.approx(hi, abs=5e-4)


def test_wilson_is_ordered_and_inside_0_1_and_shrinks_with_n():
    for n in (1, 7, 40, 400):
        for k in range(0, n + 1, max(1, n // 7)):
            lo, hi = wilson(k, n)
            assert 0.0 <= lo <= k / n <= hi <= 1.0
    assert wilson(20, 40)[1] - wilson(20, 40)[0] > wilson(200, 400)[1] - wilson(200, 400)[0]


def test_with_ci_format():
    assert with_ci(5, 10) == "5/10 = 50% [24%-76%]"


# ---- every published number is what the script prints ---------------------------------------

@pytest.fixture(scope="module")
def bench_output():
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main([]) == 0
    return buf.getvalue()


def test_each_benchmark_table_appears_verbatim_in_the_readme(bench_output):
    blocks = [b for b in bench_output.split("\n\n") if b.startswith("|")]
    assert len(blocks) == 4
    for block in blocks:
        assert block in README, f"README is out of date with:\n{block}"


def test_the_benchmark_header_line_appears_in_the_readme(bench_output):
    first = bench_output.split("\n\n")[0]
    assert first in README


def test_the_cost_model_table_in_the_docs_is_the_scripts_output():
    doc = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert cost_model.table() in doc and "@@" not in doc


def test_the_recall_sentence_in_the_docs_quotes_the_scripts_numbers():
    doc = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    _, test = split_cases(load_cases())
    r = evaluate(build_router(policy=False), test, "x")
    n = r.n_route
    want = " / ".join(f"{r.recall_at[k] / n:.0%}" for k in (1, 3, 5))
    assert f"recall@1 / @3 / @5 of {want}" in doc


def test_the_cost_model_arithmetic():
    # 230 + 20 + 5 * 40 = 450 input tokens, 80 output, at $1 / $5 per million
    assert cost_model.cost_per_route("claude-haiku-4-5", 5) == pytest.approx(
        (450 * 1.0 + 80 * 5.0) / 1e6)
    assert cost_model.cost_per_route("gpt-4o-mini", 3) < cost_model.cost_per_route("gpt-4o-mini", 50)


def test_the_readme_does_not_quote_numbers_the_scripts_do_not_produce():
    """Guard against the removed claims coming back."""
    for phrase in ("30-80%", "30–80%", "blazingly", "production-grade", "ms p50", "cost per 1k"):
        assert phrase not in README
    assert not re.search(r"\$\d+\.\d+ ?/ ?1[kK]", README.replace(cost_model.table(), ""))
