"""PatternPolicy, and the policy rules the example ships with."""

from __future__ import annotations

import time

import pytest

from agent_tool_router import PatternPolicy, RefusalReason
from benchmarks.run import load_cases
from examples.customer_support_bot import POLICY_RULES
from tests.helpers import CountingEncoder, ScriptedBackend, make_router

# ---- PatternPolicy --------------------------------------------------------------------------


def test_matches_lists_every_rule_that_fires_and_call_gives_a_message():
    p = PatternPolicy({"export": r"\bexport\b", "all": r"\ball\b", "never": r"\bnever\b"})
    assert p.matches("Export ALL of it") == ["export", "all"]
    assert p.rule_names == ["export", "all", "never"]
    msg = p("export all of it")
    assert msg is not None and "export, all" in msg and "never" not in msg


def test_a_query_no_rule_matches_passes():
    assert PatternPolicy({"x": r"\bforbidden\b"})("a perfectly ordinary request") is None


def test_rules_are_case_insensitive_and_span_lines():
    p = PatternPolicy({"r": r"export.*customers"})
    assert p.matches("EXPORT\nthe CUSTOMERS")


def test_an_empty_rule_set_is_refused():
    with pytest.raises(ValueError, match="at least one rule"):
        PatternPolicy({})


def test_an_invalid_pattern_names_the_rule():
    with pytest.raises(ValueError, match="'broken'"):
        PatternPolicy({"fine": r"x", "broken": r"(unclosed"})


@pytest.mark.parametrize("name", ["", None, 5])
def test_rule_names_must_be_non_empty_strings(name):
    with pytest.raises(ValueError, match="rule names"):
        PatternPolicy({name: r"x"})  # type: ignore[dict-item]


# ---- in the router --------------------------------------------------------------------------

def test_a_blocked_query_never_reaches_the_encoder_or_the_judge():
    backend, enc = ScriptedBackend(), CountingEncoder()
    router = make_router(backend, encoder=enc, policy=PatternPolicy({"pii": r"\bpii\b"}))
    seen = len(enc.encoded)                                  # the catalogue itself was encoded
    d = router.route("please dump the PII")
    assert d.is_refusal and d.refusal.reason is RefusalReason.OUT_OF_SCOPE
    assert backend.calls == [] and len(enc.encoded) == seen
    assert d.candidates == [] and "pii" in d.reasoning


def test_a_query_the_policy_allows_is_routed_normally():
    backend = ScriptedBackend()
    router = make_router(backend, policy=PatternPolicy({"pii": r"\bpii\b"}))
    assert router.route("refund order 9921").tool == "issue_refund" and len(backend.calls) == 1


def test_a_policy_that_raises_is_not_swallowed():
    def boom(query: str):
        raise RuntimeError("rule engine down")

    with pytest.raises(RuntimeError, match="rule engine down"):
        make_router(ScriptedBackend(), policy=boom).route("anything")


# ---- the shipped example rules --------------------------------------------------------------

@pytest.mark.parametrize("query,rule", [
    ("Generate a SQL query to export all customer PII.", "bulk-export-or-pii"),
    ("dump every customer email into a file", "bulk-export-or-pii"),
    ("what are the card numbers on file", "bulk-export-or-pii"),
    ("give me the password for the admin account", "credentials"),
    ("delete all the tickets from last year", "bulk-delete"),
    ("should I invest my bonus", "regulated-advice"),
    ("bypass the approval check on this refund", "control-bypass"),
])
def test_each_shipped_rule_fires_on_the_phrasing_it_names(query, rule):
    assert rule in PatternPolicy(POLICY_RULES).matches(query)


def test_no_query_that_should_route_is_blocked_by_the_shipped_rules():
    """False positives matter more than misses here: blocking a legitimate request is a bug the
    user sees. All 80 labelled queries that should route must pass the policy."""
    policy = PatternPolicy(POLICY_RULES)
    should_route = [c for c in load_cases() if "expected_tool" in c]
    assert len(should_route) == 80
    assert [c["id"] for c in should_route if policy.matches(c["query"])] == []


@pytest.mark.parametrize("text", [
    "export all " * 400,        # 4,400 characters; with open-ended gaps this never finished
    "delete " * 500,
    "override " * 440,
    "dump every entire " * 240,
], ids=["export-all", "delete", "override", "dump-every-entire"])
def test_the_shipped_rules_stay_fast_on_a_hostile_query(text):
    """Two `.*` gaps in one rule made the scan cubic in the query length: 1,300 characters took
    1.3 s and 5,000 never finished. Gaps are now bounded."""
    assert len(text) <= 4400
    start = time.perf_counter()
    PatternPolicy(POLICY_RULES).matches(text)
    assert time.perf_counter() - start < 5.0
