"""Router: construction, input limits, threshold edges, failure modes, determinism, threads."""

from __future__ import annotations

import threading

import pytest

from agent_tool_router import EmbeddingCache, RefusalReason, Router, Tool
from agent_tool_router.llm_judge import LLMJudge
from tests.helpers import CATALOGUE, CountingEncoder, ScriptedBackend, make_router, verdict

# ---- Tool -----------------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [{"name": "", "description": "d"}, {"name": "n", "description": ""}])
def test_a_tool_needs_a_name_and_a_description(kw):
    with pytest.raises(ValueError):
        Tool(**kw)


def test_embed_text_joins_name_description_and_tags():
    assert Tool(name="a", description="b").embed_text() == "a — b"
    assert Tool(name="a", description="b", tags=["x", "y"]).embed_text() == "a — b — x y"


# ---- construction ---------------------------------------------------------------------------

@pytest.mark.parametrize("top_k", [0, -1, 2.5, "3", True, None])
def test_top_k_must_be_a_positive_integer(top_k):
    with pytest.raises(ValueError, match="top_k"):
        Router(CATALOGUE, top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("t", [-0.01, 1.01, "0.5", None, float("nan")])
def test_threshold_must_be_a_number_between_0_and_1(t):
    with pytest.raises(ValueError, match="confidence_threshold"):
        Router(CATALOGUE, confidence_threshold=t)  # type: ignore[arg-type]


@pytest.mark.parametrize("t", [0, 1, 0.0, 1.0, 0.5])
def test_threshold_endpoints_are_allowed(t):
    assert Router(CATALOGUE, confidence_threshold=t).confidence_threshold == t


def test_other_settings_are_validated():
    with pytest.raises(ValueError, match="on_judge_error"):
        Router(CATALOGUE, on_judge_error="ignore")  # type: ignore[arg-type]
    for bad in (0, -5, 1.5, True, "10"):
        with pytest.raises(ValueError, match="max_query_chars"):
            Router(CATALOGUE, max_query_chars=bad)  # type: ignore[arg-type]


def test_catalogue_must_be_non_empty_with_unique_names():
    with pytest.raises(ValueError, match="at least one tool"):
        Router([])
    with pytest.raises(ValueError, match="unique"):
        Router([CATALOGUE[0], CATALOGUE[0]])


def test_tools_can_be_any_iterable_and_len_counts_them():
    assert len(Router(iter(CATALOGUE))) == 4 and set(Router(tuple(CATALOGUE)).tools) == {
        t.name for t in CATALOGUE}


def test_the_default_router_is_fully_offline():
    d = Router(CATALOGUE).route("please refund order 1234")
    assert d.latency_ms >= 0 and (d.is_route or d.is_refusal)


# ---- query limits ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", ["", "   ", "\n\t", None, 5, b"refund"])
def test_blank_or_non_text_queries_are_rejected(q):
    with pytest.raises(ValueError, match="non-empty string"):
        make_router(ScriptedBackend()).route(q)  # type: ignore[arg-type]


def test_the_length_limit_is_inclusive_and_the_error_says_what_to_change():
    router = make_router(ScriptedBackend(), max_query_chars=50)
    assert router.route("refund " * 7 + "x").tool == "issue_refund"           # 50 characters
    with pytest.raises(ValueError, match=r"51 characters.*max_query_chars"):
        router.route("refund " * 7 + "xy")


def test_a_long_query_is_refused_before_any_work_is_done():
    enc, backend = CountingEncoder(), ScriptedBackend()
    router = make_router(backend, encoder=enc, policy=lambda q: pytest.fail("policy ran"))
    before = len(enc.encoded)
    with pytest.raises(ValueError):
        router.route("x" * 5000)
    assert len(enc.encoded) == before and backend.calls == []


@pytest.mark.parametrize("q", [
    "\U0001f600\U0001f680 refund", "استرداد order", "refund\x00order",
    "‮esrever refund", "a" * 4000,
])
def test_odd_but_legal_text_never_crashes(q):
    d = make_router(ScriptedBackend()).route(q)
    assert d.is_route != d.is_refusal


# ---- decisions ------------------------------------------------------------------------------

def test_exactly_one_of_tool_and_refusal_is_set_on_every_path():
    cases = [verdict("issue_refund"), verdict(None, 0.1, "no_matching_tool"),
             verdict("issue_refund", 0.1), "garbage", verdict("kb_search")]
    for reply in cases:
        d = make_router(ScriptedBackend(reply)).route("refund order 9921")
        assert (d.tool is None) != (d.refusal is None) and d.is_route != d.is_refusal


def test_a_route_carries_confidence_reasoning_candidates_and_no_error():
    d = make_router(ScriptedBackend(verdict("issue_refund", 0.77, reasoning="clear"))).route(
        "refund order 9921")
    assert (d.tool, d.confidence, d.reasoning, d.error) == ("issue_refund", 0.77, "clear", None)
    assert len(d.candidates) == 3 and d.candidates[0].score >= d.candidates[1].score


def test_confidence_equal_to_the_threshold_routes_and_just_below_refuses():
    at = make_router(ScriptedBackend(verdict("issue_refund", 0.6)), confidence_threshold=0.6)
    assert at.route("refund order 9921").is_route
    below = make_router(ScriptedBackend(verdict("issue_refund", 0.5999)), confidence_threshold=0.6)
    d = below.route("refund order 9921")
    assert d.refusal.reason is RefusalReason.LOW_CONFIDENCE and d.refusal.judge_confidence == 0.5999
    assert "0.60" in d.refusal.message and d.tool is None


def test_threshold_zero_accepts_anything_and_one_accepts_only_certainty():
    assert make_router(ScriptedBackend(verdict("issue_refund", 0.0)),
                       confidence_threshold=0.0).route("refund order").is_route
    assert make_router(ScriptedBackend(verdict("issue_refund", 0.99)),
                       confidence_threshold=1.0).route("refund order").is_refusal
    assert make_router(ScriptedBackend(verdict("issue_refund", 1.0)),
                       confidence_threshold=1.0).route("refund order").is_route


def test_a_judge_refusal_keeps_the_judges_reason_and_lists_the_shortlist():
    d = make_router(ScriptedBackend(verdict(None, 0.3, "ambiguous_match", "two fit"))).route(
        "look it up")
    assert d.refusal.reason is RefusalReason.AMBIGUOUS_MATCH and d.refusal.message == "two fit"
    assert d.refusal.candidates == [c.name for c in d.candidates] and d.confidence == 0.3


def test_a_tool_outside_the_shortlist_is_not_routed():
    d = make_router(ScriptedBackend(verdict("schedule_call")), top_k=1).route("refund order 9921")
    assert d.is_refusal and d.refusal.reason is RefusalReason.LOW_CONFIDENCE


def test_top_k_larger_than_the_catalogue_is_clamped():
    assert len(make_router(ScriptedBackend(), top_k=50).route("refund order").candidates) == 4


def test_the_judge_sees_only_top_k_candidates():
    backend = ScriptedBackend()
    make_router(backend, top_k=2).route("refund order 9921")
    assert backend.calls[0]["user"].count('"name"') == 2


def test_the_judge_is_told_about_a_tools_arguments_only_when_it_has_some():
    backend = ScriptedBackend()
    make_router(backend, top_k=4).route("refund order 9921")
    user = backend.calls[0]["user"]
    assert user.count('"args"') == 1 and '"order_id": "string"' in user


# ---- judge failures -------------------------------------------------------------------------

def test_a_failing_judge_raises_by_default():
    with pytest.raises(ConnectionError, match="down"):
        make_router(ScriptedBackend(ConnectionError("down"))).route("refund order")


def test_a_failing_judge_can_degrade_to_a_refusal_that_says_it_was_an_outage():
    d = make_router(ScriptedBackend(ConnectionError("x" * 900)), on_judge_error="refuse").route(
        "refund order")
    assert d.is_refusal and d.refusal.reason is RefusalReason.LOW_CONFIDENCE
    assert d.error.startswith("ConnectionError: x") and len(d.error) == 300
    assert "ConnectionError" in d.refusal.message and "x" * 50 not in d.refusal.message
    assert d.refusal.candidates and d.confidence == 0.0


def test_a_normal_low_confidence_refusal_has_no_error_set():
    d = make_router(ScriptedBackend("garbage"), on_judge_error="refuse").route("refund order")
    assert d.is_refusal and d.error is None


def test_programming_errors_in_the_judge_are_not_hidden_when_set_to_raise():
    class Bad:
        model = "bad"

        def complete(self, *a, **k):
            return object().missing      # AttributeError

    with pytest.raises(AttributeError):
        make_router(None, judge=LLMJudge(Bad())).route("refund order")


# ---- determinism, cache, threads ------------------------------------------------------------

def test_the_same_query_gets_the_same_decision_every_time():
    router = make_router(None)
    runs = [router.route("please refund order 9921") for _ in range(5)]
    assert len({(r.tool, r.confidence, tuple(c.name for c in r.candidates)) for r in runs}) == 1


def test_the_cache_is_used_and_does_not_change_the_answer():
    enc, cache = CountingEncoder(), EmbeddingCache()
    router = make_router(None, encoder=enc, cache=cache)
    a = router.route("please refund order 9921")
    encoded = len(enc.encoded)
    b = router.route("please refund   order 9921")             # same after whitespace folding
    assert len(enc.encoded) == encoded and cache.stats()["hits"] == 1
    assert (a.tool, a.confidence) == (b.tool, b.confidence)


def test_concurrent_routing_gives_each_thread_the_single_threaded_answer():
    queries = ["refund order 9921", "open a ticket for the crash", "search the knowledge base",
               "book a callback tomorrow", "tell me a joke"]
    router = make_router(None, cache=EmbeddingCache(max_entries=3))
    expected = {q: (r.tool, r.refusal.reason if r.refusal else None)
                for q in queries for r in [router.route(q)]}
    errors: list[str] = []

    def work(n: int) -> None:
        for i in range(60):
            q = queries[(n + i) % len(queries)]
            r = router.route(q)
            got = (r.tool, r.refusal.reason if r.refusal else None)
            if got != expected[q]:
                errors.append(f"{q!r}: {got} != {expected[q]}")

    threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
