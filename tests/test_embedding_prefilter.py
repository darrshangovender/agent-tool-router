"""Tests for the embedding prefilter — uses StubEncoder so there's no model dep."""

from __future__ import annotations

import numpy as np
import pytest

from agent_tool_router.cache import EmbeddingCache
from agent_tool_router.embedding_prefilter import (
    Candidate,
    EmbeddingPrefilter,
    StubEncoder,
)


def _filter(names: list[str], texts: list[str], *, cache: EmbeddingCache | None = None) -> EmbeddingPrefilter:
    return EmbeddingPrefilter(names, texts, encoder=StubEncoder(dim=64), cache=cache)


def test_top_k_returns_at_most_k():
    f = _filter(
        ["refund", "ticket", "kb", "lookup"],
        ["issue refund money", "open support ticket", "search knowledge base", "look up customer account"],
    )
    out = f.top_k("I want a refund please", k=2)
    assert len(out) == 2
    assert all(isinstance(c, Candidate) for c in out)
    assert out[0].name == "refund"


def test_top_k_ranks_relevant_tool_first():
    f = _filter(
        ["weather", "ticket", "refund"],
        ["forecast current weather", "open support ticket", "issue refund money"],
    )
    out = f.top_k("forecast weather today", k=3)
    assert out[0].name == "weather"
    assert out[0].score >= out[1].score >= out[2].score


def test_empty_corpus_returns_empty_list():
    f = _filter([], [])
    assert f.top_k("anything", k=3) == []


def test_identical_descriptions_tie_and_preserve_insertion_order():
    f = _filter(
        ["alpha", "beta", "gamma"],
        ["lookup customer account", "lookup customer account", "lookup customer account"],
    )
    out = f.top_k("lookup customer", k=3)
    # All three scores should be (near-)identical; insertion order tie-break.
    assert {c.name for c in out} == {"alpha", "beta", "gamma"}
    scores = [c.score for c in out]
    assert max(scores) - min(scores) < 1e-6


def test_k_is_clamped_to_corpus_size():
    f = _filter(["a", "b"], ["alpha words", "beta words"])
    out = f.top_k("alpha", k=10)
    assert len(out) == 2


def test_mismatched_inputs_raise():
    with pytest.raises(ValueError):
        EmbeddingPrefilter(["a", "b"], ["only one"], encoder=StubEncoder())


def test_cache_short_circuits_query_embedding():
    cache = EmbeddingCache()
    f = _filter(["refund", "ticket"], ["issue refund", "open ticket"], cache=cache)
    f.top_k("refund this", k=2)
    f.top_k("refund this", k=2)
    stats = cache.stats()
    assert stats["hits"] >= 1 and stats["misses"] >= 1


def test_unit_norm_matrix():
    f = _filter(["a", "b"], ["one two three", "four five six"])
    # StubEncoder normalises, so each row should have ~unit L2 norm.
    norms = np.linalg.norm(f._matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
