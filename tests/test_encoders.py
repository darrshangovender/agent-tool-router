"""The offline encoders, the encoder factory and the prefilter's guards."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from agent_tool_router.embedding_prefilter import (
    EmbeddingPrefilter,
    HashingEncoder,
    StubEncoder,
    _unit_rows,
    make_encoder,
)


def cos(enc, a: str, b: str) -> float:
    va, vb = enc.encode([a, b])
    return float(va @ vb)


# ---- HashingEncoder -------------------------------------------------------------------------

def test_rows_are_unit_length_float32():
    out = HashingEncoder().encode(["refund my order", "open a support ticket"])
    assert out.dtype == np.float32 and out.shape == (2, 512)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("text", ["", "   ", "the a an of", "?!?!", "—"])
def test_text_with_no_content_words_is_a_zero_vector_not_nan(text):
    out = HashingEncoder().encode([text])
    assert np.all(out == 0.0) and np.all(np.isfinite(out))


def test_case_and_punctuation_do_not_change_the_vector():
    enc = HashingEncoder()
    a, b = enc.encode(["Refund, my ORDER!", "refund my order"])
    assert np.allclose(a, b)


def test_stopwords_are_ignored():
    assert cos(HashingEncoder(), "how do I refund the order", "refund order") == pytest.approx(1.0)


def test_shared_words_score_higher_than_unrelated_text():
    enc = HashingEncoder()
    related = cos(enc, "refund the order", "issue a refund for an order")
    unrelated = cos(enc, "refund the order", "search the knowledge base article")
    assert related > 0.4 > unrelated


def test_it_is_lexical_not_semantic():
    """The documented limit: a paraphrase with no shared word scores near zero."""
    assert abs(cos(HashingEncoder(), "I want my money back", "reimbursement")) < 0.25


def test_word_endings_share_character_trigrams():
    enc = HashingEncoder()
    assert cos(enc, "refunded", "refunding") > cos(enc, "refunded", "scheduling") + 0.2


def test_non_latin_scripts_are_tokenised():
    """The word pattern is Unicode-aware: Arabic and Chinese text is not silently dropped."""
    enc = HashingEncoder()
    assert cos(enc, "استرداد المبلغ", "استرداد المبلغ") == pytest.approx(1.0)
    assert np.any(enc.encode(["退款订单"]) != 0)


def test_batch_matches_single_encoding():
    enc = HashingEncoder()
    texts = ["refund", "open a ticket", "schedule a callback"]
    batch = enc.encode(texts)
    for i, t in enumerate(texts):
        assert np.allclose(batch[i], enc.encode([t])[0])


def test_empty_batch_gives_an_empty_matrix():
    assert HashingEncoder().encode([]).shape == (0, 512)


def test_dimension_is_validated():
    with pytest.raises(ValueError, match="dim"):
        HashingEncoder(dim=4)
    assert HashingEncoder(dim=64).encode(["x ray"]).shape == (1, 64)


def test_vectors_are_identical_across_processes_with_different_hash_seeds():
    """The builtin hash() is salted per process; this encoder must not depend on it."""
    code = ("import sys; sys.path.insert(0, '.');"
            "from agent_tool_router.embedding_prefilter import HashingEncoder;"
            "v = HashingEncoder().encode(['refund my order 9921'])[0];"
            "print(round(float(v[:64] @ v[:64]), 6), int(abs(v).argmax()))")
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                       env={**os.environ, "PYTHONHASHSEED": seed}).stdout
        for seed in ("1", "2", "random")
    }
    assert len(outs) == 1


# ---- StubEncoder / factory ------------------------------------------------------------------

def test_stub_encoder_is_unit_norm_and_keeps_zero_rows_finite():
    out = StubEncoder(dim=32).encode(["refund order", ""])
    assert np.isclose(np.linalg.norm(out[0]), 1.0) and np.all(out[1] == 0)


def test_make_encoder_backends():
    assert isinstance(make_encoder("hashing"), HashingEncoder)
    assert isinstance(make_encoder(), HashingEncoder)
    assert isinstance(make_encoder("stub"), StubEncoder)
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        make_encoder("word2vec")


# ---- _unit_rows -----------------------------------------------------------------------------

def test_unit_rows_normalises_and_leaves_zero_rows_alone():
    out = _unit_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_unit_rows_lifts_a_vector_to_a_row():
    assert _unit_rows(np.array([0.0, 2.0])).shape == (1, 2)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_unit_rows_rejects_non_finite_values(bad):
    with pytest.raises(ValueError, match="NaN or infinite"):
        _unit_rows(np.array([[1.0, bad]]))


# ---- prefilter guards -----------------------------------------------------------------------

class Unnormalised:
    model_name = "unnormalised"

    def encode(self, texts):
        return np.array([[10.0 * (1 + len(t)) if i == 0 else 0.0 for i in range(3)] for t in texts])


def test_unnormalised_encoder_output_still_gives_cosine_scores():
    f = EmbeddingPrefilter(["a", "b"], ["x", "yy"], encoder=Unnormalised())
    assert all(abs(c.score) <= 1.0 + 1e-6 for c in f.top_k("zzz", k=2))


class Broken:
    model_name = "broken"

    def encode(self, texts):
        return np.full((len(texts), 4), np.nan)


def test_a_nan_encoder_fails_loudly_at_construction():
    with pytest.raises(ValueError, match="NaN"):
        EmbeddingPrefilter(["a"], ["x"], encoder=Broken())


def test_top_k_validates_k_and_orders_by_score():
    f = EmbeddingPrefilter(
        ["refund", "ticket", "kb"],
        ["issue refund money back", "open support ticket", "search knowledge base"],
        encoder=HashingEncoder(),
    )
    with pytest.raises(ValueError, match="k must be"):
        f.top_k("refund", k=0)
    scores = [c.score for c in f.top_k("refund money", k=3)]
    assert scores == sorted(scores, reverse=True) and f.top_k("refund money", k=3)[0].name == "refund"


def test_a_query_with_no_content_words_scores_everything_zero():
    f = EmbeddingPrefilter(["a", "b"], ["alpha text", "beta text"], encoder=HashingEncoder())
    assert [c.score for c in f.top_k("???", k=2)] == [0.0, 0.0]
    assert [c.name for c in f.top_k("???", k=2)] == ["a", "b"]        # insertion order


def test_dim_and_model_name_properties():
    f = EmbeddingPrefilter(["a"], ["alpha"], encoder=HashingEncoder(dim=32))
    assert (f.dim, f.model_name) == (32, "hashing-v1")
    assert EmbeddingPrefilter([], []).dim == 0
