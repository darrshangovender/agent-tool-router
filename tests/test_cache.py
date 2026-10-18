"""EmbeddingCache: bounds, recency, normalisation, persistence, corruption, threads."""

from __future__ import annotations

import sqlite3
import threading

import numpy as np
import pytest

from agent_tool_router.cache import EmbeddingCache
from agent_tool_router.embedding_prefilter import EmbeddingPrefilter, HashingEncoder
from tests.helpers import CountingEncoder


def vec(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


# ---- bounds and recency ---------------------------------------------------------------------

def test_least_recently_used_entry_is_evicted_first():
    c = EmbeddingCache(max_entries=2)
    c.put("a", "m", vec(1, 0))
    c.put("b", "m", vec(0, 1))
    c.get("a", "m")                          # a is now the most recent
    c.put("c", "m", vec(1, 1))
    assert c.get("b", "m") is None and c.get("a", "m") is not None and len(c) == 2


def test_overwriting_a_key_does_not_grow_the_cache():
    c = EmbeddingCache(max_entries=2)
    for i in range(5):
        c.put("same", "m", vec(i, 1))
    assert len(c) == 1 and c.get("same", "m")[0] == 4.0


def test_unbounded_cache_keeps_everything():
    c = EmbeddingCache(max_entries=None)
    for i in range(300):
        c.put(f"q{i}", "m", vec(i, 1))
    assert len(c) == 300


@pytest.mark.parametrize("bad", [0, -1])
def test_max_entries_must_be_positive(bad):
    with pytest.raises(ValueError, match="max_entries"):
        EmbeddingCache(max_entries=bad)


# ---- keys -----------------------------------------------------------------------------------

def test_whitespace_is_folded_but_digits_and_case_are_not():
    c = EmbeddingCache()
    c.put("show  order\n#9921", "m", vec(1, 0))
    assert c.get("show order #9921", "m") is not None
    assert c.get("show order #9922", "m") is None
    assert c.get("Show order #9921", "m") is None


def test_a_custom_normaliser_can_fold_templated_queries():
    c = EmbeddingCache(normalize=lambda q: "".join("#" if ch.isdigit() else ch for ch in q))
    c.put("show order 9921", "m", vec(1, 0))
    assert c.get("show order 1234", "m") is not None


def test_the_model_name_is_part_of_the_key():
    c = EmbeddingCache()
    c.put("q", "model-a", vec(1, 0))
    assert c.get("q", "model-b") is None and c.get("q", "model-a") is not None


# ---- values ---------------------------------------------------------------------------------

def test_cached_vectors_are_read_only_copies():
    c = EmbeddingCache()
    original = vec(1, 0)
    c.put("q", "m", original)
    original[0] = 99.0                                    # the caller's array is not the cache's
    got = c.get("q", "m")
    assert got[0] == 1.0 and got.dtype == np.float32
    with pytest.raises(ValueError):
        got[0] = 5.0


def test_stats_and_clear():
    c = EmbeddingCache()
    assert c.stats() == {"hits": 0, "misses": 0, "hit_rate": 0.0, "size": 0}
    c.get("q", "m")
    c.put("q", "m", vec(1, 0))
    c.get("q", "m")
    c.get("q", "m")
    assert c.stats() == {"hits": 2, "misses": 1, "hit_rate": pytest.approx(2 / 3), "size": 1}
    c.clear()
    assert c.stats() == {"hits": 0, "misses": 0, "hit_rate": 0.0, "size": 0}


# ---- prefilter integration ------------------------------------------------------------------

def test_a_cache_hit_skips_the_encoder_and_returns_the_same_ranking():
    enc = CountingEncoder()
    cache = EmbeddingCache()
    f = EmbeddingPrefilter(["a", "b"], ["alpha refund", "beta ticket"], encoder=enc, cache=cache)
    first = f.top_k("refund please", k=2)
    calls = len(enc.encoded)
    second = f.top_k("refund please", k=2)
    assert len(enc.encoded) == calls and first == second and cache.stats()["hits"] == 1


def test_a_cache_filled_by_an_encoder_with_another_width_is_refused_not_misread():
    cache = EmbeddingCache()
    cache.put("refund please", "hashing-v1", vec(1, 0, 0))            # a 3-wide vector
    f = EmbeddingPrefilter(["a"], ["alpha"], encoder=HashingEncoder(), cache=cache)
    with pytest.raises(ValueError, match="dimensions"):
        f.top_k("refund please")


# ---- persistence ----------------------------------------------------------------------------

def test_entries_survive_a_restart(tmp_path):
    db = tmp_path / "sub" / "cache.db"
    c = EmbeddingCache(db)
    c.put("q1", "m", vec(1, 2, 3))
    c.close()
    again = EmbeddingCache(db)
    assert np.allclose(again.get("q1", "m"), [1, 2, 3]) and len(again) == 1
    again.close()


def test_the_bound_applies_to_the_file_as_well(tmp_path):
    db = tmp_path / "c.db"
    c = EmbeddingCache(db, max_entries=3)
    for i in range(10):
        c.put(f"q{i}", "m", vec(i, 1))
    c.close()
    rows = sqlite3.connect(db).execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert rows == 3
    reopened = EmbeddingCache(db, max_entries=3)
    assert [reopened.get(f"q{i}", "m") is not None for i in range(10)] == [False] * 7 + [True] * 3
    reopened.close()


def test_a_smaller_bound_on_reopen_keeps_the_newest(tmp_path):
    db = tmp_path / "c.db"
    c = EmbeddingCache(db, max_entries=None)
    for i in range(6):
        c.put(f"q{i}", "m", vec(i, 1))
    c.close()
    small = EmbeddingCache(db, max_entries=2)
    assert len(small) == 2 and small.get("q5", "m") is not None and small.get("q4", "m") is not None
    small.close()


def test_a_corrupt_row_is_skipped_and_the_rest_load(tmp_path):
    db = tmp_path / "c.db"
    c = EmbeddingCache(db)
    c.put("good", "m", vec(1, 0))
    c.close()
    with sqlite3.connect(db) as raw:
        raw.execute("INSERT INTO embeddings (key, model, vector, dim) VALUES ('x', 'm', ?, 8)",
                    (b"\x00\x01\x02",))
        raw.execute("INSERT INTO embeddings (key, model, vector, dim) VALUES ('y', 'm', ?, 5)",
                    (vec(1, 2).tobytes(),))
    loaded = EmbeddingCache(db)
    assert loaded.get("good", "m") is not None and len(loaded) == 1
    loaded.close()


def test_rewriting_a_key_moves_it_to_the_newest_position_on_disk(tmp_path):
    db = tmp_path / "c.db"
    c = EmbeddingCache(db, max_entries=None)
    c.put("old", "m", vec(1, 0))
    c.put("new", "m", vec(0, 1))
    c.put("old", "m", vec(1, 1))
    c.close()
    small = EmbeddingCache(db, max_entries=1)
    assert small.get("old", "m") is not None and small.get("new", "m") is None
    small.close()


def test_clear_empties_the_file_too(tmp_path):
    db = tmp_path / "c.db"
    c = EmbeddingCache(db)
    c.put("q", "m", vec(1, 0))
    c.clear()
    c.close()
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0


def test_a_closed_cache_still_works_in_memory(tmp_path):
    c = EmbeddingCache(tmp_path / "c.db")
    c.close()
    c.put("q", "m", vec(1, 0))
    c.close()                                                        # closing twice is harmless
    assert c.get("q", "m") is not None


# ---- threads --------------------------------------------------------------------------------

def test_concurrent_use_neither_raises_nor_exceeds_the_bound(tmp_path):
    c = EmbeddingCache(tmp_path / "c.db", max_entries=50)
    errors: list[BaseException] = []

    def work(n: int) -> None:
        try:
            for i in range(120):
                c.put(f"q{n}-{i % 70}", "m", vec(n, i))
                c.get(f"q{(n + 1) % 6}-{i % 70}", "m")
        except BaseException as e:  # noqa: BLE001 - reported by the assertion
            errors.append(e)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and len(c) <= 50
    assert c.stats()["hits"] + c.stats()["misses"] == 6 * 120
    c.close()
