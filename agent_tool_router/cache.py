"""Bounded in-memory (+ optional SQLite) cache for query embeddings.

The tool corpus is embedded once at Router construction and held in memory, which is free at
routing time. What repeats under traffic is the query-side embedding, and this cache short-circuits
the encoder call for a query it has seen.

  * Keys are `sha256(model::normalised-query)`. The default normaliser only collapses whitespace,
    so "show order #9921" and "show order #9922" are different keys. Pass `normalize=` to fold
    templated queries (for example mask digits) if your traffic allows it.
  * Memory is bounded: at most `max_entries` vectors are held, least recently used first out.
  * Persistence to SQLite is optional (`path`). The file keeps the same bound, and a corrupt row is
    skipped on load instead of failing construction.
  * Cached vectors are read-only arrays: a caller cannot corrupt the cache by mutating one.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import numpy as np


def _default_normalize(query: str) -> str:
    return " ".join(query.split())


def _key(query: str, model: str) -> str:
    h = hashlib.sha256(f"{model}::{query}".encode()).hexdigest()
    return h[:32]


class EmbeddingCache:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_entries: int | None = 10_000,
        normalize: Callable[[str], str] | None = None,
    ) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._mem: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.RLock()
        self.max_entries = max_entries
        self._normalize = normalize or _default_normalize
        self.hits = 0
        self.misses = 0
        self.path: Path | None = Path(path) if path else None
        self._conn: sqlite3.Connection | None = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            self._init_db()
            self._warm_from_disk()

    # ---- public API ----------------------------------------------------

    def get(self, query: str, model: str) -> np.ndarray | None:
        k = _key(self._normalize(query), model)
        with self._lock:
            vec = self._mem.get(k)
            if vec is not None:
                self._mem.move_to_end(k)
                self.hits += 1
                return vec
            self.misses += 1
            return None

    def put(self, query: str, model: str, vector: np.ndarray) -> None:
        k = _key(self._normalize(query), model)
        vec = np.array(vector, dtype=np.float32, copy=True)
        vec.setflags(write=False)
        with self._lock:
            self._mem[k] = vec
            self._mem.move_to_end(k)
            evicted = []
            while self.max_entries is not None and len(self._mem) > self.max_entries:
                evicted.append(self._mem.popitem(last=False)[0])
            if self._conn is not None:
                self._persist(k, model, vec, evicted)

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            total = self.hits + self.misses
            rate = self.hits / total if total else 0.0
            return {"hits": self.hits, "misses": self.misses, "hit_rate": rate,
                    "size": len(self._mem)}

    def clear(self) -> None:
        with self._lock:
            self._mem.clear()
            self.hits = 0
            self.misses = 0
            if self._conn is not None:
                with self._conn:
                    self._conn.execute("DELETE FROM embeddings")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __len__(self) -> int:
        with self._lock:
            return len(self._mem)

    # ---- SQLite plumbing ----------------------------------------------

    def _init_db(self) -> None:
        assert self._conn is not None
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " key TEXT NOT NULL UNIQUE,"
                " model TEXT NOT NULL,"
                " vector BLOB NOT NULL,"
                " dim INTEGER NOT NULL"
                ")"
            )

    def _warm_from_disk(self) -> None:
        assert self._conn is not None
        limit = "" if self.max_entries is None else f" LIMIT {int(self.max_entries)}"
        rows = self._conn.execute(
            "SELECT key, vector, dim FROM (SELECT id, key, vector, dim FROM embeddings "
            f"ORDER BY id DESC{limit}) ORDER BY id"
        ).fetchall()
        for k, blob, dim in rows:
            try:
                vec = np.frombuffer(blob, dtype=np.float32).reshape(dim).copy()
            except ValueError:
                continue                     # a truncated or corrupt row: skip it, keep the rest
            vec.setflags(write=False)
            self._mem[k] = vec

    def _persist(self, key: str, model: str, vector: np.ndarray, evicted: list[str]) -> None:
        assert self._conn is not None
        with self._conn:
            # REPLACE would keep the row's old position; delete first so recency is by write order.
            self._conn.execute("DELETE FROM embeddings WHERE key = ?", (key,))
            self._conn.execute(
                "INSERT INTO embeddings (key, model, vector, dim) VALUES (?, ?, ?, ?)",
                (key, model, vector.tobytes(), vector.shape[0]),
            )
            for k in evicted:
                self._conn.execute("DELETE FROM embeddings WHERE key = ?", (k,))
