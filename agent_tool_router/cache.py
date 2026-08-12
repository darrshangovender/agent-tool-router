"""In-memory + SQLite cache for query embeddings.

The tool corpus is embedded once at Router construction and held in memory —
that's already free at routing time. What gets repeated, especially under
production traffic with templated queries ("show order #X", "refund #Y"), is
the query-side embedding. This cache short-circuits the embedding call for
exact-match query strings.

Persistence to SQLite is optional; without a path, the cache is RAM-only.
Hits and misses are tracked so benchmarks can report cache hit rate.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np


def _key(query: str, model: str) -> str:
    h = hashlib.sha256(f"{model}::{query}".encode("utf-8")).hexdigest()
    return h[:32]


class EmbeddingCache:
    def __init__(self, path: str | Path | None = None) -> None:
        self._mem: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.path: Path | None = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._warm_from_disk()

    # ---- public API ----------------------------------------------------

    def get(self, query: str, model: str) -> np.ndarray | None:
        k = _key(query, model)
        with self._lock:
            vec = self._mem.get(k)
            if vec is not None:
                self.hits += 1
                return vec
            self.misses += 1
            return None

    def put(self, query: str, model: str, vector: np.ndarray) -> None:
        k = _key(query, model)
        with self._lock:
            self._mem[k] = vector
        if self.path:
            self._persist(k, model, vector)

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        rate = self.hits / total if total else 0.0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": rate, "size": len(self._mem)}

    def clear(self) -> None:
        with self._lock:
            self._mem.clear()
            self.hits = 0
            self.misses = 0

    # ---- SQLite plumbing ----------------------------------------------

    def _init_db(self) -> None:
        assert self.path is not None
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                " key TEXT PRIMARY KEY,"
                " model TEXT NOT NULL,"
                " vector BLOB NOT NULL,"
                " dim INTEGER NOT NULL"
                ")"
            )

    def _warm_from_disk(self) -> None:
        assert self.path is not None
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT key, vector, dim FROM embeddings").fetchall()
        for k, blob, dim in rows:
            self._mem[k] = np.frombuffer(blob, dtype=np.float32).reshape(dim).copy()

    def _persist(self, key: str, model: str, vector: np.ndarray) -> None:
        assert self.path is not None
        v32 = vector.astype(np.float32, copy=False)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (key, model, vector, dim) VALUES (?, ?, ?, ?)",
                (key, model, v32.tobytes(), v32.shape[0]),
            )
