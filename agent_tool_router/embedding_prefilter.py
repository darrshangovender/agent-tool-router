"""Embedding-based prefilter for the tool catalogue.

Stage 1 of the two-stage router: cheap, deterministic, and runs every query.
We embed each tool's name + description once at construction, then for each
query embed the query and rank tools by cosine similarity.

Default encoder is `sentence-transformers/all-MiniLM-L6-v2` (local, ~80MB,
no API key, ~5ms per query on CPU). Pass `backend="openai"` to swap in
`text-embedding-3-small` instead — same interface, different vendor.

The prefilter intentionally does NOT decide. It produces a shortlist and a
similarity floor; the LLM judge actually picks. Two reasons:

  1. Embeddings are good at "is this query in the same neighbourhood as this
     tool description" but they cannot reason about argument applicability
     or the difference between "look up" and "look up and refund".
  2. Putting the embedding score in front of the judge lets the judge use it
     as evidence ("the top candidate scored 0.81, the next 0.34 — clear win").
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .cache import EmbeddingCache


class Encoder(Protocol):
    """Anything that can turn a list of strings into an (N, D) float32 array."""

    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Built-in encoders
# ---------------------------------------------------------------------------


class SentenceTransformerEncoder:
    """Local encoder. Default for demos and tests — no API key required."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        # Lazy import: keeps the base install light when callers bring their own encoder.
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32, copy=False)


class OpenAIEncoder:
    """OpenAI-hosted encoder. Activated by `--openai-embeddings` in the demos."""

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        import os

        from openai import OpenAI  # noqa: WPS433

        self.model_name = model_name
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def encode(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.model_name, input=texts)
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # OpenAI vectors are already unit-norm but normalise defensively.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class StubEncoder:
    """Deterministic hash-based encoder. Used in tests so the suite has no model dependency.

    Buckets are chosen with blake2b rather than the builtin ``hash()``: string
    hashing is salted per interpreter (PYTHONHASHSEED), so ``hash()`` put the same
    token in a different bucket on every run. That made rankings vary run to run —
    the suite failed intermittently — and it silently invalidated any
    ``EmbeddingCache`` written by an earlier process, since the cached vectors came
    from a different hash space than the ones being compared against them.
    """

    def __init__(self, dim: int = 64, model_name: str = "stub") -> None:
        self.dim = dim
        self.model_name = model_name

    @staticmethod
    def _bucket(token: str) -> int:
        return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                idx = self._bucket(tok) % self.dim
                out[i, idx] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def make_encoder(backend: str = "sentence-transformers", model_name: str | None = None) -> Encoder:
    if backend in ("sentence-transformers", "st", "local"):
        return SentenceTransformerEncoder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    if backend == "openai":
        return OpenAIEncoder(model_name or "text-embedding-3-small")
    if backend == "stub":
        return StubEncoder()
    raise ValueError(f"Unknown embedding backend: {backend!r}")


# ---------------------------------------------------------------------------
# Prefilter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A single shortlisted tool with its prefilter score."""

    name: str
    score: float  # cosine similarity, in [-1, 1] (typically [0, 1] for unit-norm)


class EmbeddingPrefilter:
    """Holds the encoded tool corpus and ranks queries against it."""

    def __init__(
        self,
        tool_names: list[str],
        tool_texts: list[str],
        *,
        encoder: Encoder | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        if len(tool_names) != len(tool_texts):
            raise ValueError("tool_names and tool_texts must be the same length")
        self.tool_names = list(tool_names)
        self.encoder: Encoder = encoder or make_encoder("stub" if not tool_names else "sentence-transformers")
        self.cache = cache
        self._matrix: np.ndarray = (
            self.encoder.encode(tool_texts) if tool_texts else np.zeros((0, 0), dtype=np.float32)
        )

    @property
    def model_name(self) -> str:
        return self.encoder.model_name

    @property
    def dim(self) -> int:
        return int(self._matrix.shape[1]) if self._matrix.size else 0

    def top_k(self, query: str, k: int = 3) -> list[Candidate]:
        """Return up to k candidates ranked by cosine similarity, descending.

        Empty corpus → empty list. Duplicate descriptions tie on score and the
        tie-break is insertion order (numpy argsort is stable for `kind='stable'`).
        """
        if not self.tool_names:
            return []
        k = max(1, min(k, len(self.tool_names)))
        qvec = self._embed_query(query)
        scores = self._matrix @ qvec  # both are unit-norm → cosine sim
        # Sort the negated scores rather than reversing an ascending sort: a
        # stable ascending sort keeps ties in insertion order, but reversing the
        # whole array flips them, so the *last*-registered of two equally-scoring
        # tools came out on top — the opposite of the documented tie-break.
        order = np.argsort(-scores, kind="stable")[:k]
        return [Candidate(name=self.tool_names[i], score=float(scores[i])) for i in order]

    def _embed_query(self, query: str) -> np.ndarray:
        if self.cache is not None:
            cached = self.cache.get(query, self.encoder.model_name)
            if cached is not None:
                return cached
        vec = self.encoder.encode([query])[0]
        if self.cache is not None:
            self.cache.put(query, self.encoder.model_name, vec)
        return vec
