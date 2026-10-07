"""Deterministic stand-ins for the embedding and rerank models (no downloads in tests)."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from newsrag.embeddings import Vectors

_WORD = re.compile(r"[a-z0-9]+")
DIM = 64


class FakeEmbedder:
    """Hashed bag-of-words: texts sharing words get similar vectors."""

    name = "fake-embedder"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> Vectors:
        self.calls += 1
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                if len(word) < 3:
                    continue
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[row, h % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)


class FakeReranker:
    """Score = share of query words present in the passage."""

    name = "fake-reranker"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = {w for w in _WORD.findall(query.lower()) if len(w) >= 3}
        return [len(q & set(_WORD.findall(p.lower()))) / (len(q) or 1) for p in passages]


class BrokenReranker:
    name = "broken"

    def score(self, query: str, passages: list[str]) -> list[float]:
        raise RuntimeError("model failed to load")
