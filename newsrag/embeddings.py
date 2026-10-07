"""Local embedding and rerank models (REQUIREMENTS FR14, FR16 step 5, FR11c).

Models load lazily on first use, so commands that never search or ingest never import torch.
`embedding_model` values:
  - a sentence-transformers model name (default `sentence-transformers/all-MiniLM-L6-v2`)
  - `ollama:<model>` to embed with a local Ollama server (e.g. `ollama:nomic-embed-text`)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol

import httpx
import numpy as np
import numpy.typing as npt

log = logging.getLogger("newsrag.embeddings")

Vectors = npt.NDArray[np.float32]


class EmbeddingError(Exception):
    pass


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> Vectors:
        """Return one L2-normalised float32 row per text."""
        ...


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance of each passage to the query, 0..1 (higher is better)."""
        ...


def _normalise(m: Vectors) -> Vectors:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, name: str) -> None:
        self.name = name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError("sentence-transformers is not installed") from exc
            try:
                self._model = SentenceTransformer(self.name)
            except Exception as exc:
                raise EmbeddingError(f"could not load embedding model {self.name!r}") from exc
        return self._model

    def embed(self, texts: list[str]) -> Vectors:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._load().encode(texts, normalize_embeddings=True, batch_size=32)
        return np.asarray(vectors, dtype=np.float32)


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self.name = f"ollama:{model}"
        self._model = model
        self._url = base_url.rstrip("/") + "/api/embed"

    def embed(self, texts: list[str]) -> Vectors:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        try:
            resp = httpx.post(self._url, json={"model": self._model, "input": texts}, timeout=120)
            resp.raise_for_status()
            rows = resp.json()["embeddings"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise EmbeddingError(f"Ollama embedding failed: {type(exc).__name__}") from exc
        return _normalise(np.asarray(rows, dtype=np.float32))


class CrossEncoderReranker:
    def __init__(self, name: str) -> None:
        self.name = name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.name)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        raw = self._load().predict([(query, p) for p in passages])
        return [1.0 / (1.0 + math.exp(-float(x))) for x in raw]


def make_embedder(name: str, ollama_base_url: str | None = None) -> Embedder:
    if name.startswith("ollama:"):
        return OllamaEmbedder(name.split(":", 1)[1], ollama_base_url or "http://localhost:11434")
    return SentenceTransformerEmbedder(name)


def to_blob(v: Vectors) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes) -> Vectors:
    return np.frombuffer(b, dtype=np.float32)
