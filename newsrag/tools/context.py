"""Everything a tool needs, created once per session. Models load lazily."""

from __future__ import annotations

from dataclasses import dataclass, field

from newsrag.config import AppConfig
from newsrag.embeddings import CrossEncoderReranker, Embedder, Reranker, make_embedder
from newsrag.secrets import KeyStore
from newsrag.settings import Settings
from newsrag.store import Store
from newsrag.workspace import Workspace


@dataclass
class ToolContext:
    workspace: Workspace
    store: Store
    cfg: AppConfig
    settings: Settings
    keys: KeyStore
    _embedder: Embedder | None = field(default=None, repr=False)
    _reranker: Reranker | None = field(default=None, repr=False)

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            base = self.settings.llm.base_url if self.settings.llm.provider == "ollama" else None
            self._embedder = make_embedder(self.settings.retrieval.embedding_model, base)
            self.workspace.check_embedding_model(self._embedder.name)
        return self._embedder

    @property
    def reranker(self) -> Reranker | None:
        if not self.settings.retrieval.rerank:
            return None
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.settings.retrieval.rerank_model)
        return self._reranker
