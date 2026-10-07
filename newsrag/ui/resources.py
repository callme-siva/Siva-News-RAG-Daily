"""Shared, cached resources for the UI. Tests replace `make_models` to avoid downloads."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from newsrag.embeddings import CrossEncoderReranker, Embedder, Reranker, make_embedder


@st.cache_resource(show_spinner="Loading the embedding model...")
def _embedder(name: str, base_url: str | None) -> Embedder:
    return make_embedder(name, base_url)


@st.cache_resource(show_spinner="Loading the rerank model...")
def _reranker(name: str) -> Reranker:
    return CrossEncoderReranker(name)


def _default_models(embedding: str, base_url: str | None, rerank: str) -> tuple[Embedder, Reranker]:
    return _embedder(embedding, base_url), _reranker(rerank)


# Replaced in tests with fakes. Signature: (embedding_model, ollama_base_url, rerank_model).
make_models: Callable[[str, str | None, str], tuple[Embedder, Reranker]] = _default_models
