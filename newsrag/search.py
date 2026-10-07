"""Hybrid retrieval: filter -> keyword + semantic -> RRF fusion -> rerank -> group (FR16).

Every step works without an API key. If the embedding model is unavailable the search runs
keyword-only; if the reranker cannot load, results keep their fused order. Both cases are
reported in `SearchResult.notes`.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from newsrag.embeddings import Embedder, EmbeddingError, Reranker
from newsrag.settings import RetrievalSettings
from newsrag.store import Filters, Store

log = logging.getLogger("newsrag.search")

RRF_K = 60


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regions: list[str] | None = None
    categories: list[str] | None = None
    date_from: date | None = None
    date_to: date | None = None

    def to_store(self) -> Filters:
        start = (
            int(datetime.combine(self.date_from, time.min, tzinfo=UTC).timestamp())
            if self.date_from
            else None
        )
        end = (
            int(datetime.combine(self.date_to, time.max, tzinfo=UTC).timestamp())
            if self.date_to
            else None
        )
        return Filters(self.regions, self.categories, start, end)


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    item_id: str
    title: str
    url: str
    source: str
    region: str
    category: str
    published_date: str
    text: str
    summary: str
    engine: str
    found_by: list[str]
    fused_score: float
    semantic_score: float | None = None
    rerank_score: float | None = None


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    mode: str
    reranked: bool
    hits: list[SearchHit] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def rrf(rankings: dict[str, list[str]], weights: dict[str, float]) -> dict[str, float]:
    """Reciprocal rank fusion: sum of weight / (k + rank) over the lists an id appears in."""
    scores: dict[str, float] = {}
    for name, ids in rankings.items():
        w = weights.get(name, 1.0)
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (RRF_K + rank)
    return scores


def search(
    store: Store,
    query: str,
    filters: SearchFilters,
    settings: RetrievalSettings,
    embedder: Embedder | None,
    reranker: Reranker | None,
) -> SearchResult:
    mode = settings.search_mode
    result = SearchResult(query=query, mode=mode, reranked=False)
    f = filters.to_store()
    rankings: dict[str, list[str]] = {}
    similarity: dict[str, float] = {}

    if mode in ("hybrid", "keyword"):
        rankings["keyword"] = store.keyword_candidates(query, f, settings.candidates_k)

    if mode in ("hybrid", "semantic"):
        if embedder is None:
            result.notes.append("semantic search unavailable: no embedding model")
        else:
            try:
                ids, matrix = store.vectors(f)
                if ids:
                    qv = embedder.embed([query])[0]
                    if matrix.shape[1] != qv.shape[0]:
                        raise EmbeddingError("index was built with a different embedding size")
                    sims = matrix @ qv
                    order = np.argsort(-sims)[: settings.candidates_k]
                    rankings["semantic"] = [ids[i] for i in order]
                    similarity = {ids[i]: float(sims[i]) for i in order}
            except EmbeddingError as exc:
                result.notes.append(f"semantic search skipped: {exc}")

    w = settings.keyword_weight
    fused = rrf(rankings, {"keyword": 2 * w, "semantic": 2 * (1 - w)})
    if not fused:
        return result
    # Relevance floor: vector search always returns *something*, so a semantic-only match
    # must be similar enough. Keyword matches pass (the query words are in the text).
    keyword_ids = set(rankings.get("keyword", []))
    relevant = [
        cid
        for cid in fused
        if cid in keyword_ids or similarity.get(cid, 0.0) >= settings.min_similarity
    ]
    dropped = len(fused) - len(relevant)
    if dropped and not relevant:
        result.notes.append("no stored article is relevant enough to this query")
    ordered = sorted(relevant, key=lambda cid: -fused[cid])[: settings.candidates_k]
    details = store.chunk_details(ordered)
    ordered = [cid for cid in ordered if cid in details]

    rerank_scores: dict[str, float] = {}
    if settings.rerank and reranker is not None:
        try:
            scores = reranker.score(query, [details[c]["text"] for c in ordered])
            rerank_scores = dict(zip(ordered, scores, strict=True))
            ordered.sort(key=lambda c: -rerank_scores[c])
            result.reranked = True
        except Exception as exc:
            log.warning("Rerank skipped: %s", type(exc).__name__)
            result.notes.append(f"rerank skipped: {type(exc).__name__}")
    elif settings.rerank:
        result.notes.append("rerank skipped: no rerank model")

    per_article: dict[str, int] = {}
    for cid in ordered:
        d = details[cid]
        score = rerank_scores.get(cid)
        if result.reranked and score is not None and score < settings.min_score:
            continue
        if per_article.get(d["item_id"], 0) >= settings.max_chunks_per_article:
            continue
        per_article[d["item_id"]] = per_article.get(d["item_id"], 0) + 1
        result.hits.append(
            SearchHit(
                chunk_id=cid,
                item_id=d["item_id"],
                title=d["title"],
                url=d["url"],
                source=d["source"],
                region=d["region"],
                category=d["category"],
                published_date=datetime.fromtimestamp(d["published_ts"], tz=UTC).date().isoformat(),
                text=d["text"],
                summary=d["summary"],
                engine=d["engine"],
                found_by=[name for name, ids in rankings.items() if cid in ids],
                fused_score=round(fused[cid], 6),
                semantic_score=None if cid not in similarity else round(similarity[cid], 4),
                rerank_score=None if score is None else round(score, 4),
            )
        )
        if len(result.hits) >= settings.top_k:
            break
    return result
