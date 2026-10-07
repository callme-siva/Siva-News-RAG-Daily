"""Store new items: skip seen, merge cross-day duplicates, process, chunk, embed, write.

Order matters (REQUIREMENTS 5.2.1):
  1. DD2 skip URLs already seen        -> no LLM call is spent on them
  2. DD5 same URL, changed text        -> reprocess only if on_update = "replace"
  3. DD3 near-duplicate of a stored item in the last N days -> attach as extra source
  4. process -> irrelevant items are marked seen (never reprocessed) but not stored
  5. chunk + embed, then one transaction per item (DD4)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from newsrag.embeddings import Embedder, EmbeddingError
from newsrag.engines import Engine, process_all
from newsrag.models import Item
from newsrag.pipeline.chunk import build_chunks
from newsrag.pipeline.dedupe import headline_words, jaccard, same_story
from newsrag.secrets import KeyStore
from newsrag.settings import Settings
from newsrag.store import Store

log = logging.getLogger("newsrag.ingest")

BORDERLINE_MARGIN = 0.2


class IngestReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new: int = 0
    skipped_seen: int = 0
    merged_duplicate: int = 0
    updated: int = 0
    irrelevant: int = 0
    failed: int = 0
    by_engine: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


def _cross_day_match(
    item: Item,
    recent: list[tuple[str, str]],
    threshold: float,
    embedder: Embedder | None,
    embed_threshold: float,
) -> str | None:
    """Return the stored item_id this item duplicates, if any."""
    words = headline_words(item.title)
    borderline: list[tuple[str, str]] = []
    for item_id, title in recent:
        other = headline_words(title)
        if same_story(words, other, threshold):
            return item_id
        if (
            len(words) >= 3
            and len(other) >= 3
            and jaccard(words, other) >= threshold - BORDERLINE_MARGIN
        ):
            borderline.append((item_id, title))
    if not borderline or embedder is None:
        return None
    try:
        vecs = embedder.embed([item.title, *(t for _, t in borderline)])
    except EmbeddingError:
        return None
    sims = vecs[1:] @ vecs[0]
    best = int(np.argmax(sims))
    return borderline[best][0] if float(sims[best]) >= embed_threshold else None


def reindex(
    store: Store,
    embedder: Embedder,
    settings: Settings,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Re-chunk and re-embed every stored item with the current chunk settings and
    embedding model. No fetching and no LLM calls. Returns the number of items rebuilt."""
    r = settings.retrieval
    pairs = list(store.iter_stored())
    for n, (item, processed) in enumerate(pairs, start=1):
        chunks = build_chunks(item, processed, r.chunk_size, r.chunk_overlap)
        store.replace_chunks(processed.item_id, chunks, embedder.embed([c.text for c in chunks]))
        if on_progress is not None:
            on_progress(n, len(pairs))
    return len(pairs)


async def ingest(
    items: list[Item],
    store: Store,
    engine: Engine,
    embedder: Embedder,
    settings: Settings,
    keys: KeyStore,
    *,
    now: datetime,
    on_progress: Callable[[int, int], None] | None = None,
    stop: asyncio.Event | None = None,
) -> IngestReport:
    s = settings.sources
    r = settings.retrieval
    report = IngestReport()

    seen = store.seen_urls(i.url for i in items)
    to_process: list[Item] = []
    replacing: set[str] = set()
    for item in items:
        if item.url in seen:
            if s.on_update == "replace":
                stored_hash = store.content_hash(item.item_id)
                if stored_hash is not None and stored_hash != item.content_hash:
                    to_process.append(item)
                    replacing.add(item.item_id)
                    continue
            report.skipped_seen += 1
            continue
        to_process.append(item)

    since_ts = int((now - timedelta(days=s.dedupe_window_days)).timestamp())
    recent_cache: dict[str, list[tuple[str, str]]] = {}
    fresh: list[Item] = []
    for item in to_process:
        if item.item_id in replacing or s.dedupe_window_days == 0:
            fresh.append(item)
            continue
        recent = recent_cache.setdefault(
            item.category, store.recent_titles(item.category, since_ts)
        )
        match = _cross_day_match(item, recent, s.dedupe_threshold, embedder, s.embed_dup_threshold)
        if match is not None:
            store.attach_source(match, item.source, item.url)
            report.merged_duplicate += 1
            continue
        fresh.append(item)

    processed = await process_all(
        engine,
        fresh,
        concurrency=settings.llm.concurrency,
        on_progress=on_progress,
        stop=stop,
    )
    by_id = {p.item_id: p for p in processed}

    for item in fresh:
        p = by_id.get(item.item_id)
        if p is None:  # stopped before this item; it stays unseen and is retried next run
            continue
        if not p.relevant:
            store.mark_seen([item.url, *item.also_reported_urls], item.item_id, "irrelevant")
            report.irrelevant += 1
            continue
        chunks = build_chunks(item, p, r.chunk_size, r.chunk_overlap)
        try:
            vectors = embedder.embed([c.text for c in chunks])
            replace = item.item_id in replacing
            store.write_item(item, p, chunks, vectors, replace=replace)
        except Exception as exc:
            message = keys.redact(f"{item.source}: {type(exc).__name__}: {exc}")[:300]
            log.warning("Could not store %s: %s", item.url, message)
            report.failed += 1
            report.errors.append(message)
            continue
        if item.item_id in replacing:
            report.updated += 1
        else:
            report.new += 1
        report.by_engine[p.engine.value] = report.by_engine.get(p.engine.value, 0) + 1
    return report
