"""Deterministic pipeline: fetch, filter, dedupe, rank. No LLM involved."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from newsrag.config import AppConfig
from newsrag.models import Item
from newsrag.pipeline.dedupe import dedupe_batch
from newsrag.pipeline.fetch import fetch_all, selected_sources
from newsrag.pipeline.filter import FilterStats, filter_items
from newsrag.pipeline.rank import rank_and_cap
from newsrag.secrets import KeyStore
from newsrag.settings import Settings
from newsrag.sources.base import SourceResult
from newsrag.sources.http import HttpFetcher


class CollectReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[SourceResult]
    fetched: int
    filter: FilterStats
    merged_duplicates: int
    capped: int
    items: list[Item] = Field(default_factory=list)


def process_batch(
    items: list[Item], cfg: AppConfig, settings: Settings, now: datetime
) -> tuple[list[Item], FilterStats, int, int]:
    """Filter, dedupe and rank already-fetched items. Pure, so it is easy to test."""
    s = settings.sources
    kept, stats = filter_items(items, cfg, now=now, max_age_hours=s.max_age_hours)
    unique, merged = dedupe_batch(kept, s.dedupe_threshold)
    ranked = rank_and_cap(unique, s.max_per_group)
    return ranked, stats, merged, len(unique) - len(ranked)


async def collect(
    cfg: AppConfig,
    settings: Settings,
    keys: KeyStore,
    *,
    http: HttpFetcher | None = None,
    now: datetime | None = None,
) -> CollectReport:
    now = now or datetime.now(UTC)
    results = await fetch_all(
        selected_sources(cfg, settings),
        keys,
        max_age_hours=settings.sources.max_age_hours,
        http=http,
    )
    fetched = [i for r in results for i in r.items]
    ranked, stats, merged, capped = process_batch(fetched, cfg, settings, now)
    return CollectReport(
        sources=[r.model_copy(update={"items": []}) for r in results],
        fetched=len(fetched),
        filter=stats,
        merged_duplicates=merged,
        capped=capped,
        items=ranked,
    )
