"""Run every enabled source in parallel and collect per-source results (FR1, R11)."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta

from newsrag.config import AppConfig, SourceConfig
from newsrag.secrets import KeyStore
from newsrag.settings import Settings
from newsrag.sources import MissingKey, build_adapter
from newsrag.sources import gdelt as gdelt_mod
from newsrag.sources.base import SourceResult
from newsrag.sources.http import HttpFetcher

log = logging.getLogger("newsrag.pipeline.fetch")

SOURCE_TIMEOUT_S = 90.0
MAX_PARALLEL = 8


def selected_sources(cfg: AppConfig, settings: Settings) -> list[SourceConfig]:
    regions = set(settings.sources.regions)
    return [s for s in cfg.sources if s.enabled and s.region in regions]


def default_fetcher() -> HttpFetcher:
    return HttpFetcher(min_interval_s={gdelt_mod.HOST: gdelt_mod.MIN_INTERVAL_S})


async def fetch_source(
    source: SourceConfig, http: HttpFetcher, keys: KeyStore, since: datetime
) -> SourceResult:
    start = time.monotonic()
    try:
        adapter = build_adapter(source, keys)
    except MissingKey as exc:
        return SourceResult(source=source.name, status="skipped", error=f"no {exc} key")
    try:
        items = await asyncio.wait_for(adapter.fetch(http, since), timeout=SOURCE_TIMEOUT_S)
    except Exception as exc:
        message = keys.redact(f"{type(exc).__name__}: {exc}")[:300]
        log.warning("Source %s failed: %s", source.name, message)
        return SourceResult(
            source=source.name,
            status="error",
            error=message,
            elapsed_s=round(time.monotonic() - start, 2),
        )
    return SourceResult(
        source=source.name,
        status="ok" if items else "empty",
        count=len(items),
        items=items,
        elapsed_s=round(time.monotonic() - start, 2),
    )


async def fetch_all(
    sources: list[SourceConfig],
    keys: KeyStore,
    *,
    since: datetime | None = None,
    max_age_hours: int = 48,
    http: HttpFetcher | None = None,
) -> list[SourceResult]:
    """Fetch all `sources` concurrently. One failing source never stops the others."""
    since = since or datetime.now(UTC) - timedelta(hours=max_age_hours)
    gate = asyncio.Semaphore(MAX_PARALLEL)
    fetcher = http or default_fetcher()

    async def one(source: SourceConfig) -> SourceResult:
        async with gate:
            return await fetch_source(source, fetcher, keys, since)

    if http is not None:
        return list(await asyncio.gather(*(one(s) for s in sources)))
    async with fetcher:
        return list(await asyncio.gather(*(one(s) for s in sources)))
