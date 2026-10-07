"""One ingest run: fetch -> process -> store (FR23-FR25). Shared by the CLI, UI and tools."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from newsrag.config import AppConfig
from newsrag.embeddings import Embedder
from newsrag.engines import select_engine
from newsrag.ingest import IngestReport, ingest
from newsrag.pipeline import collect
from newsrag.secrets import KeyStore
from newsrag.settings import Settings
from newsrag.sources.http import HttpFetcher
from newsrag.store import Store


class RunOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    started_at: datetime
    finished_at: datetime
    lookback_hours: int
    fetched: int
    kept_after_filters: int
    engine: str
    engine_reason: str
    failed_sources: list[str] = Field(default_factory=list)
    ingest: IngestReport
    auto_cleanup_removed: int = 0


def catchup_hours(settings: Settings, last_run: datetime | None, now: datetime) -> int:
    """FR25: look back to the last run, at least max_age_hours, at most max_catchup_days."""
    base = settings.sources.max_age_hours
    if last_run is None:
        return base
    since = int((now - last_run).total_seconds() // 3600) + 1
    return max(base, min(since, settings.sources.max_catchup_days * 24))


async def run_ingest(
    store: Store,
    cfg: AppConfig,
    settings: Settings,
    keys: KeyStore,
    embedder: Embedder,
    *,
    regions: list[str] | None = None,
    limit: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
    http: HttpFetcher | None = None,
) -> RunOutcome:
    """Raises EngineUnavailable when settings.llm.mode is 'llm' and no LLM can be used."""
    started = datetime.now(UTC)
    hours = catchup_hours(settings, store.last_run(), started)
    fetch_settings = settings.model_copy(deep=True)
    fetch_settings.sources.max_age_hours = hours
    if regions:
        fetch_settings.sources.regions = regions
    report = await collect(cfg, fetch_settings, keys, http=http, now=started)
    items = report.items[:limit] if limit else report.items

    choice = await select_engine(settings.llm, cfg, keys, task="processing")
    try:
        result = await ingest(
            items,
            store,
            choice.engine,
            embedder,
            settings,
            keys,
            now=started,
            on_progress=on_progress,
        )
    finally:
        await choice.aclose()

    finished = datetime.now(UTC)
    outcome = RunOutcome(
        started_at=started,
        finished_at=finished,
        lookback_hours=hours,
        fetched=report.fetched,
        kept_after_filters=len(report.items),
        engine=choice.engine.label,
        engine_reason=choice.reason,
        failed_sources=[s.source for s in report.sources if s.status == "error"],
        ingest=result,
    )
    if settings.data.auto_cleanup and settings.data.retention_days:
        outcome.auto_cleanup_removed = store.cleanup(
            settings.data.retention_days, now=finished, apply=True
        ).items
    store.record_run(started, finished, outcome.model_dump(mode="json"))
    return outcome
