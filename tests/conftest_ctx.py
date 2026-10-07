"""A ready-made ToolContext over a temp workspace with synthetic stored articles."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from newsrag.config import load_config
from newsrag.engines import RuleEngine
from newsrag.ingest import ingest
from newsrag.models import Item
from newsrag.secrets import KeyStore
from newsrag.settings import Settings
from newsrag.store import Store
from newsrag.tools import ToolContext
from newsrag.workspace import open_workspace
from tests.fakes import FakeEmbedder, FakeReranker
from tests.helpers import item


def articles(now: datetime) -> list[Item]:
    def at(hours: float) -> datetime:
        return now - timedelta(hours=hours)

    rows = [
        (
            "Central bank raises interest rates to 6.75% to curb inflation",
            "The central bank raised its policy rate by 25 basis points to 6.75%. Inflation stayed "
            "above target.",
            "IN",
            "Finance",
            "Fixture Times",
            2,
        ),
        (
            "Rupee weakens after rate decision",
            "The rupee fell 0.3% against the dollar after the rate decision.",
            "IN",
            "Finance",
            "Fixture Markets",
            3,
        ),
        (
            "Federal Reserve holds interest rates steady",
            "The Federal Reserve kept rates unchanged and signalled patience on inflation.",
            "US",
            "Finance",
            "Fixture Wire",
            5,
        ),
        (
            "Chipmaker FXCHIP shares jump after earnings beat",
            "Shares of FXCHIP rose 12% after quarterly earnings beat forecasts.",
            "US",
            "Technology",
            "Fixture Tech",
            4,
        ),
        (
            "Parliament passes data protection bill",
            "The bill passed with 300 votes after a long debate in parliament.",
            "EU",
            "Politics",
            "Fixture Europe",
            6,
        ),
        (
            "Old story about energy policy from last month",
            "Ministers discussed energy policy and bills.",
            "EU",
            "Politics",
            "Fixture Europe",
            24 * 20,
        ),
    ]
    out = []
    for title, body, region, cat, source, hours in rows:
        it = item(title, body=body, region=region, category=cat, source=source)
        out.append(it.model_copy(update={"published_at": at(hours)}))
    return out


@contextmanager
def tool_context(tmp_path: Path, now: datetime | None = None) -> Iterator[ToolContext]:
    now = now or datetime.now(UTC)
    ws = open_workspace(tmp_path / "ws")
    store = Store(ws.db_path)
    cfg = load_config()
    settings = Settings()
    settings.sources.dedupe_window_days = 0
    ctx = ToolContext(
        workspace=ws,
        store=store,
        cfg=cfg,
        settings=settings,
        keys=KeyStore(),
        _embedder=FakeEmbedder(),
        _reranker=FakeReranker(),
    )
    asyncio.run(
        ingest(articles(now), store, RuleEngine(cfg), ctx.embedder, settings, ctx.keys, now=now)
    )
    try:
        yield ctx
    finally:
        store.close()
