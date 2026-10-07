"""Shared helpers for tests: synthetic items and a mock HTTP transport."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from newsrag.config import SourceConfig
from newsrag.models import FetchedVia, Item
from newsrag.sources.http import HttpFetcher

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)

Handler = Callable[[httpx.Request], httpx.Response]


def item(
    title: str,
    *,
    url: str | None = None,
    body: str = "",
    source: str = "Fixture Outlet",
    region: str = "IN",
    category: str = "Finance",
    hours_ago: float = 1.0,
) -> Item:
    slug = "-".join(title.lower().split())[:60]
    return Item(
        title=title,
        body=body,
        url=url or f"https://example.com/{slug}",
        source=source,
        region=region,
        category=category,
        published_at=NOW - timedelta(hours=hours_ago),
        fetched_via=FetchedVia.RSS,
    )


def rss_source(
    name: str = "Fixture Feed", url: str = "https://example.com/feed.xml"
) -> SourceConfig:
    return SourceConfig(name=name, type="rss", region="IN", category="Finance", url=url)


def fetcher(handler: Handler, attempts: int = 3) -> HttpFetcher:
    """HttpFetcher on a mock transport, no backoff waits."""
    return HttpFetcher(transport=httpx.MockTransport(handler), backoff_s=0, attempts=attempts)


def robots_ok(handler: Handler) -> Handler:
    """Wrap a handler so every robots.txt request returns 404 (allow all)."""

    def wrapped(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return handler(request)

    return wrapped
