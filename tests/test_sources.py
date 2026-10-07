from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
import pytest

from newsrag.config import SourceConfig
from newsrag.models import FetchedVia
from newsrag.pipeline.fetch import fetch_source
from newsrag.secrets import REDACTED, KeyStore
from newsrag.sources import MissingKey, build_adapter
from newsrag.sources.gdelt import GdeltAdapter
from newsrag.sources.http import BlockedByRobots
from newsrag.sources.keyed import GNewsAdapter, NewsDataAdapter
from newsrag.sources.rss import RSSAdapter
from tests.helpers import FIXTURES, NOW, fetcher, robots_ok, rss_source

FEED = (FIXTURES / "synthetic_feed.xml").read_bytes()


def test_rss_parse_handles_real_world_quirks() -> None:
    src = SourceConfig(
        name="Fixture Feed",
        type="rss",
        region="IN",
        category="Finance",
        url="https://example.com/feed.xml",
        timezone="Asia/Kolkata",
    )
    items = RSSAdapter(src).parse(FEED, now=NOW)
    by_title = {i.title: i for i in items}
    assert len(items) == 4  # missing-link and empty-title entries skipped

    rates = by_title["Fixture Central Bank holds benchmark interest rates steady"]
    assert rates.url == "https://example.com/markets/fixture-rates"
    assert rates.body.startswith("The fixture central bank kept rates unchanged")
    assert "&amp;" not in rates.body and "track()" not in rates.body
    assert rates.fetched_via == FetchedVia.RSS and rates.source == "Fixture Feed"

    naive = by_title["Reserve note on overnight liquidity operations"]
    assert naive.published_at == datetime(2026, 10, 7, 7, 20, tzinfo=UTC)

    sebi = by_title["Regulator issues circular on mutual fund disclosures"]
    assert sebi.published_at == datetime(2026, 10, 4, 18, 30, tzinfo=UTC)

    undated = by_title["Ministry publishes quarterly trade statistics"]
    assert undated.published_at == NOW and undated.tags == {"date_estimated": "true"}


def test_rss_unparseable_content_raises() -> None:
    with pytest.raises(ValueError, match="could not be parsed"):
        RSSAdapter(rss_source()).parse(b"<html>not a feed", now=NOW)


def test_gdelt_parse() -> None:
    src = SourceConfig(name="G", type="gdelt", region="IN", category="Finance", query="x")
    data = {
        "articles": [
            {
                "url": "https://example.org/a",
                "title": "Fixture GDELT headline",
                "seendate": "20261006T083000Z",
                "domain": "example.org",
                "language": "English",
            },
            {"url": "https://example.org/b", "title": "No date", "language": "English"},
            {
                "url": "https://example.org/c",
                "title": "Other language",
                "seendate": "20261006T083000Z",
                "language": "Hindi",
            },
        ]
    }
    items = GdeltAdapter(src).parse(data)
    assert [i.title for i in items] == ["Fixture GDELT headline"]
    assert items[0].source == "example.org" and items[0].body == ""


def test_keyed_parsers() -> None:
    src = SourceConfig(
        name="K", type="api", api="gnews", region="US", category="Finance", query="x"
    )
    gnews = GNewsAdapter(src, "k").parse(
        {
            "articles": [
                {
                    "title": "Fixture GNews item",
                    "description": "Desc.",
                    "content": "Content [+100 chars]",
                    "url": "https://example.org/g",
                    "publishedAt": "2026-10-06T08:30:00Z",
                    "source": {"name": "Fixture Daily"},
                }
            ]
        }
    )
    assert gnews[0].source == "Fixture Daily" and gnews[0].body == "Desc. Content"

    newsdata = NewsDataAdapter(src, "k").parse(
        {
            "results": [
                {
                    "title": "Fixture NewsData item",
                    "link": "https://example.org/n",
                    "pubDate": "2026-10-06 08:30:00",
                    "source_name": "Fixture Times",
                },
                {"title": "Missing link", "pubDate": "2026-10-06 08:30:00"},
            ]
        }
    )
    assert [i.title for i in newsdata] == ["Fixture NewsData item"]


def test_keyed_source_without_key_is_skipped() -> None:
    src = SourceConfig(name="K", type="api", api="gnews", region="US", category="Finance")
    with pytest.raises(MissingKey):
        build_adapter(src, KeyStore())

    async def run() -> str:
        async with fetcher(robots_ok(lambda r: httpx.Response(200))) as http:
            res = await fetch_source(src, http, KeyStore(), NOW)
        return f"{res.status}|{res.error}"

    assert asyncio.run(run()) == "skipped|no gnews key"


def test_robots_disallow_blocks_and_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /")
        raise AssertionError("feed must not be requested")

    async def run() -> None:
        async with fetcher(handler) as http:
            with pytest.raises(BlockedByRobots):
                await http.get("https://example.com/feed.xml")
            res = await fetch_source(rss_source(), http, KeyStore(), NOW)
            assert res.status == "error" and "robots.txt disallows" in (res.error or "")

    asyncio.run(run())


@pytest.mark.parametrize(("robots_status", "allowed"), [(404, True), (418, True), (503, False)])
def test_robots_status_semantics(robots_status: int, allowed: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(robots_status if request.url.path == "/robots.txt" else 200)

    async def run() -> bool:
        async with fetcher(handler, attempts=1) as http:
            return await http.allowed("https://example.com/feed.xml")

    assert asyncio.run(run()) is allowed


def test_retries_on_503_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, content=FEED)

    async def run() -> int:
        async with fetcher(robots_ok(handler), attempts=3) as http:
            res = await fetch_source(rss_source(), http, KeyStore(), NOW)
        return res.count

    assert asyncio.run(run()) == 4
    assert calls["n"] == 3


def test_errors_never_contain_keys() -> None:
    key = "fixturekeyvalue123456"
    keys = KeyStore()
    keys.set("gnews", key)
    src = SourceConfig(
        name="K", type="api", api="gnews", region="US", category="Finance", query="x"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"invalid key {key}")

    async def run() -> str:
        async with fetcher(robots_ok(handler), attempts=1) as http:
            res = await fetch_source(src, http, keys, NOW)
        return res.error or ""

    error = asyncio.run(run())
    assert key not in error and REDACTED in error  # URL in the 401 message held apikey=...
