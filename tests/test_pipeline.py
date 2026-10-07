from __future__ import annotations

import asyncio

import httpx

from newsrag.config import AppConfig, Category, Region, SourceConfig
from newsrag.pipeline import collect, process_batch
from newsrag.pipeline.dedupe import dedupe_batch
from newsrag.pipeline.filter import filter_items
from newsrag.pipeline.rank import rank_and_cap
from newsrag.secrets import KeyStore
from newsrag.settings import Settings
from tests.helpers import FIXTURES, NOW, fetcher, item, robots_ok

CFG = AppConfig(
    regions=[Region(code="IN", name="India"), Region(code="US", name="United States")],
    categories=[
        Category(name="Finance", include=r"\b(rates?|inflation|bank)\b", exclude=r"\bhoroscope\b")
    ],
    official_domains=["rbi.org.in"],
)


def test_filter_age_exclude_include_and_official() -> None:
    items = [
        item("Bank raises rates again"),
        item("Old bank rates story", hours_ago=72),
        item("Weekly horoscope for bank workers"),
        item("Celebrity wedding photos"),
        item("Press release 42", url="https://www.rbi.org.in/press/42"),
    ]
    kept, stats = filter_items(items, CFG, now=NOW, max_age_hours=48)
    assert [i.title for i in kept] == ["Bank raises rates again", "Press release 42"]
    assert (stats.too_old, stats.excluded, stats.not_relevant, stats.kept) == (1, 1, 1, 2)


def test_dedupe_url_variants_become_one_item() -> None:
    a = item("Bank holds rates", url="https://www.example.com/x?utm_source=a", source="A")
    b = item("Bank holds rates", url="http://example.com/x/#top", source="B")
    unique, merged = dedupe_batch([a, b], threshold=0.5)
    assert len(unique) == 1 and merged == 1
    assert unique[0].also_reported_by in (("A",), ("B",))


def test_dedupe_same_story_keeps_richest_and_all_sources() -> None:
    short = item("Central bank holds interest rates steady", source="Outlet A", body="Short.")
    rich = item(
        "Central bank holds interest rates steady in October",
        url="https://example.org/rich",
        source="Outlet B",
        body="A much longer body with detail about the decision.",
    )
    other = item(
        "Central bank keeps interest rates steady", url="https://example.net/o", source="C"
    )
    unique, merged = dedupe_batch([short, other, rich], threshold=0.5)
    assert len(unique) == 1 and merged == 2
    kept = unique[0]
    assert kept.source == "Outlet B"
    assert kept.also_reported_by == ("C", "Outlet A")
    assert set(kept.also_reported_urls) == {short.url, other.url}


def test_dedupe_does_not_merge_different_or_tiny_headlines() -> None:
    a = item("Central bank holds interest rates steady")
    b = item("Chipmaker reports record quarterly revenue")
    c = item("Live updates")
    d = item("Live updates now", url="https://example.org/live2")
    unique, merged = dedupe_batch([a, b, c, d], threshold=0.5)
    assert len(unique) == 4 and merged == 0


def test_dedupe_is_order_independent() -> None:
    items = [
        item("Central bank holds interest rates steady", source="A", body="aa"),
        item("Central bank holds interest rates steady today", url="https://e.org/2", source="B"),
    ]
    one, _ = dedupe_batch(items, 0.5)
    two, _ = dedupe_batch(list(reversed(items)), 0.5)
    assert one == two


def test_rank_and_cap_per_group() -> None:
    popular = item("Popular bank story", hours_ago=5).model_copy(
        update={"also_reported_by": ("X", "Y")}
    )
    newest = item("Newest bank story", hours_ago=0.1)
    older = item("Older bank story", hours_ago=3)
    us = item("US bank story", region="US")
    ranked = rank_and_cap([older, newest, popular, us], max_per_group=2)
    assert [i.title for i in ranked] == ["Popular bank story", "Newest bank story", "US bank story"]


def test_process_batch_counts() -> None:
    items = [
        item("Bank raises rates again", source="A"),
        item("Bank raises rates again today", url="https://e.org/2", source="B"),
        item("Old bank rates story", hours_ago=100),
    ]
    ranked, stats, merged, capped = process_batch(items, CFG, Settings(), NOW)
    assert (len(ranked), stats.too_old, merged, capped) == (1, 1, 1, 0)


def test_collect_end_to_end_with_one_failing_source() -> None:
    """One good feed, one broken feed: the run completes and reports both."""
    cfg = CFG.model_copy(
        update={
            "categories": [Category(name="Finance")],
            "sources": [
                SourceConfig(
                    name="Good",
                    type="rss",
                    region="IN",
                    category="Finance",
                    url="https://good.example.com/feed.xml",
                    timezone="Asia/Kolkata",
                ),
                SourceConfig(
                    name="Broken",
                    type="rss",
                    region="IN",
                    category="Finance",
                    url="https://broken.example.com/feed.xml",
                ),
                SourceConfig(
                    name="Off",
                    type="rss",
                    region="IN",
                    category="Finance",
                    url="https://off.example.com/feed.xml",
                    enabled=False,
                ),
            ],
        }
    )
    feed = (FIXTURES / "synthetic_feed.xml").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "good.example.com":
            return httpx.Response(200, content=feed)
        if request.url.host == "broken.example.com":
            return httpx.Response(404)
        raise AssertionError(f"unexpected request {request.url}")

    settings = Settings()
    settings.sources.regions = ["IN"]

    async def run() -> None:
        async with fetcher(robots_ok(handler), attempts=1) as http:
            report = await collect(cfg, settings, KeyStore(), http=http, now=NOW)
        statuses = {r.source: r.status for r in report.sources}
        assert statuses == {"Good": "ok", "Broken": "error"}
        assert report.fetched == 4
        assert all(not r.items for r in report.sources)
        # The SEBI-style item is about 2 days old relative to NOW and is dropped as too old.
        assert report.filter.too_old == 1
        assert len(report.items) == 3

    asyncio.run(run())
