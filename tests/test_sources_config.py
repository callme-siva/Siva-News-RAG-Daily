from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from newsrag.config import SourceConfig, load_config
from newsrag.sources_config import (
    add_source,
    check_feed,
    effective_config,
    export_opml,
    load_overrides,
    parse_opml,
    remove_source,
    reset_sources,
    set_enabled,
)
from newsrag.store import Store, StoreError
from tests.conftest_ctx import tool_context
from tests.helpers import FIXTURES, fetcher

BASE = load_config()
NEW = SourceConfig(
    name="My Fixture Feed",
    type="rss",
    url="https://example.com/feed.xml",
    region="IN",
    category="Finance",
)


def test_defaults_untouched_until_changed(tmp_path: Path) -> None:
    assert effective_config(tmp_path) == BASE
    assert not (tmp_path / "sources.json").exists()


def test_add_disable_enable_remove(tmp_path: Path) -> None:
    add_source(tmp_path, NEW)
    names = [s.name for s in effective_config(tmp_path).sources]
    assert "My Fixture Feed" in names and len(names) == len(BASE.sources) + 1

    set_enabled(tmp_path, "PIB", False)
    cfg = effective_config(tmp_path)
    assert next(s for s in cfg.sources if s.name == "PIB").enabled is False

    set_enabled(tmp_path, "PIB", True)  # back to default -> override removed
    assert "PIB" not in load_overrides(tmp_path).enabled

    set_enabled(tmp_path, "GDELT India Finance", True)  # off by default
    assert next(
        s for s in effective_config(tmp_path).sources if s.name == "GDELT India Finance"
    ).enabled

    set_enabled(tmp_path, "My Fixture Feed", False)
    assert not next(
        s for s in effective_config(tmp_path).sources if s.name == "My Fixture Feed"
    ).enabled

    remove_source(tmp_path, "My Fixture Feed")
    assert "My Fixture Feed" not in [s.name for s in effective_config(tmp_path).sources]
    with pytest.raises(KeyError):
        remove_source(tmp_path, "PIB")  # shipped sources can only be disabled

    reset_sources(tmp_path)
    assert effective_config(tmp_path) == BASE


def test_add_rejects_duplicates_and_unknown_region(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Duplicate source"):
        add_source(tmp_path, NEW.model_copy(update={"name": "PIB"}))
    with pytest.raises(ValidationError, match="unknown region"):
        add_source(tmp_path, NEW.model_copy(update={"region": "JP"}))
    assert not (tmp_path / "sources.json").exists()


def test_check_feed_ok_blocked_and_not_a_feed() -> None:
    feed = (FIXTURES / "synthetic_feed.xml").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "blocked.example.com" and request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /")
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.host == "html.example.com":
            return httpx.Response(200, text="<html><body>hello</body></html>")
        return httpx.Response(200, content=feed)

    async def run() -> tuple[object, object, object]:
        async with fetcher(handler, attempts=1) as http:
            return (
                await check_feed("https://good.example.com/feed.xml", http),
                await check_feed("https://blocked.example.com/feed.xml", http),
                await check_feed("https://html.example.com/page", http),
            )

    ok, blocked, html = asyncio.run(run())
    assert ok.ok and ok.title == "Synthetic Finance Feed" and len(ok.headlines) == 3  # type: ignore[attr-defined]
    assert not blocked.ok and "robots.txt" in (blocked.error or "")  # type: ignore[attr-defined]
    assert not html.ok and "no feed entries" in (html.error or "")  # type: ignore[attr-defined]


def test_opml_round_trip() -> None:
    text = export_opml(BASE)
    sources = parse_opml(text, "US", "Technology")
    rss = [s for s in BASE.sources if s.type == "rss"]
    assert [s.url for s in sources] == [s.url for s in rss]
    assert {(s.region, s.category) for s in sources} == {(s.region, s.category) for s in rss}
    with pytest.raises(ValueError, match="not valid OPML"):
        parse_opml("<not xml", "US", "Technology")


def test_delete_where_preview_and_apply(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    with tool_context(tmp_path, now) as ctx:
        store: Store = ctx.store
        assert store.delete_where(sources=["Fixture Europe"], apply=False) == 2
        assert store.counts(None, None)["total"] == 6
        assert store.delete_where(regions=["IN"], categories=["Finance"], apply=True) == 2
        cutoff = int((now - timedelta(days=10)).timestamp())
        assert store.delete_where(to_ts=cutoff, apply=True) == 1
        assert store.counts(None, None)["total"] == 3
        assert store.verify(repair=False).clean
        with pytest.raises(StoreError):
            store.delete_where(apply=True)
        assert "Fixture Tech" in store.sources_in_store()
