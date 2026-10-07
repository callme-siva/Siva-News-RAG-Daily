"""RSS and Atom feeds (no key)."""

from __future__ import annotations

import logging
import warnings
from datetime import UTC, datetime
from typing import Any

import feedparser

from newsrag.config import SourceConfig
from newsrag.models import FetchedVia, Item
from newsrag.sources.http import HttpFetcher
from newsrag.sources.parsing import clean_text, from_struct_time, parse_date

log = logging.getLogger("newsrag.sources.rss")


class RSSAdapter:
    def __init__(self, config: SourceConfig) -> None:
        if config.type != "rss" or not config.url:
            raise ValueError(f"{config.name!r} is not an rss source")
        self.config = config

    async def fetch(self, http: HttpFetcher, since: datetime) -> list[Item]:
        assert self.config.url is not None
        resp = await http.get(self.config.url)
        return self.parse(resp.content, now=datetime.now(UTC))

    def parse(self, content: bytes, now: datetime) -> list[Item]:
        with warnings.catch_warnings():
            # feedparser warns about its updated_parsed -> published_parsed fallback;
            # we read both fields explicitly, so the warning is noise here.
            warnings.simplefilter("ignore", DeprecationWarning)
            return self._parse(content, now)

    def _parse(self, content: bytes, now: datetime) -> list[Item]:
        feed = feedparser.parse(content)
        if feed.bozo and not feed.entries:
            raise ValueError(f"Feed could not be parsed: {feed.get('bozo_exception')}")
        items: list[Item] = []
        for entry in feed.entries:
            item = self._to_item(entry, now)
            if item is not None:
                items.append(item)
        return items

    def _to_item(self, entry: Any, now: datetime) -> Item | None:
        title = clean_text(entry.get("title"))
        url = entry.get("link") or entry.get("id")
        if not title or not url:
            return None
        published = (
            from_struct_time(entry.get("published_parsed"))
            or from_struct_time(entry.get("updated_parsed"))
            or parse_date(entry.get("published"), self.config.timezone)
            or parse_date(entry.get("updated"), self.config.timezone)
        )
        tags: dict[str, str] = {}
        if published is None:
            published = now
            tags["date_estimated"] = "true"
        content = entry.get("content")
        body_raw = content[0].get("value") if content else entry.get("summary")
        try:
            return Item(
                title=title,
                body=clean_text(body_raw),
                url=url,
                source=self.config.name,
                region=self.config.region,
                category=self.config.category,
                published_at=published,
                fetched_via=FetchedVia.RSS,
                tags=tags,
            )
        except ValueError as exc:
            log.debug("Skipping entry from %s: %s", self.config.name, exc)
            return None
