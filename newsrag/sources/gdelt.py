"""GDELT DOC 2.0 API (no key). Headlines only, no body text.

GDELT allows one request every 5 seconds per client, so the fetcher spaces requests
to its host (see `MIN_INTERVAL_S`) and retries 429 with backoff.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from newsrag.config import SourceConfig
from newsrag.models import FetchedVia, Item
from newsrag.sources.http import HttpFetcher
from newsrag.sources.parsing import clean_text, parse_date

log = logging.getLogger("newsrag.sources.gdelt")

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
HOST = "api.gdeltproject.org"
MIN_INTERVAL_S = 6.0
MAX_RECORDS = 50


class GdeltAdapter:
    def __init__(self, config: SourceConfig) -> None:
        if config.type != "gdelt" or not config.query:
            raise ValueError(f"{config.name!r} is not a gdelt source")
        self.config = config

    async def fetch(self, http: HttpFetcher, since: datetime) -> list[Item]:
        assert self.config.query is not None
        params = {
            "query": self.config.query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(MAX_RECORDS),
            "timespan": self.config.params.get("timespan", "2d"),
            "sort": "DateDesc",
        }
        resp = await http.get(API_URL, params=params)
        text = resp.text.strip()
        if not text.startswith("{"):
            raise ValueError(f"GDELT returned a non-JSON response: {text[:120]}")
        return self.parse(resp.json())

    def parse(self, data: dict[str, Any]) -> list[Item]:
        items: list[Item] = []
        for article in data.get("articles", []):
            title = clean_text(article.get("title"))
            url = article.get("url")
            published = parse_date(article.get("seendate"))
            if not title or not url or published is None:
                continue
            if article.get("language", "English") != "English":
                continue
            try:
                items.append(
                    Item(
                        title=title,
                        url=url,
                        source=article.get("domain") or self.config.name,
                        region=self.config.region,
                        category=self.config.category,
                        published_at=published,
                        fetched_via=FetchedVia.GDELT,
                        tags={"via": self.config.name},
                    )
                )
            except ValueError as exc:
                log.debug("Skipping GDELT article: %s", exc)
        return items
