"""Optional free-tier news APIs that need a key (REQUIREMENTS section 4).

These adapters are only built when the key is present in the in-memory KeyStore.
They have been written against the providers' published response shapes and tested
with mocked responses; they have not yet been run against the live APIs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from newsrag.config import SourceConfig
from newsrag.models import FetchedVia, Item
from newsrag.sources.http import HttpFetcher
from newsrag.sources.parsing import clean_text, parse_date

log = logging.getLogger("newsrag.sources.keyed")


def _item(
    config: SourceConfig,
    *,
    title: Any,
    url: Any,
    body: Any,
    published: datetime | None,
    source: Any,
) -> Item | None:
    title_s = clean_text(title if isinstance(title, str) else None)
    if not title_s or not isinstance(url, str) or published is None:
        return None
    try:
        return Item(
            title=title_s,
            body=clean_text(body if isinstance(body, str) else None),
            url=url,
            source=source if isinstance(source, str) and source else config.name,
            region=config.region,
            category=config.category,
            published_at=published,
            fetched_via=FetchedVia.API,
            tags={"via": config.name},
        )
    except ValueError as exc:
        log.debug("Skipping %s article: %s", config.api, exc)
        return None


class GNewsAdapter:
    """gnews.io v4 search. The API takes the key as an `apikey` query parameter."""

    URL = "https://gnews.io/api/v4/search"

    def __init__(self, config: SourceConfig, key: str) -> None:
        self.config = config
        self._key = key

    async def fetch(self, http: HttpFetcher, since: datetime) -> list[Item]:
        params = {
            "q": self.config.query or "",
            "lang": "en",
            "max": "10",
            "from": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            **self.config.params,
            "apikey": self._key,
        }
        resp = await http.get(self.URL, params=params)
        return self.parse(resp.json())

    def parse(self, data: dict[str, Any]) -> list[Item]:
        out = []
        for a in data.get("articles", []):
            item = _item(
                self.config,
                title=a.get("title"),
                url=a.get("url"),
                body=" ".join(x for x in (a.get("description"), a.get("content")) if x),
                published=parse_date(a.get("publishedAt")),
                source=(a.get("source") or {}).get("name"),
            )
            if item:
                out.append(item)
        return out


class NewsDataAdapter:
    """newsdata.io latest-news endpoint. The key is sent in the X-ACCESS-KEY header."""

    URL = "https://newsdata.io/api/1/latest"

    def __init__(self, config: SourceConfig, key: str) -> None:
        self.config = config
        self._key = key

    async def fetch(self, http: HttpFetcher, since: datetime) -> list[Item]:
        params = {"q": self.config.query or "", "language": "en", **self.config.params}
        resp = await http.get(self.URL, params=params, headers={"X-ACCESS-KEY": self._key})
        return self.parse(resp.json())

    def parse(self, data: dict[str, Any]) -> list[Item]:
        out = []
        for a in data.get("results", []) or []:
            item = _item(
                self.config,
                title=a.get("title"),
                url=a.get("link"),
                body=" ".join(x for x in (a.get("description"), a.get("content")) if x),
                published=parse_date(a.get("pubDate"), "UTC"),
                source=a.get("source_name") or a.get("source_id"),
            )
            if item:
                out.append(item)
        return out


KEYED_ADAPTERS: dict[str, type[GNewsAdapter] | type[NewsDataAdapter]] = {
    "gnews": GNewsAdapter,
    "newsdata": NewsDataAdapter,
}
