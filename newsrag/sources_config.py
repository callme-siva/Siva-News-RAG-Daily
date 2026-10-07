"""Per-workspace source changes on top of the shipped config (REQUIREMENTS FR32-FR37).

The shipped `config.yaml` is never edited. A workspace stores only its differences in
`sources.json`: sources the user added, and on/off overrides by name. Removing an override
restores the default.
"""

from __future__ import annotations

import asyncio
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import feedparser
from pydantic import BaseModel, ConfigDict, Field

from newsrag.config import AppConfig, SourceConfig, load_config
from newsrag.sources.http import BlockedByRobots, HttpFetcher
from newsrag.sources.parsing import clean_text

OVERRIDES_FILE = "sources.json"


class SourceOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: dict[str, bool] = Field(default_factory=dict)
    added: list[SourceConfig] = Field(default_factory=list)


def load_overrides(root: Path) -> SourceOverrides:
    path = root / OVERRIDES_FILE
    if not path.exists():
        return SourceOverrides()
    return SourceOverrides.model_validate_json(path.read_text("utf-8"))


def save_overrides(root: Path, overrides: SourceOverrides) -> None:
    path = root / OVERRIDES_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(overrides.model_dump_json(indent=2), "utf-8")
    tmp.replace(path)


def effective_config(root: Path | None, base: AppConfig | None = None) -> AppConfig:
    """Shipped config + this workspace's added sources and on/off overrides."""
    base = base or load_config()
    if root is None:
        return base
    ov = load_overrides(root)
    shipped = {s.name for s in base.sources}
    sources = [s for s in base.sources] + [a for a in ov.added if a.name not in shipped]
    sources = [
        s.model_copy(update={"enabled": ov.enabled[s.name]}) if s.name in ov.enabled else s
        for s in sources
    ]
    return AppConfig.model_validate(
        {**base.model_dump(), "sources": [s.model_dump() for s in sources]}
    )


def is_user_added(root: Path, name: str) -> bool:
    return any(a.name == name for a in load_overrides(root).added)


def set_enabled(root: Path, name: str, enabled: bool, base: AppConfig | None = None) -> None:
    base = base or load_config()
    ov = load_overrides(root)
    shipped = {s.name: s.enabled for s in base.sources}
    added = {a.name: a for a in ov.added}
    if name in added:
        ov.added = [
            a.model_copy(update={"enabled": enabled}) if a.name == name else a for a in ov.added
        ]
    elif name in shipped:
        if shipped[name] == enabled:
            ov.enabled.pop(name, None)
        else:
            ov.enabled[name] = enabled
    else:
        raise KeyError(name)
    save_overrides(root, ov)


def add_source(root: Path, source: SourceConfig, base: AppConfig | None = None) -> None:
    """Add a user source. Validates against the effective config (unique name, known region
    and category) before saving."""
    cfg = effective_config(root, base)
    AppConfig.model_validate(
        {
            **cfg.model_dump(),
            "sources": [*(s.model_dump() for s in cfg.sources), source.model_dump()],
        }
    )
    ov = load_overrides(root)
    ov.added.append(source)
    save_overrides(root, ov)


def remove_source(root: Path, name: str) -> None:
    """Remove a user-added source. Shipped sources can only be disabled."""
    ov = load_overrides(root)
    if not any(a.name == name for a in ov.added):
        raise KeyError(f"{name!r} is not a user-added source")
    ov.added = [a for a in ov.added if a.name != name]
    ov.enabled.pop(name, None)
    save_overrides(root, ov)


def reset_sources(root: Path) -> None:
    (root / OVERRIDES_FILE).unlink(missing_ok=True)


# ---------- feed validation (FR33) ----------


class FeedCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    url: str
    title: str = ""
    headlines: list[str] = Field(default_factory=list)
    entries: int = 0
    error: str | None = None


async def check_feed(url: str, http: HttpFetcher | None = None) -> FeedCheck:
    """robots.txt check, fetch, parse. Returns the feed title and 3 latest headlines."""
    if http is None:
        async with HttpFetcher(attempts=2) as own:
            return await _check(url, own)
    return await _check(url, http)


async def _check(url: str, http: HttpFetcher) -> FeedCheck:
    try:
        resp = await http.get(url)
    except BlockedByRobots:
        return FeedCheck(ok=False, url=url, error="the site's robots.txt does not allow this URL")
    except Exception as exc:
        return FeedCheck(ok=False, url=url, error=f"could not fetch: {type(exc).__name__}")
    feed = feedparser.parse(resp.content)
    if not feed.entries:
        return FeedCheck(
            ok=False, url=url, error="no feed entries found (is this an RSS or Atom URL?)"
        )
    return FeedCheck(
        ok=True,
        url=url,
        title=clean_text(feed.feed.get("title")) or url,
        headlines=[clean_text(e.get("title")) for e in feed.entries[:3]],
        entries=len(feed.entries),
    )


def check_feed_sync(url: str) -> FeedCheck:
    return asyncio.run(check_feed(url))


# ---------- OPML (FR34) ----------


def export_opml(cfg: AppConfig) -> str:
    root = ET.Element("opml", version="2.0")
    head = ET.SubElement(root, "head")
    ET.SubElement(head, "title").text = "newsrag sources"
    body = ET.SubElement(root, "body")
    for s in cfg.sources:
        if s.type != "rss" or not s.url:
            continue
        ET.SubElement(
            body,
            "outline",
            type="rss",
            text=s.name,
            title=s.name,
            xmlUrl=s.url,
            category=f"{s.region}/{s.category}",
        )
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def parse_opml(text: str, default_region: str, default_category: str) -> list[SourceConfig]:
    """Read RSS outlines. `category="REGION/Category"` is honoured when present."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"not valid OPML: {exc}") from exc
    out: list[SourceConfig] = []
    for node in root.iter("outline"):
        url = node.get("xmlUrl")
        if not url:
            continue
        region, category = default_region, default_category
        tag = node.get("category", "")
        if "/" in tag:
            region, category = tag.split("/", 1)
        out.append(
            SourceConfig(
                name=(node.get("title") or node.get("text") or url)[:80],
                type="rss",
                url=url,
                region=region,
                category=category,
            )
        )
    return out


def opml_json_preview(sources: list[SourceConfig]) -> str:
    return json.dumps([s.model_dump(exclude_defaults=True) for s in sources], indent=2)
