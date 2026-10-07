"""Drop stale and off-topic items (FR4). Deterministic: same input, same output."""

from __future__ import annotations

from datetime import datetime, timedelta
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from newsrag.config import AppConfig
from newsrag.models import Item


class FilterStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kept: int = 0
    too_old: int = 0
    excluded: int = 0
    not_relevant: int = 0
    unknown_category: int = 0


def is_official(url: str, official_domains: list[str]) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in official_domains)


def filter_items(
    items: list[Item], cfg: AppConfig, *, now: datetime, max_age_hours: int
) -> tuple[list[Item], FilterStats]:
    """Keep items that are recent, not excluded, and match their category's include
    pattern (official sources always pass include)."""
    cutoff = now - timedelta(hours=max_age_hours)
    patterns = {c.name: (c.include_re(), c.exclude_re()) for c in cfg.categories}
    stats = FilterStats()
    kept: list[Item] = []
    for item in items:
        if item.published_at < cutoff:
            stats.too_old += 1
            continue
        if item.category not in patterns:
            stats.unknown_category += 1
            continue
        include, exclude = patterns[item.category]
        text = f"{item.title} {item.body}"
        if exclude and exclude.search(text):
            stats.excluded += 1
            continue
        if include and not is_official(item.url, cfg.official_domains) and not include.search(text):
            stats.not_relevant += 1
            continue
        kept.append(item)
    stats.kept = len(kept)
    return kept, stats
