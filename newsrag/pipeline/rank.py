"""Rank and cap per region and category (FR7)."""

from __future__ import annotations

from collections import defaultdict

from newsrag.models import Item


def rank_key(item: Item) -> tuple[int, float, str]:
    """More outlets first, then newest, then a stable tiebreak."""
    return (-len(item.also_reported_by), -item.published_at.timestamp(), item.item_id)


def rank_and_cap(items: list[Item], max_per_group: int) -> list[Item]:
    groups: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for item in items:
        groups[(item.region, item.category)].append(item)
    out: list[Item] = []
    for key in sorted(groups):
        out.extend(sorted(groups[key], key=rank_key)[:max_per_group])
    return out
