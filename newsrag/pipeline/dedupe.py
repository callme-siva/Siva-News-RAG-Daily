"""Merge duplicates within one fetch (DD1, DD3 within-batch, FR6).

Two items are the same story if they share a normalised URL, or if their headlines'
significant words overlap by at least `threshold` (Jaccard). The richest version
(longest body) is kept and every other outlet and URL is recorded on it.
Cross-day near-duplicates are handled against the store in stage 4.
"""

from __future__ import annotations

import re

from newsrag.models import Item

_WORD = re.compile(r"[a-z0-9]+")
MIN_WORD_LEN = 4
MIN_SIGNIFICANT_WORDS = 3


def headline_words(title: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(title.lower()) if len(w) >= MIN_WORD_LEN)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def same_story(a: frozenset[str], b: frozenset[str], threshold: float) -> bool:
    if len(a) < MIN_SIGNIFICANT_WORDS or len(b) < MIN_SIGNIFICANT_WORDS:
        return False
    return jaccard(a, b) >= threshold


class _Group:
    def __init__(self, item: Item) -> None:
        self.primary = item
        self.words = headline_words(item.title)
        self.sources: dict[str, None] = {}
        self.urls: dict[str, None] = {}

    def absorb(self, other: Item) -> None:
        if other.source != self.primary.source:
            self.sources.setdefault(other.source, None)
        if other.url != self.primary.url:
            self.urls.setdefault(other.url, None)
        for src in other.also_reported_by:
            if src != self.primary.source:
                self.sources.setdefault(src, None)
        for url in other.also_reported_urls:
            if url != self.primary.url:
                self.urls.setdefault(url, None)

    def result(self) -> Item:
        return self.primary.model_copy(
            update={
                "also_reported_by": tuple(sorted(self.sources)),
                "also_reported_urls": tuple(sorted(self.urls)),
            }
        )


def dedupe_batch(items: list[Item], threshold: float) -> tuple[list[Item], int]:
    """Return (unique items, number merged away). Order-independent for ties."""
    ordered = sorted(items, key=lambda i: (-len(i.body), i.published_at, i.item_id))
    groups: list[_Group] = []
    by_url: dict[str, _Group] = {}
    merged = 0
    for item in ordered:
        group = by_url.get(item.item_id)
        if group is None:
            words = headline_words(item.title)
            group = next((g for g in groups if same_story(g.words, words, threshold)), None)
        if group is None:
            group = _Group(item)
            groups.append(group)
        else:
            group.absorb(item)
            merged += 1
        by_url.setdefault(item.item_id, group)
    return [g.result() for g in groups], merged
