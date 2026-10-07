from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from newsrag.models import EngineName, Entities, FetchedVia, Item, Processed


def _item(**overrides: object) -> Item:
    data: dict[str, object] = {
        "title": "Fixture headline",
        "body": "Fixture body text.",
        "url": "https://www.example.com/story/?utm_source=feed",
        "source": "Example",
        "region": "IN",
        "category": "Finance",
        "published_at": datetime(2026, 10, 6, 9, 0),
        "fetched_via": FetchedVia.RSS,
    }
    data.update(overrides)
    return Item.model_validate(data)


def test_item_url_is_canonical_and_id_is_stable() -> None:
    a = _item()
    b = _item(url="http://example.com/story#top")
    assert a.url == "https://example.com/story"
    assert a.item_id == b.item_id


def test_naive_datetimes_become_utc() -> None:
    assert _item().published_at.tzinfo == UTC


def test_item_rejects_unknown_fields_and_empty_title() -> None:
    with pytest.raises(ValidationError):
        _item(unexpected="x")
    with pytest.raises(ValidationError):
        _item(title="")


def test_item_dump_includes_ids() -> None:
    dumped = _item().model_dump()
    assert {"item_id", "content_hash"} <= dumped.keys()


def test_processed_limits_key_facts() -> None:
    with pytest.raises(ValidationError):
        Processed(
            item_id="x",
            relevant=True,
            category="Finance",
            summary="s",
            key_facts=["1", "2", "3", "4", "5", "6"],
            engine=EngineName.RULES,
        )


def test_entities_flat_dedupes_in_order() -> None:
    e = Entities(companies=["Acme"], organisations=["Acme", "Agency"], people=["A. Person"])
    assert e.flat() == ["Acme", "Agency", "A. Person"]
