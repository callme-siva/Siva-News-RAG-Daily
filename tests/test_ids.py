from __future__ import annotations

import pytest

from newsrag.ids import chunk_id, content_hash, item_id_for_key, item_id_for_url, normalise_url

CANONICAL = "https://example.com/news/story-1"


@pytest.mark.parametrize(
    "variant",
    [
        "https://example.com/news/story-1",
        "http://example.com/news/story-1",
        "https://www.example.com/news/story-1",
        "https://EXAMPLE.com/news/story-1/",
        "https://example.com/news/story-1#comments",
        "https://example.com/news/story-1?utm_source=x&utm_medium=y",
        "https://example.com:443/news/story-1?fbclid=abc&ref=home",
        "https://example.com/news/story-1?maca=en-rss-en-all-1573-rdf",
        "  https://www.example.com/news/story-1/?gclid=1#top  ",
    ],
)
def test_url_variants_normalise_to_one_form(variant: str) -> None:
    assert normalise_url(variant) == CANONICAL
    assert item_id_for_url(variant) == item_id_for_url(CANONICAL)


def test_meaningful_query_params_are_kept_and_sorted() -> None:
    a = normalise_url("https://example.com/a?id=2&page=1&utm_campaign=z")
    b = normalise_url("https://example.com/a?page=1&id=2")
    assert a == b == "https://example.com/a?id=2&page=1"


def test_different_articles_get_different_ids() -> None:
    assert item_id_for_url("https://example.com/a") != item_id_for_url("https://example.com/b")


def test_path_case_is_preserved() -> None:
    assert normalise_url("https://example.com/News/A") == "https://example.com/News/A"


@pytest.mark.parametrize("bad", ["", "   ", "ftp://example.com/x", "not a url", "https://"])
def test_invalid_urls_raise(bad: str) -> None:
    with pytest.raises(ValueError):
        normalise_url(bad)


def test_chunk_id_is_deterministic() -> None:
    assert chunk_id("abc", 0) == "abc:0"
    with pytest.raises(ValueError):
        chunk_id("abc", -1)


def test_natural_key_ids() -> None:
    a = item_id_for_key("gold", "2026-10-07", "source-a")
    assert a == item_id_for_key("gold", "2026-10-07", "source-a")
    assert a != item_id_for_key("gold", "2026-10-08", "source-a")
    # Separator prevents ("ab", "c") colliding with ("a", "bc").
    assert item_id_for_key("ab", "c") != item_id_for_key("a", "bc")
    with pytest.raises(ValueError):
        item_id_for_key("gold", "")


def test_content_hash_ignores_whitespace_only_changes() -> None:
    assert content_hash("Rates  held\nsteady.") == content_hash("Rates held steady.")
    assert content_hash("Rates held steady.") != content_hash("Rates cut.")
