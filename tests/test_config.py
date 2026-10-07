from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from newsrag.config import load_config


def test_shipped_defaults_load() -> None:
    cfg = load_config()
    assert cfg.region_codes() == ["US", "EU", "IN"]
    assert {c.name for c in cfg.categories} == {"Technology", "Finance", "Politics"}
    regions_covered = {(s.region, s.category) for s in cfg.sources if s.enabled}
    assert len(regions_covered) == 9, "every region x category has at least one enabled source"
    assert all(s.type != "gdelt" or not s.enabled for s in cfg.sources)
    assert not any("news.google.com" in (s.url or "") for s in cfg.sources)


def test_politics_exclude_does_not_drop_diplomacy() -> None:
    """Regression for the n8n bug: 'ipl' must not match inside 'diplomat' or 'multiple'."""
    politics = load_config().category("Politics")
    exclude = politics.exclude_re()
    include = politics.include_re()
    assert exclude is not None and include is not None
    text = "Diplomats from multiple countries met at the summit"
    assert not exclude.search(text)
    assert include.search(text)
    assert exclude.search("IPL final draws record crowd")


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(body, "utf-8")
    return p


BASE = """
regions: [{code: IN, name: India}]
categories: [{name: Finance}]
"""


def test_source_must_reference_known_region_and_category(tmp_path: Path) -> None:
    src = "{name: X, type: rss, url: 'https://e.com/f', region: US, category: Finance}"
    bad = BASE + f"sources: [{src}]"
    with pytest.raises(ValidationError, match="unknown region"):
        load_config(_write(tmp_path, bad))


def test_source_needs_field_for_its_type(tmp_path: Path) -> None:
    bad = BASE + "sources: [{name: X, type: gdelt, region: IN, category: Finance}]"
    with pytest.raises(ValidationError, match="needs 'query'"):
        load_config(_write(tmp_path, bad))


def test_invalid_regex_is_rejected(tmp_path: Path) -> None:
    bad = "regions: [{code: IN, name: India}]\ncategories: [{name: F, include: '(unclosed'}]\n"
    with pytest.raises(ValidationError, match="Invalid regex"):
        load_config(_write(tmp_path, bad))


def test_duplicate_source_names_rejected(tmp_path: Path) -> None:
    src = "{name: X, type: rss, url: 'https://e.com/f', region: IN, category: Finance}"
    with pytest.raises(ValidationError, match="Duplicate source"):
        load_config(_write(tmp_path, BASE + f"sources: [{src}, {src}]"))
