"""Core data models shared by every layer (REQUIREMENTS FR2, FR8, section 11).

`Item` is generic so the same store can later hold call transcripts or other daily data.
News-specific fields (region, category) are plain strings validated against config.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from newsrag.ids import content_hash, item_id_for_url, normalise_url


def utc_now() -> datetime:
    return datetime.now(UTC)


class EngineName(StrEnum):
    """Which engine produced a piece of text. Shown as a provenance label (R6)."""

    RULES = "rules"
    LLM = "llm"


class FetchedVia(StrEnum):
    RSS = "rss"
    GOOGLE_NEWS = "google_news"
    GDELT = "gdelt"
    API = "api"
    FILE = "file"


class Item(BaseModel):
    """One collected item (a news article in v1), normalised from any source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(min_length=1)
    body: str = ""
    url: str
    source: str = Field(min_length=1)
    region: str = Field(min_length=1)
    category: str = Field(min_length=1)
    published_at: datetime
    fetched_via: FetchedVia
    kind: str = "news"
    also_reported_by: tuple[str, ...] = ()
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def _canonical_url(cls, v: str) -> str:
        return normalise_url(v)

    @field_validator("published_at")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=UTC)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def item_id(self) -> str:
        return item_id_for_url(self.url)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        return content_hash(f"{self.title}\n{self.body}")


class Entities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    companies: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    organisations: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)

    def flat(self) -> list[str]:
        seen: dict[str, None] = {}
        for name in (*self.companies, *self.organisations, *self.people, *self.countries):
            seen.setdefault(name, None)
        return list(seen)


class Processed(BaseModel):
    """Output of `Engine.process` for one item (FR8). Same shape for both engines."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    relevant: bool
    category: str
    summary: str
    key_facts: list[str] = Field(default_factory=list, max_length=5)
    entities: Entities = Field(default_factory=Entities)
    engine: EngineName
    fallback_reason: str | None = None


class Chunk(BaseModel):
    """A retrievable piece of an item. `text` carries its own header so it is citable alone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    item_id: str
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    """Result of one fetch run (DD7, FR24)."""

    model_config = ConfigDict(extra="forbid")

    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    new: int = 0
    skipped_seen: int = 0
    merged_duplicate: int = 0
    updated: int = 0
    failed: int = 0
    errors: list[str] = Field(default_factory=list)
