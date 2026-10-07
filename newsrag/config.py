"""Load and validate `config.yaml`: regions, categories, sources (REQUIREMENTS R9, FR36).

The shipped defaults live inside the package. A user file can replace them via
`load_config(path)`. Workspace-level source edits are layered on top in stage 2.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Google News RSS is not a source type: news.google.com/robots.txt disallows every /rss path.
SourceType = Literal["rss", "gdelt", "api"]


class Region(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=r"^[A-Z]{2,3}$")
    name: str = Field(min_length=1)


class Category(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    include: str | None = None
    exclude: str | None = None

    @field_validator("include", "exclude")
    @classmethod
    def _valid_regex(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                re.compile(v, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"Invalid regex {v!r}: {exc}") from exc
        return v

    def include_re(self) -> re.Pattern[str] | None:
        return re.compile(self.include, re.IGNORECASE) if self.include else None

    def exclude_re(self) -> re.Pattern[str] | None:
        return re.compile(self.exclude, re.IGNORECASE) if self.exclude else None


class SourceConfig(BaseModel):
    """One configured source. `url` is used by rss; `query` by gdelt and api sources;
    `api` names the key-based adapter (gnews, newsdata). `timezone` (IANA name) is applied
    to dates the feed publishes without one. `params` are extra adapter query parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    type: SourceType
    region: str
    category: str
    url: str | None = None
    query: str | None = None
    api: str | None = None
    timezone: str | None = None
    params: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    note: str | None = None

    @model_validator(mode="after")
    def _fields_for_type(self) -> SourceConfig:
        needed = {"rss": "url", "gdelt": "query", "api": "api"}[self.type]
        if not getattr(self, needed):
            raise ValueError(f"Source {self.name!r} of type {self.type!r} needs {needed!r}")
        if self.timezone:
            try:
                ZoneInfo(self.timezone)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError(f"Source {self.name!r}: unknown timezone") from exc
        return self


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    regions: list[Region] = Field(min_length=1)
    categories: list[Category] = Field(min_length=1)
    official_domains: list[str] = Field(default_factory=list)
    sources: list[SourceConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references_exist(self) -> AppConfig:
        regions = {r.code for r in self.regions}
        categories = {c.name for c in self.categories}
        if len(regions) != len(self.regions):
            raise ValueError("Duplicate region codes")
        if len(categories) != len(self.categories):
            raise ValueError("Duplicate category names")
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate source names")
        for s in self.sources:
            if s.region not in regions:
                raise ValueError(f"Source {s.name!r}: unknown region {s.region!r}")
            if s.category not in categories:
                raise ValueError(f"Source {s.name!r}: unknown category {s.category!r}")
        return self

    def region_codes(self) -> list[str]:
        return [r.code for r in self.regions]

    def category(self, name: str) -> Category:
        for c in self.categories:
            if c.name == name:
                return c
        raise KeyError(name)


def default_config_text() -> str:
    return resources.files("newsrag.defaults").joinpath("config.yaml").read_text("utf-8")


def load_config(path: Path | None = None) -> AppConfig:
    """Load config from `path`, or the shipped defaults when `path` is None."""
    text = path.read_text("utf-8") if path else default_config_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a mapping at the top level")
    return AppConfig.model_validate(data)
