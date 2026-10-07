"""The one interface every source implements (REQUIREMENTS R8)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from newsrag.config import SourceConfig
from newsrag.models import Item
from newsrag.sources.http import HttpFetcher

SourceStatus = Literal["ok", "empty", "error", "skipped"]


class SourceResult(BaseModel):
    """What one source produced in one run. `error` is already redacted."""

    model_config = ConfigDict(extra="forbid")

    source: str
    status: SourceStatus
    count: int = 0
    items: list[Item] = Field(default_factory=list)
    error: str | None = None
    elapsed_s: float = 0.0


class SourceAdapter(Protocol):
    config: SourceConfig

    async def fetch(self, http: HttpFetcher, since: datetime) -> list[Item]:
        """Return normalised items published after `since` where the source allows it.
        Raise on failure; the pipeline turns exceptions into an error result."""
        ...
