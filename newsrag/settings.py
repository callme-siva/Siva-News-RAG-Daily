"""User settings, persisted as `settings.json` in the workspace (REQUIREMENTS section 6).

Settings never hold API keys (R3): there is no field for one, unknown fields are rejected,
and `save_settings` refuses to write anything that looks like a secret.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from newsrag.secrets import looks_like_secret

FONT_SIZES_PX: dict[str, int] = {"A": 16, "A+": 18, "A++": 20}


class _Group(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Appearance(_Group):
    font_size: Literal["A", "A+", "A++"] = "A"
    theme: Literal["light", "dark", "system"] = "system"
    high_contrast: bool = False

    @property
    def font_px(self) -> int:
        return FONT_SIZES_PX[self.font_size]


class LLMSettings(_Group):
    mode: Literal["auto", "rules", "llm"] = "auto"
    provider: Literal["anthropic", "gemini", "openai_compatible", "ollama", "local_openai"] = (
        "ollama"
    )
    base_url: str | None = "http://localhost:11434"
    model_processing: str | None = None
    model_chat: str | None = None
    model_digest: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=64, le=16384)
    context_length: int | None = Field(default=None, ge=512)
    timeout_s: int = Field(default=120, ge=5, le=600)
    concurrency: int = Field(default=1, ge=1, le=16)


class RetrievalSettings(_Group):
    search_mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    top_k: int = Field(default=8, ge=1, le=50)
    candidates_k: int = Field(default=30, ge=1, le=200)
    keyword_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_chunks_per_article: int = Field(default=2, ge=1, le=10)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    chunk_size: int = Field(default=1600, ge=200, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=2000)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    memory_turns: int = Field(default=6, ge=0, le=50)

    @model_validator(mode="after")
    def _consistent(self) -> RetrievalSettings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.top_k > self.candidates_k:
            raise ValueError("top_k must not exceed candidates_k")
        return self


class SourceSettings(_Group):
    regions: list[str] = Field(default_factory=lambda: ["US", "EU", "IN"])
    max_age_hours: int = Field(default=48, ge=1, le=24 * 30)
    max_per_group: int = Field(default=15, ge=1, le=500)
    max_catchup_days: int = Field(default=7, ge=1, le=60)
    dedupe_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    dedupe_window_days: int = Field(default=3, ge=0, le=60)
    embed_dup_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    on_update: Literal["ignore", "replace"] = "ignore"
    fetch_full_text: bool = False


class BriefingSettings(_Group):
    items_per_category: int = Field(default=5, ge=1, le=20)
    style: Literal["brief", "detailed"] = "brief"
    email_enabled: bool = False


class DataSettings(_Group):
    retention_days: int | None = Field(default=90, ge=1)
    auto_cleanup: bool = False


class SafetySettings(_Group):
    show_provenance: bool = True


class Settings(_Group):
    appearance: Appearance = Field(default_factory=Appearance)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    sources: SourceSettings = Field(default_factory=SourceSettings)
    briefing: BriefingSettings = Field(default_factory=BriefingSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)


SETTINGS_FILE = "settings.json"


class SecretInSettingsError(ValueError):
    """Raised when a settings value looks like an API key. Keys never go to disk (R3)."""


def _walk_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _walk_strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _walk_strings(v)]
    return []


def load_settings(workspace_dir: Path) -> Settings:
    """Load settings from the workspace, or defaults if the file does not exist yet."""
    path = workspace_dir / SETTINGS_FILE
    if not path.exists():
        return Settings()
    return Settings.model_validate_json(path.read_text("utf-8"))


def save_settings(workspace_dir: Path, settings: Settings) -> Path:
    data = settings.model_dump(mode="json")
    if any(looks_like_secret(s) for s in _walk_strings(data)):
        raise SecretInSettingsError("A settings value looks like an API key; refusing to save")
    path = workspace_dir / SETTINGS_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), "utf-8")
    tmp.replace(path)
    return path
