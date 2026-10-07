from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from newsrag.settings import (
    SETTINGS_FILE,
    RetrievalSettings,
    SecretInSettingsError,
    Settings,
    load_settings,
    save_settings,
)


def test_defaults_match_requirements() -> None:
    s = Settings()
    assert s.appearance.font_size == "A" and s.appearance.font_px == 16
    assert s.appearance.theme == "system"
    assert s.llm.mode == "auto"
    r = s.retrieval
    assert (r.search_mode, r.top_k, r.candidates_k, r.rerank) == ("hybrid", 8, 30, True)
    assert (r.chunk_size, r.chunk_overlap) == (1600, 200)
    assert s.sources.dedupe_window_days == 3 and s.sources.on_update == "ignore"
    assert s.data.retention_days == 90


def test_font_sizes() -> None:
    s = Settings()
    s.appearance.font_size = "A++"
    assert s.appearance.font_px == 20
    with pytest.raises(ValidationError):
        s.appearance.font_size = "A+++"  # type: ignore[assignment]


def test_roundtrip(tmp_path: Path) -> None:
    s = Settings()
    s.retrieval.top_k = 5
    s.appearance.theme = "dark"
    save_settings(tmp_path, s)
    assert load_settings(tmp_path) == s


def test_missing_file_gives_defaults(tmp_path: Path) -> None:
    assert load_settings(tmp_path) == Settings()


def test_unknown_fields_rejected_so_no_key_field_can_sneak_in(tmp_path: Path) -> None:
    (tmp_path / SETTINGS_FILE).write_text('{"llm": {"api_key": "x"}}', "utf-8")
    with pytest.raises(ValidationError):
        load_settings(tmp_path)


def test_settings_have_no_key_like_fields() -> None:
    def names(model: type[BaseModel]) -> list[str]:
        out: list[str] = []
        for name, field in model.model_fields.items():
            out.append(name)
            ann = field.annotation
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                out.extend(names(ann))
        return out

    secret_name = re.compile(r"(^|_)(api_?key|key|secret|password|token|credentials?)$")
    assert not [n for n in names(Settings) if secret_name.search(n)]


def test_save_refuses_secret_shaped_values(tmp_path: Path) -> None:
    s = Settings()
    s.llm.model_chat = "sk-ant-api03-" + "x" * 30
    with pytest.raises(SecretInSettingsError):
        save_settings(tmp_path, s)
    assert not (tmp_path / SETTINGS_FILE).exists()


@pytest.mark.parametrize(
    "kwargs",
    [{"chunk_size": 500, "chunk_overlap": 500}, {"top_k": 40, "candidates_k": 30}],
)
def test_inconsistent_retrieval_settings_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        RetrievalSettings(**kwargs)
