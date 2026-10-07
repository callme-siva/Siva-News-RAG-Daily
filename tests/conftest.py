from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from newsrag.secrets import KEYS


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep every test away from the real app-config folder and real keys."""
    monkeypatch.setenv("NEWSRAG_CONFIG_DIR", str(tmp_path / "_appconfig"))
    KEYS.clear()
    yield
    KEYS.clear()
