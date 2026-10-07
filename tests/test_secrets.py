from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pytest

from newsrag.logging_setup import setup_logging
from newsrag.secrets import REDACTED, KeyStore, looks_like_secret, redact

FAKE_ANTHROPIC = "sk-ant-api03-" + "A1b2C3d4" * 5
FAKE_GEMINI = "AIza" + "Zx9_" * 9
FAKE_PLAIN = "plainkeyvalue12345"


def test_env_loading_and_names() -> None:
    store = KeyStore()
    found = store.load_from_env(
        {"ANTHROPIC_API_KEY": FAKE_ANTHROPIC, "GOOGLE_API_KEY": FAKE_GEMINI, "OTHER": "x"}
    )
    assert found == ["anthropic", "gemini"]
    assert store.get("anthropic") == FAKE_ANTHROPIC
    assert not store.has("gnews")


def test_blank_key_removes_entry() -> None:
    store = KeyStore()
    store.set("anthropic", FAKE_ANTHROPIC)
    store.set("anthropic", "  ")
    assert not store.has("anthropic")


def test_store_never_reveals_values() -> None:
    store = KeyStore()
    store.set("anthropic", FAKE_ANTHROPIC)
    assert FAKE_ANTHROPIC not in repr(store) and FAKE_ANTHROPIC not in str(store)
    with pytest.raises(TypeError):
        pickle.dumps(store)


@pytest.mark.parametrize(
    "text",
    [
        f"auth failed for {FAKE_ANTHROPIC}",
        f"GET https://api.example.com/v1?apiKey={FAKE_PLAIN}&q=rbi",
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        f"key {FAKE_GEMINI} rejected",
    ],
)
def test_pattern_redaction(text: str) -> None:
    out = redact(text)
    assert REDACTED in out
    for secret in (FAKE_ANTHROPIC, FAKE_PLAIN, FAKE_GEMINI, "abcdefghijklmnopqrstuvwxyz"):
        assert secret not in out


def test_known_value_redaction_even_without_pattern() -> None:
    store = KeyStore()
    store.set("gnews", FAKE_PLAIN)
    assert store.redact(f"bad key {FAKE_PLAIN}") == f"bad key {REDACTED}"
    assert looks_like_secret(FAKE_ANTHROPIC)
    assert not looks_like_secret("claude-sonnet-5-5")


def test_logs_on_disk_never_contain_keys(tmp_path: Path) -> None:
    store = KeyStore()
    store.set("gnews", FAKE_PLAIN)
    logger = setup_logging(tmp_path, store=store)
    logger.info("calling with %s", FAKE_PLAIN)
    try:
        raise RuntimeError(f"provider said: invalid key {FAKE_ANTHROPIC}")
    except RuntimeError:
        logger.exception("request failed")
    for h in logger.handlers:
        h.flush()
    text = (tmp_path / "newsrag.log").read_text("utf-8")
    assert FAKE_PLAIN not in text and FAKE_ANTHROPIC not in text
    assert text.count(REDACTED) >= 2
    assert "RuntimeError" in text
    logging.getLogger("newsrag").handlers.clear()
