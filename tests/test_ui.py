"""Headless UI tests with Streamlit's AppTest. Models and network are replaced with fakes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from newsrag.secrets import KEYS
from newsrag.sources_config import FeedCheck
from newsrag.ui import resources
from newsrag.ui.theme import css
from tests.conftest_ctx import tool_context
from tests.fakes import FakeEmbedder, FakeReranker

APP = str(Path(__file__).parents[1] / "newsrag" / "ui" / "app.py")
TIMEOUT = 60


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    with tool_context(tmp_path, datetime.now(UTC)) as ctx:
        root = ctx.workspace.root
    monkeypatch.setattr(resources, "make_models", lambda e, b, r: (FakeEmbedder(), FakeReranker()))
    monkeypatch.setenv("NEWSRAG_WORKSPACE", str(root))
    monkeypatch.setenv("NEWSRAG_TEST_WS", str(root))
    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield root


def page_script(page: str) -> None:
    """Runs inside AppTest: build the context for the test workspace, render one page."""
    import os
    from pathlib import Path

    import streamlit as st

    from newsrag.secrets import KEYS
    from newsrag.settings import load_settings
    from newsrag.sources_config import effective_config
    from newsrag.store import Store
    from newsrag.tools import ToolContext
    from newsrag.ui import pages
    from newsrag.workspace import open_workspace

    ws = open_workspace(Path(os.environ["NEWSRAG_TEST_WS"]), create=False)
    store = Store(ws.db_path)
    st.session_state["_ctx"] = ToolContext(
        workspace=ws,
        store=store,
        cfg=effective_config(ws.root),
        settings=load_settings(ws.root),
        keys=KEYS,
    )
    try:
        getattr(pages, page)()
    finally:
        store.close()


def run_page(page: str) -> AppTest:
    at = AppTest.from_function(page_script, args=(page,), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    return at


def test_css_sizes_and_themes() -> None:
    assert "font-size: 16px" in css("A", "light")
    assert "font-size: 20px" in css("A++", "dark")
    assert "prefers-color-scheme: dark" in css("A+", "system")
    assert "prefers-color-scheme" not in css("A", "light")


def test_start_screen_without_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEWSRAG_WORKSPACE", raising=False)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Open a workspace"
    at.text_input[0].set_value(str(tmp_path / "new-ws")).run()
    next(b for b in at.button if b.label == "Open or create").click().run()
    assert not at.exception
    assert (tmp_path / "new-ws" / "workspace.json").exists()


def test_full_app_opens_today_with_header_and_sidebar(workspace: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    assert any("Daily News Intelligence" in m.value for m in at.markdown)
    assert any("Rules" in m.value for m in at.sidebar.markdown)  # engine badge (no LLM here)
    assert at.header[0].value == "Today's briefing"


@pytest.mark.parametrize(
    "page",
    [
        "today_page",
        "ask_page",
        "browse_page",
        "fetch_page",
        "sources_page",
        "data_page",
        "settings_page",
    ],
)
def test_every_page_renders(workspace: Path, page: str) -> None:
    run_page(page)


def test_today_writes_and_shows_briefing(workspace: Path) -> None:
    at = run_page("today_page")
    next(b for b in at.button if b.label == "Write briefing").click().run()
    assert not at.exception
    assert list((workspace / "digests").glob("*.json"))
    assert any("Top of the day" in m.value for m in at.markdown)


def test_ask_gives_cited_answer(workspace: Path) -> None:
    at = run_page("ask_page")
    at.chat_input[0].set_value("interest rates inflation").run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Here is what I found" in text and "[1]" in text


def test_browse_search_and_latest(workspace: Path) -> None:
    at = run_page("browse_page")
    assert any("articles" in c.value for c in at.caption)
    at.text_input[0].set_value("FXCHIP").run()
    assert not at.exception
    assert any("FXCHIP" in m.value for m in at.markdown)


def test_keys_are_memory_only(workspace: Path) -> None:
    KEYS.clear()
    at = run_page("settings_page")
    secret = "sk-ant-api03-" + "U" * 30
    at.text_input(key="key_input_anthropic").set_value(secret).run()
    next(b for b in at.button if b.key == "apply_anthropic").click().run()
    assert KEYS.get("anthropic") == secret
    assert at.text_input(key="key_input_anthropic").value == ""  # field cleared after use
    next(b for b in at.button if b.label == "Save settings").click().run()
    assert not at.exception
    saved = (workspace / "settings.json").read_text()
    assert secret not in saved
    for path in workspace.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".log", ".md", ".html"}:
            assert secret not in path.read_text(errors="ignore")
    KEYS.clear()


def test_settings_save_changes(workspace: Path) -> None:
    at = run_page("settings_page")
    top_k = next(n for n in at.number_input if n.label == "Results (top_k)")
    top_k.set_value(5).run()
    next(b for b in at.button if b.label == "Save settings").click().run()
    assert not at.exception
    assert json.loads((workspace / "settings.json").read_text())["retrieval"]["top_k"] == 5


def test_sources_add_feed_after_check(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "newsrag.ui.pages.check_feed_sync",
        lambda url: FeedCheck(
            ok=True, url=url, title="Fixture Feed", headlines=["A", "B", "C"], entries=3
        ),
    )
    at = run_page("sources_page")
    at.text_input[0].set_value("My Fixture Feed")
    at.text_input[1].set_value("https://example.com/feed.xml")
    next(b for b in at.button if b.label == "Check feed").click().run()
    assert not at.exception
    next(b for b in at.button if b.label == "Save 'My Fixture Feed'").click().run()
    assert not at.exception
    saved = json.loads((workspace / "sources.json").read_text())
    assert saved["added"][0]["name"] == "My Fixture Feed"


def test_data_verify(workspace: Path) -> None:
    at = run_page("data_page")
    next(b for b in at.button if b.label == "Verify integrity").click().run()
    assert not at.exception
    assert any("Clean" in s.value for s in at.success)
