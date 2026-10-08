"""Streamlit entry point: `newsrag ui` or `streamlit run newsrag/ui/app.py`.

Flow: open a workspace -> header (font size, theme) -> sidebar (workspace, engine) -> page.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, cast

import streamlit as st

from newsrag.engines import select_engine
from newsrag.secrets import KEYS
from newsrag.settings import load_settings, save_settings
from newsrag.sources_config import effective_config
from newsrag.store import Store
from newsrag.tools import ToolContext
from newsrag.ui import pages
from newsrag.ui.theme import css
from newsrag.workspace import (
    WorkspaceError,
    open_workspace,
    recent_workspaces,
    remember_workspace,
)

st.set_page_config(
    page_title="Daily News Intelligence", page_icon=":material/newspaper:", layout="wide"
)

if "keys_loaded" not in st.session_state:
    KEYS.load_from_env()
    st.session_state.keys_loaded = True


def workspace_screen() -> None:
    st.markdown(css("A", "system"), unsafe_allow_html=True)
    st.title("Open a workspace")
    st.markdown(
        '<p class="nr-muted">All data for a workspace (articles, search index, digests, '
        "logs, settings) is stored in one folder. API keys are never stored.</p>",
        unsafe_allow_html=True,
    )
    recent = [p for p in recent_workspaces() if (p / "workspace.json").exists()]
    if recent:
        st.subheader("Recent")
        for p in recent:
            c1, c2 = st.columns([5, 1])
            c1.markdown(
                f"**{p.name}**  \n<span class='nr-muted'>{p}</span>", unsafe_allow_html=True
            )
            if c2.button("Open", key=f"open-{p}", help=f"Open {p}"):
                st.session_state.ws_path = str(p)
                st.rerun()
    st.subheader("Choose or create a folder")
    path = st.text_input(
        "Folder path",
        placeholder=str(Path.home() / "NewsRAG" / "personal"),
        help="An existing workspace folder, or an empty or new folder to create one in.",
    )
    if st.button("Open or create", type="primary", disabled=not path.strip()):
        try:
            ws = open_workspace(Path(path.strip()))
        except WorkspaceError as exc:
            st.error(str(exc))
        else:
            remember_workspace(ws.root)
            st.session_state.ws_path = str(ws.root)
            st.rerun()


def header(ws_root: Path) -> None:
    settings = load_settings(ws_root)
    a = settings.appearance
    left, right = st.columns([2, 3], vertical_alignment="center")
    left.markdown(f"#### Daily News Intelligence · {ws_root.name}")
    theme_labels = {
        "light": ":material/light_mode: Light",
        "dark": ":material/dark_mode: Dark",
        "system": ":material/desktop_windows: System",
    }
    with right.container(horizontal=True, horizontal_alignment="right", gap="small"):
        font = st.segmented_control(
            "Text size",
            ["A", "A+", "A++"],
            default=a.font_size,
            key="font_size",
            required=True,
            label_visibility="collapsed",
            width="content",
            help="Text size: regular, large, extra large",
        )
        theme = st.segmented_control(
            "Theme",
            list(theme_labels),
            default=a.theme,
            key="theme",
            required=True,
            format_func=theme_labels.__getitem__,
            label_visibility="collapsed",
            width="content",
            help="Theme: light, dark, or follow the system",
        )
    changed = False
    if font and font != a.font_size:
        a.font_size, changed = cast(Any, font), True
    if theme and theme != a.theme:
        a.theme, changed = cast(Any, theme), True
    if changed:
        save_settings(ws_root, settings)
    st.markdown(css(a.font_size, a.theme, a.high_contrast), unsafe_allow_html=True)


def engine_status(ctx: ToolContext) -> str:
    """Probe once per session (and after Settings changes); show as a sidebar badge."""
    if "engine_label" not in st.session_state:
        choice = asyncio.run(select_engine(ctx.settings.llm, ctx.cfg, KEYS, task="chat"))
        asyncio.run(choice.aclose())
        st.session_state.engine_label = choice.engine.label
        st.session_state.engine_reason = choice.reason
    return str(st.session_state.engine_label)


def sidebar(ctx: ToolContext) -> None:
    with st.sidebar:
        st.markdown(f"**Workspace**  \n:material/folder: {ctx.workspace.root.name}")
        if st.button("Switch workspace", icon=":material/swap_horiz:"):
            for k in ("ws_path", "engine_label", "engine_reason", "chat_session", "chat_log"):
                st.session_state.pop(k, None)
            st.rerun()
        label = engine_status(ctx)
        kind = "ok" if label.startswith("LLM") else ""
        st.markdown(
            f"**Engine**  \n<span class='nr-badge {kind}'>{label}</span>", unsafe_allow_html=True
        )
        st.caption(st.session_state.get("engine_reason", ""))
        if st.button("Recheck engine", icon=":material/refresh:"):
            st.session_state.pop("engine_label", None)
            st.rerun()


def main() -> None:
    preset = os.environ.get("NEWSRAG_WORKSPACE")
    if "ws_path" not in st.session_state and preset and (Path(preset) / "workspace.json").exists():
        st.session_state.ws_path = preset
    if "ws_path" not in st.session_state:
        workspace_screen()
        return
    try:
        ws = open_workspace(Path(st.session_state.ws_path), create=False)
    except WorkspaceError as exc:
        st.error(str(exc))
        st.session_state.pop("ws_path", None)
        return
    header(ws.root)
    store = Store(ws.db_path)
    try:
        ctx = ToolContext(
            workspace=ws,
            store=store,
            cfg=effective_config(ws.root),
            settings=load_settings(ws.root),
            keys=KEYS,
        )
        st.session_state["_ctx"] = ctx
        sidebar(ctx)
        nav = st.navigation(
            [
                st.Page(
                    pages.today_page,
                    title="Today",
                    icon=":material/newspaper:",
                    url_path="today",
                    default=True,
                ),
                st.Page(pages.ask_page, title="Ask", icon=":material/chat:", url_path="ask"),
                st.Page(
                    pages.topic_page,
                    title="Topic brief",
                    icon=":material/summarize:",
                    url_path="topic",
                ),
                st.Page(
                    pages.browse_page, title="Browse", icon=":material/search:", url_path="browse"
                ),
                st.Page(
                    pages.fetch_page, title="Fetch", icon=":material/refresh:", url_path="fetch"
                ),
                st.Page(
                    pages.sources_page,
                    title="Sources",
                    icon=":material/rss_feed:",
                    url_path="sources",
                ),
                st.Page(pages.data_page, title="Data", icon=":material/database:", url_path="data"),
                st.Page(
                    pages.settings_page,
                    title="Settings",
                    icon=":material/settings:",
                    url_path="settings",
                ),
            ]
        )
        nav.run()
    finally:
        store.close()


main()
