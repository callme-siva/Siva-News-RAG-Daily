"""The seven pages. Each is a thin layer over tested functions; no business logic here."""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from newsrag.briefing import (
    REGION_NAMES,
    Digest,
    llm_digest,
    render_markdown,
    rule_digest,
    select_articles,
    write_digest,
)
from newsrag.chat import ChatFilters, ChatSession, ask
from newsrag.config import SourceConfig
from newsrag.engines import EngineUnavailable, select_engine
from newsrag.ingest import reindex
from newsrag.llm.http_clients import OllamaClient
from newsrag.pipeline.fetch import fetch_all
from newsrag.runner import run_ingest
from newsrag.search import SearchFilters, search
from newsrag.secrets import KEYS
from newsrag.settings import Settings, load_settings, save_settings
from newsrag.sources_config import (
    add_source,
    check_feed_sync,
    export_opml,
    is_user_added,
    parse_opml,
    remove_source,
    reset_sources,
    set_enabled,
)
from newsrag.store import backup_workspace, restore_workspace
from newsrag.tools import ToolContext
from newsrag.ui import resources
from newsrag.workspace import WorkspaceError

KEY_FIELDS = {
    "anthropic": "Anthropic (Claude)",
    "gemini": "Google Gemini",
    "openai_compatible": "OpenAI-compatible API",
    "gnews": "GNews (news source)",
    "newsdata": "NewsData.io (news source)",
    "smtp": "Email password (SMTP)",
}


def ctx() -> ToolContext:
    c = st.session_state["_ctx"]
    assert isinstance(c, ToolContext)
    return c


def with_models(c: ToolContext) -> bool:
    """Load the embedding and rerank models into the context. False if the workspace was
    indexed with a different embedding model (the page explains how to fix it)."""
    r = c.settings.retrieval
    base = c.settings.llm.base_url if c.settings.llm.provider == "ollama" else None
    embedder, reranker = resources.make_models(r.embedding_model, base, r.rerank_model)
    try:
        c.workspace.check_embedding_model(embedder.name)
    except WorkspaceError as exc:
        st.error(f"{exc} Use Data → Rebuild index.")
        return False
    c._embedder, c._reranker = embedder, reranker
    return True


def _set_progress(bar: Any, done: int, total: int) -> None:
    bar.progress(done / max(total, 1))


def badge(text: str, ok: bool = False) -> str:
    return f"<span class='nr-badge {'ok' if ok else ''}'>{text}</span>"


def muted(text: str) -> None:
    st.markdown(f"<p class='nr-muted'>{text}</p>", unsafe_allow_html=True)


def regions_and_categories(c: ToolContext) -> tuple[list[str], list[str]]:
    return c.cfg.region_codes(), [x.name for x in c.cfg.categories]


async def _write_digest(c: ToolContext) -> Digest:
    now = datetime.now(UTC)
    s = c.settings
    articles, start = select_articles(
        c.store, c.cfg, now=now, per_group=s.briefing.items_per_category
    )
    base = rule_digest(articles, c.cfg, s.sources.regions, now=now, start=start)
    choice = await select_engine(s.llm, c.cfg, KEYS, task="digest")
    try:
        if choice.client is not None:
            return await llm_digest(choice.client, base, s.briefing, KEYS)
        return base
    finally:
        await choice.aclose()


# ---------------------------------------------------------------- Today


def today_page() -> None:
    c = ctx()
    st.header("Today's briefing")
    files = sorted(c.workspace.digests_dir.glob("*.json"), reverse=True)
    top_left, top_right = st.columns([3, 1], vertical_alignment="bottom")
    if top_right.button("Write briefing", icon=":material/edit_note:", use_container_width=True):
        with st.spinner("Writing the briefing..."):
            try:
                d = asyncio.run(_write_digest(c))
            except EngineUnavailable as exc:
                st.error(f"LLM not available: {exc}")
                return
            write_digest(d, c.workspace.digests_dir)
        st.rerun()
    if not files:
        st.info("No briefing yet. Fetch news (Fetch page), then write today's briefing.")
        return
    chosen = top_left.selectbox("Briefing date", [f.stem for f in files], index=0)
    digest = Digest.model_validate_json(
        (c.workspace.digests_dir / f"{chosen}.json").read_text("utf-8")
    )
    codes = [s.region for s in digest.sections]
    options = ["All", *dict.fromkeys(codes)]
    region = st.segmented_control(
        "Region",
        options,
        default="All",
        format_func=lambda r: REGION_NAMES.get(r, r),
        key="today_region",
    )
    if region and region != "All":
        digest = digest.model_copy(
            update={"sections": [s for s in digest.sections if s.region == region]}
        )
    written = "LLM" if digest.engine == "llm" else "Template"
    st.markdown(
        badge(f"{written} · {len(digest.articles)} articles", ok=digest.engine == "llm"),
        unsafe_allow_html=True,
    )
    st.markdown(render_markdown(digest).split("\n", 2)[2])
    html_file = c.workspace.digests_dir / f"{chosen}.html"
    if html_file.exists():
        st.download_button(
            "Download HTML", html_file.read_bytes(), file_name=html_file.name, mime="text/html"
        )


# ---------------------------------------------------------------- Ask


def ask_page() -> None:
    c = ctx()
    st.header("Ask")
    muted(
        "Answers come only from stored articles, with numbered sources. "
        "If nothing relevant is stored, you'll be told."
    )
    regions, categories = regions_and_categories(c)
    session: ChatSession = st.session_state.setdefault(
        "chat_session", ChatSession(memory_turns=c.settings.retrieval.memory_turns)
    )
    log: list[dict[str, Any]] = st.session_state.setdefault("chat_log", [])
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1], vertical_alignment="bottom")
    sel_regions = f1.multiselect(
        "Regions", regions, default=session.filters.regions or [], placeholder="All regions"
    )
    sel_cats = f2.multiselect(
        "Categories",
        categories,
        default=session.filters.categories or [],
        placeholder="All categories",
    )
    days = f3.select_slider(
        "Time range (days)",
        [1, 3, 7, 14, 30, 90],
        value=session.filters.days if session.filters.days in (1, 3, 7, 14, 30, 90) else 7,
    )
    if f4.button("Clear chat", icon=":material/delete_sweep:"):
        session.clear()
        log.clear()
        st.rerun()
    session.filters = ChatFilters(
        regions=sel_regions or None, categories=sel_cats or None, days=days
    )

    for turn in log:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            _show_answer(turn)

    question = st.chat_input("Ask about the news, e.g. What did the RBI decide this week?")
    if not question:
        return
    with st.chat_message("user"):
        st.write(question)
    if not with_models(c):
        return
    with st.chat_message("assistant"), st.spinner("Searching stored news..."):

        async def go() -> dict[str, Any]:
            try:
                choice = await select_engine(c.settings.llm, c.cfg, KEYS, task="chat")
            except EngineUnavailable as exc:
                return {"question": question, "error": f"LLM not available: {exc}"}
            try:
                a = await ask(c, session, question, choice.client, date.today())
            finally:
                await choice.aclose()
            return {"question": question, **a.model_dump(mode="json")}

        turn = asyncio.run(go())
        log.append(turn)
        _show_answer(turn)


def _show_answer(turn: dict[str, Any]) -> None:
    if "error" in turn:
        st.error(turn["error"])
        return
    st.markdown(turn["text"])
    is_llm = turn["engine"] == "llm"
    label = "LLM · generated from cited articles" if is_llm else "Rules · article text"
    st.markdown(badge(f"{label} · {turn['retrieved']} passages"), unsafe_allow_html=True)
    if turn["sources"]:
        with st.expander(f"Sources ({len(turn['sources'])})", expanded=is_llm):
            for s in turn["sources"]:
                st.markdown(
                    f"[{s['n']}] [{s['title']}]({s['url']}) · {s['source']}, {s['published_date']}"
                )
    for note in turn["notes"]:
        st.caption(f"Note: {note}")


# ---------------------------------------------------------------- Browse


def browse_page() -> None:
    c = ctx()
    st.header("Browse")
    regions, categories = regions_and_categories(c)
    q1, q2 = st.columns([3, 1], vertical_alignment="bottom")
    query = q1.text_input(
        "Search", placeholder="Search headlines and text (leave empty for latest)"
    )
    mode = q2.selectbox(
        "Mode",
        ["hybrid", "semantic", "keyword"],
        index=["hybrid", "semantic", "keyword"].index(c.settings.retrieval.search_mode),
    )
    f1, f2, f3 = st.columns(3)
    sel_regions = f1.multiselect("Regions", regions, placeholder="All regions")
    sel_cats = f2.multiselect("Categories", categories, placeholder="All categories")
    days = f3.select_slider("Published in the last (days)", [1, 3, 7, 14, 30, 90, 365], value=7)
    date_from = date.today() - timedelta(days=days - 1)

    if not query.strip():
        start = int(datetime.combine(date_from, time.min, tzinfo=UTC).timestamp())
        rows = c.store.recent_items(start, int(datetime.now(UTC).timestamp()))
        rows = [
            r
            for r in rows
            if (not sel_regions or r["region"] in sel_regions)
            and (not sel_cats or r["category"] in sel_cats)
        ]
        st.caption(f"{len(rows)} articles")
        for r in rows[:100]:
            _article_card(
                r["title"],
                r["url"],
                r["source"],
                r["region"],
                r["category"],
                datetime.fromtimestamp(r["published_ts"], tz=UTC).date().isoformat(),
                r["summary"],
                r["engine"],
                extra="",
            )
        return

    if not with_models(c):
        return
    rs = c.settings.retrieval.model_copy(update={"search_mode": mode})
    with st.spinner("Searching..."):
        result = search(
            c.store,
            query,
            SearchFilters(
                regions=sel_regions or None, categories=sel_cats or None, date_from=date_from
            ),
            rs,
            c._embedder,
            c._reranker,
        )
    st.caption(f"{len(result.hits)} results · mode {result.mode} · reranked {result.reranked}")
    for note in result.notes:
        st.caption(f"Note: {note}")
    for h in result.hits:
        scores = [f"found by {'+'.join(h.found_by)}"]
        if h.semantic_score is not None:
            scores.append(f"similarity {h.semantic_score:.2f}")
        if h.rerank_score is not None:
            scores.append(f"rerank {h.rerank_score:.2f}")
        _article_card(
            h.title,
            h.url,
            h.source,
            h.region,
            h.category,
            h.published_date,
            h.summary,
            h.engine,
            " · ".join(scores),
        )


def _article_card(
    title: str,
    url: str,
    source: str,
    region: str,
    category: str,
    day: str,
    summary: str,
    engine: str,
    extra: str,
) -> None:
    with st.container(border=True):
        st.markdown(f"**[{title}]({url})**")
        meta = f"{source} · {region} · {category} · {day} · written by {engine}"
        muted(f"{meta}{' · ' + extra if extra else ''}")
        if summary and summary != title:
            st.write(summary[:500])


# ---------------------------------------------------------------- Fetch


def fetch_page() -> None:
    c = ctx()
    st.header("Fetch new news")
    runs = c.store.runs(10)
    if runs:
        last = runs[0]["finished"]
        hours = (datetime.now(UTC) - last).total_seconds() / 3600
        muted(
            f"Last fetched {hours:.0f} hours ago. A fetch looks back to the last run "
            f"(up to {c.settings.sources.max_catchup_days} days)."
        )
    else:
        muted("Nothing fetched yet in this workspace.")
    c1, c2, c3 = st.columns([1, 1, 2])
    limit = c2.number_input("Process at most (0 = all)", min_value=0, value=0, step=10)
    also_digest = c3.checkbox("Write today's briefing afterwards", value=True)
    if c1.button("Fetch now", type="primary", icon=":material/download:"):
        if not with_models(c):
            return
        bar = st.progress(0.0, text="Fetching sources...")

        def progress(done: int, total: int) -> None:
            bar.progress(done / max(total, 1), text=f"Processing {done}/{total}")

        try:
            outcome = asyncio.run(
                run_ingest(
                    c.store,
                    c.cfg,
                    c.settings,
                    KEYS,
                    c._embedder,  # type: ignore[arg-type]
                    limit=int(limit),
                    on_progress=progress,
                )
            )
        except EngineUnavailable as exc:
            st.error(f"LLM not available: {exc}")
            return
        bar.progress(1.0, text="Done")
        r = outcome.ingest
        m = st.columns(5)
        m[0].metric("New", r.new)
        m[1].metric("Skipped (seen)", r.skipped_seen)
        m[2].metric("Merged duplicates", r.merged_duplicate)
        m[3].metric("Not relevant", r.irrelevant)
        m[4].metric("Failed", r.failed)
        st.caption(f"Engine: {outcome.engine} · {outcome.engine_reason}")
        if outcome.failed_sources:
            st.warning("Sources with errors: " + ", ".join(outcome.failed_sources))
        if also_digest:
            with st.spinner("Writing the briefing..."):
                write_digest(asyncio.run(_write_digest(c)), c.workspace.digests_dir)
            st.success("Briefing written. See the Today page.")
        runs = c.store.runs(10)
    if runs:
        st.subheader("Run history")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "finished": r["finished"].strftime("%Y-%m-%d %H:%M UTC"),
                        "new": r.get("ingest", r).get("new"),
                        "skipped": r.get("ingest", r).get("skipped_seen"),
                        "merged": r.get("ingest", r).get("merged_duplicate"),
                        "failed": r.get("ingest", r).get("failed"),
                        "engine": r.get("engine", ""),
                        "source errors": ", ".join(r.get("failed_sources", [])),
                    }
                    for r in runs
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


# ---------------------------------------------------------------- Sources


def sources_page() -> None:
    c = ctx()
    root = c.workspace.root
    st.header("Sources")
    muted(
        "Shipped sources are never edited; your changes are stored in this workspace. "
        "New feeds are checked (robots.txt, fetch, parse) before they are saved."
    )
    last = c.store.last_run_summary() or {}
    failed = set(last.get("failed_sources", []))
    rows = [
        {
            "On": s.enabled,
            "Name": s.name,
            "Type": s.type,
            "Region": s.region,
            "Category": s.category,
            "Location": s.url or s.query or s.api or "",
            "Added by": "you" if is_user_added(root, s.name) else "default",
            "Last run": ("error" if s.name in failed else "ok") if last and s.enabled else "-",
        }
        for s in c.cfg.sources
    ]
    edited = st.data_editor(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        disabled=["Name", "Type", "Region", "Category", "Location", "Added by", "Last run"],
        column_config={
            "On": st.column_config.CheckboxColumn(help="Include this source when fetching")
        },
        key="sources_editor",
    )
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Save on/off changes", icon=":material/save:"):
        for before, after in zip(rows, edited.to_dict("records"), strict=True):
            if bool(after["On"]) != before["On"]:
                set_enabled(root, str(before["Name"]), bool(after["On"]))
        st.success("Saved.")
        st.rerun()
    if b2.button("Check all enabled sources", icon=":material/network_check:"):
        enabled = [s for s in c.cfg.sources if s.enabled]
        with st.spinner(f"Checking {len(enabled)} sources..."):
            results = asyncio.run(fetch_all(enabled, KEYS, max_age_hours=24 * 30))
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Source": r.source,
                        "Status": r.status,
                        "Items": r.count,
                        "Detail": r.error or "",
                    }
                    for r in results
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    b3.download_button(
        "Export OPML",
        export_opml(c.cfg),
        file_name="newsrag-sources.opml",
        mime="text/x-opml",
        icon=":material/upload_file:",
    )
    if b4.button(
        "Restore defaults", icon=":material/restart_alt:", help="Remove all your source changes"
    ):
        st.session_state.confirm_reset_sources = True
    if st.session_state.get("confirm_reset_sources"):
        st.warning("Remove all added sources and on/off changes in this workspace?")
        y, n = st.columns(2)
        if y.button("Yes, restore defaults"):
            reset_sources(root)
            st.session_state.confirm_reset_sources = False
            st.rerun()
        if n.button("Cancel"):
            st.session_state.confirm_reset_sources = False
            st.rerun()

    regions, categories = regions_and_categories(c)
    st.subheader("Add an RSS feed")
    with st.form("add_feed"):
        a1, a2 = st.columns(2)
        name = a1.text_input("Name", placeholder="Example Daily - Markets")
        url = a2.text_input("Feed URL", placeholder="https://example.com/feed.xml")
        a3, a4 = st.columns(2)
        region = a3.selectbox("Region", regions)
        category = a4.selectbox("Category", categories)
        checked = st.form_submit_button("Check feed", icon=":material/fact_check:")
    if checked:
        if not name.strip() or not url.strip():
            st.error("Enter a name and a feed URL.")
        else:
            with st.spinner("Checking robots.txt, fetching and parsing..."):
                st.session_state.feed_check = (
                    check_feed_sync(url.strip()),
                    name.strip(),
                    region,
                    category,
                )
    pending = st.session_state.get("feed_check")
    if pending:
        result, p_name, p_region, p_category = pending
        if not result.ok:
            st.error(f"Not saved: {result.error}")
        else:
            st.success(f"**{result.title}**: {result.entries} entries")
            for h in result.headlines:
                st.markdown(f"- {h}")
            if st.button(f"Save '{p_name}'", type="primary"):
                try:
                    add_source(
                        root,
                        SourceConfig(
                            name=p_name,
                            type="rss",
                            url=result.url,
                            region=p_region,
                            category=p_category,
                        ),
                    )
                except ValidationError as exc:
                    st.error(str(exc.errors()[0]["msg"]))
                else:
                    st.session_state.pop("feed_check")
                    st.rerun()

    mine = [s.name for s in c.cfg.sources if is_user_added(root, s.name)]
    if mine:
        st.subheader("Remove a feed you added")
        r1, r2 = st.columns([3, 1], vertical_alignment="bottom")
        target = r1.selectbox("Feed", mine)
        if r2.button("Remove", icon=":material/delete:"):
            remove_source(root, target)
            st.rerun()

    st.subheader("Import OPML")
    up = st.file_uploader("OPML file", type=["opml", "xml"])
    if up is not None:
        i1, i2 = st.columns(2)
        d_region = i1.selectbox("Region for feeds without one", regions, key="opml_region")
        d_cat = i2.selectbox("Category for feeds without one", categories, key="opml_cat")
        try:
            found = parse_opml(up.getvalue().decode("utf-8", "replace"), d_region, d_cat)
        except ValueError as exc:
            st.error(str(exc))
            return
        existing = {s.name for s in c.cfg.sources} | {s.url for s in c.cfg.sources if s.url}
        new = [s for s in found if s.name not in existing and s.url not in existing]
        st.caption(
            f"{len(found)} feeds in the file, {len(new)} new. Imported feeds are added "
            "without a live check; use 'Check all enabled sources' afterwards."
        )
        if new and st.button(f"Add {len(new)} feeds"):
            errors = []
            for s in new:
                try:
                    add_source(root, s)
                except ValidationError as exc:
                    errors.append(f"{s.name}: {exc.errors()[0]['msg']}")
            if errors:
                st.warning("Skipped: " + "; ".join(errors))
            st.rerun()


# ---------------------------------------------------------------- Data


def data_page() -> None:
    c = ctx()
    st.header("Data")
    s = c.store.stats()
    m = st.columns(3)
    m[0].metric("Articles", s["items"])
    m[1].metric("Chunks", s["chunks"])
    m[2].metric("Size", f"{s['db_bytes'] / 1_000_000:.1f} MB")
    row2 = st.columns(3)
    row2[0].metric("Oldest article", s["oldest"] or "-")
    row2[1].metric("Newest article", s["newest"] or "-")
    row2[2].metric("Runs", s["runs"])
    if s["by_region_category"]:
        st.bar_chart(pd.Series(s["by_region_category"], name="articles"), horizontal=True)

    st.subheader("Clean up old articles")
    retention = c.settings.data.retention_days
    days = st.number_input(
        "Remove articles published more than N days ago", min_value=1, value=retention or 90
    )
    preview = c.store.cleanup(int(days), now=datetime.now(UTC), apply=False)
    st.write(
        f"This would remove **{preview.items}** articles and **{preview.chunks}** chunks "
        f"published before {preview.cutoff.date()}. "
        "Seen links are kept so they are not fetched again."
    )
    backup_first = st.checkbox("Back up the workspace first", value=True)
    if st.button("Clean up now", disabled=preview.items == 0, icon=":material/cleaning_services:"):
        if backup_first:
            backup_workspace(c.workspace.root, c.workspace.backups_dir, c.store)
        done = c.store.cleanup(int(days), now=datetime.now(UTC), apply=True)
        st.success(f"Removed {done.items} articles.")

    st.subheader("Delete selected articles")
    regions, categories = regions_and_categories(c)
    d1, d2, d3 = st.columns(3)
    del_sources = d1.multiselect("Sources", c.store.sources_in_store())
    del_regions = d2.multiselect("Regions", regions, key="del_regions")
    del_cats = d3.multiselect("Categories", categories, key="del_cats")
    d4, d5 = st.columns(2)
    after = d4.date_input("Published on or after", value=None)
    before = d5.date_input("Published on or before", value=None)
    from_ts = int(datetime.combine(after, time.min, tzinfo=UTC).timestamp()) if after else None
    to_ts = int(datetime.combine(before, time.max, tzinfo=UTC).timestamp()) if before else None

    def delete(apply: bool) -> int:
        return c.store.delete_where(
            sources=del_sources or None,
            regions=del_regions or None,
            categories=del_cats or None,
            from_ts=from_ts,
            to_ts=to_ts,
            apply=apply,
        )

    if del_sources or del_regions or del_cats or from_ts is not None or to_ts is not None:
        count = delete(apply=False)
        confirm = st.checkbox(f"I understand this deletes {count} articles")
        if st.button(
            "Delete", disabled=not confirm or count == 0, icon=":material/delete_forever:"
        ):
            backup_workspace(c.workspace.root, c.workspace.backups_dir, c.store)
            delete(apply=True)
            st.success(f"Deleted {count} articles (a backup was made first).")
    else:
        muted("Choose at least one condition.")

    st.subheader("Maintenance")
    x1, x2, x3 = st.columns(3)
    if x1.button("Verify integrity", icon=":material/verified:"):
        rep = c.store.verify(repair=False)
        if rep.clean:
            st.success("Clean: keyword index, chunks and vectors agree.")
        else:
            st.warning(
                f"Orphan keyword rows {rep.orphan_fts}, orphan chunks {rep.orphan_chunks}, "
                f"missing vectors {rep.missing_vectors}, missing keyword rows {rep.missing_fts}."
            )
            st.session_state.needs_repair = True
    if st.session_state.get("needs_repair") and st.button("Repair now"):
        rep = c.store.verify(repair=True)
        st.session_state.needs_repair = False
        st.success("Repaired. " + " ".join(rep.details))
    if x2.button(
        "Rebuild index",
        icon=":material/build:",
        help="Needed after changing chunk size or the embedding model",
    ):
        r = c.settings.retrieval
        base = c.settings.llm.base_url if c.settings.llm.provider == "ollama" else None
        embedder, _ = resources.make_models(r.embedding_model, base, r.rerank_model)
        bar = st.progress(0.0)
        n = reindex(
            c.store, embedder, c.settings, on_progress=lambda d, t: _set_progress(bar, d, t)
        )
        c.workspace.set_embedding_model(embedder.name)
        st.success(f"Rebuilt the index for {n} articles.")
    if x3.button("Compact database", icon=":material/compress:"):
        before_size = c.store.path.stat().st_size
        c.store.compact()
        st.success(f"{before_size:,} → {c.store.path.stat().st_size:,} bytes")

    st.subheader("Backup and restore")
    y1, y2 = st.columns(2)
    if y1.button("Create backup", icon=":material/backup:"):
        st.session_state.last_backup = str(
            backup_workspace(c.workspace.root, c.workspace.backups_dir, c.store)
        )
    if st.session_state.get("last_backup"):
        p = Path(st.session_state.last_backup)
        if p.exists():
            y1.download_button(
                "Download backup", p.read_bytes(), file_name=p.name, mime="application/zip"
            )
    up = y2.file_uploader("Restore from a backup zip", type=["zip"])
    if up is not None:
        sure = y2.checkbox("Replace this workspace's contents with the backup")
        if y2.button("Restore", disabled=not sure):
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp.write(up.getvalue())
            c.store.close()
            try:
                restore_workspace(Path(tmp.name), c.workspace.root)
            except Exception as exc:
                st.error(f"Restore failed: {exc}")
            else:
                st.success("Restored.")
            finally:
                Path(tmp.name).unlink(missing_ok=True)
            st.rerun()


# ---------------------------------------------------------------- Settings


async def _ollama_models(base_url: str) -> list[str]:
    client = OllamaClient(model="", base_url=base_url, timeout_s=3)
    try:
        return await client.list_models()
    finally:
        await client.aclose()


def _keys_section() -> None:
    st.subheader("API keys")
    muted(
        "Kept in memory for this session only. Never written to disk, logs or settings. "
        "Environment variables are loaded at start."
    )
    for name, label in KEY_FIELDS.items():
        k1, k2, k3 = st.columns([3, 1, 1], vertical_alignment="bottom")
        widget = f"key_input_{name}"
        k1.text_input(
            label, type="password", key=widget, placeholder="set" if KEYS.has(name) else "not set"
        )

        def apply(n: str = name, w: str = widget) -> None:
            value = st.session_state.get(w, "")
            if value:
                KEYS.set(n, value)
            st.session_state[w] = ""
            st.session_state.pop("engine_label", None)

        def forget(n: str = name) -> None:
            KEYS.set(n, None)
            st.session_state.pop("engine_label", None)

        k2.button("Use key", key=f"apply_{name}", on_click=apply)
        k3.button("Forget", key=f"forget_{name}", on_click=forget, disabled=not KEYS.has(name))


def settings_page() -> None:
    c = ctx()
    root = c.workspace.root
    s = load_settings(root)
    st.header("Settings")
    _keys_section()

    regions, _ = regions_and_categories(c)
    st.subheader("LLM")
    l1, l2 = st.columns(2)
    modes = ["auto", "rules", "llm"]
    s.llm.mode = cast(Any, l1.selectbox)(
        "Engine mode",
        modes,
        index=modes.index(s.llm.mode),
        help="auto: use an LLM if one works, otherwise rules",
    )
    providers = ["ollama", "local_openai", "anthropic", "gemini", "openai_compatible"]
    names = {
        "ollama": "Local: Ollama",
        "local_openai": "Local: OpenAI-compatible server",
        "anthropic": "Anthropic (Claude)",
        "gemini": "Google Gemini",
        "openai_compatible": "OpenAI-compatible API",
    }
    s.llm.provider = cast(Any, l2.selectbox)(
        "Provider", providers, index=providers.index(s.llm.provider), format_func=names.__getitem__
    )
    if s.llm.provider in ("ollama", "local_openai", "openai_compatible"):
        default_url = (
            "http://localhost:11434" if s.llm.provider == "ollama" else "http://localhost:1234/v1"
        )
        s.llm.base_url = st.text_input("Server base URL", value=s.llm.base_url or default_url)
    installed: list[str] = []
    if s.llm.provider == "ollama":
        try:
            installed = asyncio.run(_ollama_models(s.llm.base_url or "http://localhost:11434"))
        except Exception:
            muted("Ollama is not reachable at this URL, so its model list can't be shown.")
    m1, m2, m3 = st.columns(3)
    for col, field, label in (
        (m1, "model_processing", "Model for tagging articles"),
        (m2, "model_chat", "Model for chat"),
        (m3, "model_digest", "Model for the briefing"),
    ):
        current = getattr(s.llm, field) or ""
        if installed:
            opts = ["(first installed)", *installed]
            pick = col.selectbox(label, opts, index=opts.index(current) if current in opts else 0)
            setattr(s.llm, field, None if pick == "(first installed)" else pick)
        else:
            hint = "claude-opus-5-5 (default)" if s.llm.provider == "anthropic" else "model name"
            setattr(
                s.llm, field, col.text_input(label, value=current, placeholder=hint).strip() or None
            )
    n1, n2, n3, n4 = st.columns(4)
    s.llm.temperature = n1.slider(
        "Temperature",
        0.0,
        1.0,
        s.llm.temperature,
        0.05,
        help="Not sent to Claude models, which reject it",
    )
    s.llm.max_tokens = int(n2.number_input("Max output tokens", 64, 16384, s.llm.max_tokens, 64))
    s.llm.timeout_s = int(n3.number_input("Timeout (s)", 5, 600, s.llm.timeout_s, 5))
    s.llm.concurrency = int(n4.number_input("Parallel requests", 1, 16, s.llm.concurrency, 1))

    st.subheader("Retrieval")
    r = s.retrieval
    modes_r = ["hybrid", "semantic", "keyword"]
    r1, r2, r3, r4 = st.columns(4)
    r.search_mode = cast(Any, r1.selectbox)(
        "Search mode", modes_r, index=modes_r.index(r.search_mode)
    )
    r.top_k = int(r2.number_input("Results (top_k)", 1, 50, r.top_k))
    r.candidates_k = int(r3.number_input("Candidates (candidates_k)", 1, 200, r.candidates_k))
    r.max_chunks_per_article = int(
        r4.number_input("Max chunks per article", 1, 10, r.max_chunks_per_article)
    )
    r5, r6, r7, r8 = st.columns(4)
    r.keyword_weight = r5.slider(
        "Keyword weight",
        0.0,
        1.0,
        r.keyword_weight,
        0.05,
        help="0 = semantic only, 1 = keyword only",
    )
    r.min_similarity = r6.slider(
        "Relevance floor (similarity)",
        0.0,
        1.0,
        r.min_similarity,
        0.05,
        help="Semantic-only matches below this are dropped",
    )
    r.rerank = r7.toggle("Rerank results", value=r.rerank)
    r.min_score = r8.slider("Min rerank score", 0.0, 1.0, r.min_score, 0.05)
    r9, r10, r11 = st.columns(3)
    old_chunking = (r.chunk_size, r.chunk_overlap, r.embedding_model)
    r.chunk_size = int(r9.number_input("Chunk size (characters)", 200, 8000, r.chunk_size, 100))
    r.chunk_overlap = int(
        r10.number_input("Chunk overlap (characters)", 0, 2000, r.chunk_overlap, 50)
    )
    r.memory_turns = int(r11.number_input("Chat memory (turns)", 0, 50, r.memory_turns))
    r.embedding_model = st.text_input(
        "Embedding model",
        value=r.embedding_model,
        help="A sentence-transformers name, or ollama:<model>",
    )
    r.rerank_model = st.text_input("Rerank model", value=r.rerank_model)
    if (r.chunk_size, r.chunk_overlap, r.embedding_model) != old_chunking:
        st.info("Chunk or embedding changes apply to stored articles after Data → Rebuild index.")

    st.subheader("Sources and duplicates")
    src = s.sources
    p1, p2, p3 = st.columns(3)
    src.regions = p1.multiselect("Regions to fetch", regions, default=src.regions) or regions
    src.max_age_hours = int(p2.number_input("Max article age (hours)", 1, 720, src.max_age_hours))
    src.max_per_group = int(
        p3.number_input("Max per region and category", 1, 500, src.max_per_group)
    )
    p4, p5, p6 = st.columns(3)
    src.dedupe_threshold = p4.slider(
        "Duplicate headline threshold", 0.0, 1.0, src.dedupe_threshold, 0.05
    )
    src.dedupe_window_days = int(
        p5.number_input("Cross-day duplicate window (days)", 0, 60, src.dedupe_window_days)
    )
    src.on_update = cast(Any, p6.selectbox)(
        "When a stored article changes",
        ["ignore", "replace"],
        index=["ignore", "replace"].index(src.on_update),
    )

    st.subheader("Briefing and data")
    q1, q2, q3, q4 = st.columns(4)
    s.briefing.items_per_category = int(
        q1.number_input("Items per category", 1, 20, s.briefing.items_per_category)
    )
    s.briefing.style = cast(Any, q2.selectbox)(
        "Style", ["brief", "detailed"], index=["brief", "detailed"].index(s.briefing.style)
    )
    s.briefing.email_enabled = q3.toggle(
        "Email the briefing",
        value=s.briefing.email_enabled,
        help="Needs NEWSRAG_SMTP_* variables; see README",
    )
    keep_forever = q4.toggle("Keep articles forever", value=s.data.retention_days is None)
    if not keep_forever:
        s.data.retention_days = int(
            st.number_input("Retention (days)", 1, 3650, s.data.retention_days or 90)
        )
    else:
        s.data.retention_days = None
    s.data.auto_cleanup = st.toggle(
        "Clean up automatically after each fetch", value=s.data.auto_cleanup
    )
    s.appearance.high_contrast = st.toggle("High contrast", value=s.appearance.high_contrast)
    s.safety.show_provenance = st.toggle(
        "Show who wrote each item (rules or LLM)", value=s.safety.show_provenance
    )

    st.divider()
    a1, a2, a3 = st.columns(3)
    if a1.button("Save settings", type="primary", icon=":material/save:"):
        try:
            validated = Settings.model_validate(s.model_dump())
            save_settings(root, validated)
        except (ValidationError, ValueError) as exc:
            st.error(f"Not saved: {exc}")
        else:
            st.session_state.pop("engine_label", None)
            st.success("Saved.")
    if a2.button("Test LLM connection", icon=":material/lan:"):
        test = s.model_copy(deep=True)
        test.llm.mode = "llm"

        async def check_connection() -> str:
            choice = await select_engine(test.llm, c.cfg, KEYS, task="chat")
            await choice.aclose()
            return choice.engine.label

        try:
            st.success(f"Connected: {asyncio.run(check_connection())}")
        except EngineUnavailable as exc:
            st.error(f"Not available: {exc}")
    if a3.button("Reset to defaults", icon=":material/restart_alt:"):
        save_settings(root, Settings())
        st.session_state.pop("engine_label", None)
        st.rerun()
    with st.expander("Settings file (no keys)"):
        st.code(json.dumps(s.model_dump(mode="json"), indent=2), language="json")
