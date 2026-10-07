"""CLI commands that use the workspace store: run, search, data, digest, chat, tools
(REQUIREMENTS FR16-FR25, FR38-FR43, section 12)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from newsrag.briefing import (
    Digest,
    EmailNotConfigured,
    llm_digest,
    render_markdown,
    rule_digest,
    select_articles,
    send_email,
    write_digest,
)
from newsrag.chat import ChatFilters, ChatSession, ask, format_answer
from newsrag.embeddings import CrossEncoderReranker, make_embedder
from newsrag.engines import EngineUnavailable, select_engine
from newsrag.ingest import reindex
from newsrag.logging_setup import setup_logging
from newsrag.runner import run_ingest
from newsrag.search import SearchFilters, search
from newsrag.secrets import KEYS
from newsrag.settings import Settings, load_settings
from newsrag.sources_config import effective_config
from newsrag.store import Store, backup_workspace, restore_workspace
from newsrag.tools import ToolContext, tool_schemas
from newsrag.workspace import Workspace, WorkspaceError, open_workspace, recent_workspaces


def require_workspace(arg: str | None) -> Workspace:
    if arg:
        return open_workspace(Path(arg), create=False)
    recent = recent_workspaces()
    if not recent:
        raise WorkspaceError(
            "No workspace yet. Create one with: newsrag --workspace PATH workspace"
        )
    return open_workspace(recent[0], create=False)


def _embedder_for(settings: Settings, ws: Workspace):  # type: ignore[no-untyped-def]
    base = settings.llm.base_url if settings.llm.provider == "ollama" else None
    embedder = make_embedder(settings.retrieval.embedding_model, base)
    ws.check_embedding_model(embedder.name)
    return embedder


def _context(ws: Workspace, settings: Settings) -> tuple[ToolContext, Store]:
    store = Store(ws.db_path)
    ctx = ToolContext(
        workspace=ws,
        store=store,
        cfg=effective_config(ws.root),
        settings=settings,
        keys=KEYS,
        _embedder=_embedder_for(settings, ws),
    )
    return ctx, store


async def _make_digest(ctx: ToolContext, now: datetime) -> Digest:
    """Select articles, write with the LLM if available (template otherwise), save files."""
    s = ctx.settings
    articles, start = select_articles(
        ctx.store, ctx.cfg, now=now, per_group=s.briefing.items_per_category
    )
    base = rule_digest(articles, ctx.cfg, s.sources.regions, now=now, start=start)
    choice = await select_engine(s.llm, ctx.cfg, ctx.keys, task="digest")
    try:
        if choice.client is not None:
            return await llm_digest(choice.client, base, s.briefing, ctx.keys)
        return base
    finally:
        await choice.aclose()


def _publish_digest(ctx: ToolContext, digest: Digest) -> None:
    _, page = write_digest(digest, ctx.workspace.digests_dir)
    print(f"Digest ({digest.writer}, {len(digest.articles)} articles): {page}")
    if ctx.settings.briefing.email_enabled:
        try:
            print(f"Emailed to {send_email(digest, ctx.keys)}")
        except EmailNotConfigured as exc:
            print(f"Email not sent: {exc}")
        except OSError as exc:
            print(f"Email failed: {KEYS.redact(str(exc))}")


def cmd_run(args: argparse.Namespace) -> int:
    ws = require_workspace(args.workspace)
    settings = load_settings(ws.root)
    KEYS.load_from_env()
    log = setup_logging(ws.logs_dir)
    ctx, store = _context(ws, settings)

    def progress(done: int, total: int) -> None:
        print(f"\r  processed {done}/{total}", end="", flush=True)

    async def run() -> int:
        try:
            outcome = await run_ingest(
                store,
                ctx.cfg,
                settings,
                KEYS,
                ctx.embedder,
                limit=args.limit,
                on_progress=progress,
            )
        except EngineUnavailable as exc:
            print(f"LLM not available: {exc}")
            return 1
        print()
        r = outcome.ingest
        if outcome.lookback_hours > settings.sources.max_age_hours:
            print(f"Caught up: looked back {outcome.lookback_hours} hours since the last run")
        print(f"Engine: {outcome.engine} | {outcome.engine_reason}")
        print(
            f"New {r.new} | skipped as seen {r.skipped_seen} | merged duplicates "
            f"{r.merged_duplicate} | updated {r.updated} | not relevant {r.irrelevant} | "
            f"failed {r.failed}"
        )
        if r.by_engine:
            print("Written by: " + ", ".join(f"{k} {v}" for k, v in r.by_engine.items()))
        if outcome.failed_sources:
            print("Sources with errors: " + ", ".join(outcome.failed_sources))
        if outcome.auto_cleanup_removed:
            print(f"Auto cleanup removed {outcome.auto_cleanup_removed} old items")
        log.info("Run finished: new=%d failed=%d", r.new, r.failed)
        if not args.no_digest:
            _publish_digest(ctx, await _make_digest(ctx, datetime.now(UTC)))
        return 0

    try:
        return asyncio.run(run())
    finally:
        store.close()


def cmd_digest(args: argparse.Namespace) -> int:
    ws = require_workspace(args.workspace)
    settings = load_settings(ws.root)
    if args.mode:
        settings.llm.mode = args.mode
    KEYS.load_from_env()
    setup_logging(ws.logs_dir)
    ctx, store = _context(ws, settings)
    try:
        digest = asyncio.run(_make_digest(ctx, datetime.now(UTC)))
        _publish_digest(ctx, digest)
        if args.print:
            print()
            print(render_markdown(digest))
    finally:
        store.close()
    return 0


CHAT_HELP = """Commands: /region IN [US EU] | /category Finance [...] | /days N
          /all (clear filters) | /clear (forget conversation) | /quit"""


def _chat_command(line: str, session: ChatSession) -> str | None:
    parts = line.split()
    cmd, rest = parts[0].lower(), parts[1:]
    f = session.filters
    if cmd in ("/quit", "/exit"):
        return "quit"
    if cmd == "/region":
        f.regions = [r.upper() for r in rest] or None
    elif cmd == "/category":
        f.categories = [c.capitalize() for c in rest] or None
    elif cmd == "/days" and rest and rest[0].isdigit():
        f.days = max(1, int(rest[0]))
    elif cmd == "/all":
        session.filters = ChatFilters()
    elif cmd == "/clear":
        session.clear()
    else:
        return CHAT_HELP
    return f"Filters: {session.filters.describe()}"


def cmd_chat(args: argparse.Namespace) -> int:
    ws = require_workspace(args.workspace)
    settings = load_settings(ws.root)
    if args.mode:
        settings.llm.mode = args.mode
    KEYS.load_from_env()
    setup_logging(ws.logs_dir)
    ctx, store = _context(ws, settings)
    session = ChatSession(
        memory_turns=settings.retrieval.memory_turns,
        filters=ChatFilters(regions=args.region, categories=args.category, days=args.days),
    )

    async def run() -> int:
        try:
            choice = await select_engine(settings.llm, ctx.cfg, KEYS, task="chat")
        except EngineUnavailable as exc:
            print(f"LLM not available: {exc}")
            return 1
        try:
            print(f"Answering with: {choice.engine.label} | {choice.reason}")
            questions = [args.question] if args.question else None
            if questions is None:
                print(CHAT_HELP)
            while True:
                if questions is not None:
                    if not questions:
                        break
                    q = questions.pop(0)
                else:
                    try:
                        q = input("\nYou> ").strip()
                    except EOFError:
                        break
                if not q:
                    continue
                if q.startswith("/"):
                    msg = _chat_command(q, session)
                    if msg == "quit":
                        break
                    print(msg)
                    continue
                answer = await ask(ctx, session, q, choice.client, date.today())
                print("\n" + format_answer(answer))
        finally:
            await choice.aclose()
        return 0

    try:
        return asyncio.run(run())
    finally:
        store.close()


def cmd_ui(args: argparse.Namespace) -> int:
    """Start the Streamlit UI. The API keys in this shell's environment are loaded by the app."""
    app = Path(__file__).parent / "ui" / "app.py"
    env = dict(os.environ)
    if args.workspace:
        env["NEWSRAG_WORKSPACE"] = str(Path(args.workspace).expanduser().resolve())
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--client.toolbarMode",
        "minimal",
    ]
    print(f"Starting the UI at http://localhost:{args.port}  (Ctrl+C to stop)")
    return subprocess.call(cmd, env=env)


def cmd_tools(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps(tool_schemas(), indent=2))
        return 0
    for t in tool_schemas():
        flag = "changes data" if t["changes_data"] else "read-only"
        params = ", ".join(t["input_schema"].get("properties", {}))
        print(f"{t['name']:<16} [{flag}] {t['description']}\n{'':<16} params: {params}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    ws = require_workspace(args.workspace)
    settings = load_settings(ws.root)
    r = settings.retrieval
    if args.mode:
        r.search_mode = args.mode
    if args.no_rerank:
        r.rerank = False
    if args.top_k:
        r.top_k = min(args.top_k, r.candidates_k)
    filters = SearchFilters(
        regions=args.region or None,
        categories=args.category or None,
        date_from=(date.today() - timedelta(days=args.days)) if args.days else None,
    )
    embedder = _embedder_for(settings, ws)
    reranker = CrossEncoderReranker(r.rerank_model) if r.rerank else None
    with Store(ws.db_path) as store:
        result = search(store, args.query, filters, r, embedder, reranker)
    print(f"{len(result.hits)} results | mode {result.mode} | reranked {result.reranked}")
    for note in result.notes:
        print(f"  note: {note}")
    for n, h in enumerate(result.hits, start=1):
        score = f"rerank {h.rerank_score:.2f}" if h.rerank_score is not None else "no rerank"
        print(
            f"\n[{n}] {h.title}\n    {h.source} | {h.region} {h.category} | {h.published_date} | "
            f"found by {'+'.join(h.found_by)} | {score} | {h.engine}\n    {h.url}"
        )
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    ws = require_workspace(args.workspace)
    settings = load_settings(ws.root)
    with Store(ws.db_path) as store:
        if args.action == "stats":
            for key, value in store.stats().items():
                print(f"{key:<20} {value}")
        elif args.action == "cleanup":
            days = args.days or settings.data.retention_days
            if not days:
                print("Retention is 'forever'; pass --days N to clean up anyway.")
                return 1
            result = store.cleanup(days, now=datetime.now(UTC), apply=args.apply)
            verb = "Removed" if result.applied else "Would remove"
            print(
                f"{verb} {result.items} items and {result.chunks} chunks published before "
                f"{result.cutoff.date()}"
            )
            if not args.apply and result.items:
                print("Run again with --apply to delete (consider `data backup` first).")
        elif args.action == "verify":
            rep = store.verify(repair=args.repair)
            print(
                f"orphan keyword rows {rep.orphan_fts} | orphan chunks {rep.orphan_chunks} | "
                f"chunks missing vectors {rep.missing_vectors} | chunks missing keyword rows "
                f"{rep.missing_fts} | {'clean' if rep.clean else 'NOT clean'}"
            )
            for d in rep.details:
                print(f"  {d}")
            return 0 if rep.clean or rep.repaired else 1
        elif args.action == "compact":
            before = store.path.stat().st_size
            store.compact()
            print(f"Compacted: {before:,} -> {store.path.stat().st_size:,} bytes")
        elif args.action == "reindex":
            base = settings.llm.base_url if settings.llm.provider == "ollama" else None
            embedder = make_embedder(settings.retrieval.embedding_model, base)

            def progress(done: int, total: int) -> None:
                print(f"\r  rebuilt {done}/{total}", end="", flush=True)

            count = reindex(store, embedder, settings, on_progress=progress)
            ws.set_embedding_model(embedder.name)
            print(f"\nRebuilt the index for {count} items with {embedder.name}")
        elif args.action == "backup":
            print(f"Backup written: {backup_workspace(ws.root, ws.backups_dir, store)}")
    if args.action == "restore":
        if not args.path:
            print("Pass the backup zip path: newsrag data restore PATH")
            return 1
        restore_workspace(Path(args.path), ws.root)
        print(f"Restored {args.path} into {ws.root}")
    return 0


def add_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_run = sub.add_parser("run", help="fetch, process and store new items in the workspace")
    p_run.add_argument("--limit", type=int, default=0, help="process at most N items (0 = all)")
    p_run.add_argument("--no-digest", action="store_true", help="skip writing the digest")

    p_dig = sub.add_parser("digest", help="write today's briefing from stored articles")
    p_dig.add_argument("--mode", choices=["auto", "rules", "llm"], help="override engine mode")
    p_dig.add_argument("--print", action="store_true", help="also print the Markdown")

    p_chat = sub.add_parser("chat", help="ask questions about stored news (grounded, cited)")
    p_chat.add_argument("question", nargs="?", help="ask one question and exit")
    p_chat.add_argument("--region", action="append", help="US, EU or IN (repeatable)")
    p_chat.add_argument("--category", action="append", help="category (repeatable)")
    p_chat.add_argument("--days", type=int, default=7, help="time range in days (default 7)")
    p_chat.add_argument("--mode", choices=["auto", "rules", "llm"], help="override engine mode")

    p_ui = sub.add_parser("ui", help="start the web UI (Streamlit)")
    p_ui.add_argument("--port", type=int, default=8501)

    p_tools = sub.add_parser("tools", help="list agent-ready tools and their schemas")
    p_tools.add_argument("--json", action="store_true", help="print full JSON schemas")

    p_search = sub.add_parser("search", help="hybrid search over stored items")
    p_search.add_argument("query")
    p_search.add_argument("--region", action="append", help="US, EU or IN (repeatable)")
    p_search.add_argument("--category", action="append", help="category name (repeatable)")
    p_search.add_argument("--days", type=int, help="only items from the last N days")
    p_search.add_argument("--mode", choices=["hybrid", "semantic", "keyword"])
    p_search.add_argument("--no-rerank", action="store_true")
    p_search.add_argument("--top-k", type=int)

    p_data = sub.add_parser(
        "data", help="stats, cleanup, verify, compact, reindex, backup, restore"
    )
    p_data.add_argument(
        "action",
        choices=["stats", "cleanup", "verify", "compact", "reindex", "backup", "restore"],
    )
    p_data.add_argument("path", nargs="?", help="backup zip for restore")
    p_data.add_argument("--days", type=int, help="cleanup: retention days (default: setting)")
    p_data.add_argument("--apply", action="store_true", help="cleanup: actually delete")
    p_data.add_argument("--repair", action="store_true", help="verify: fix what is found")


HANDLERS = {
    "run": cmd_run,
    "search": cmd_search,
    "data": cmd_data,
    "digest": cmd_digest,
    "chat": cmd_chat,
    "tools": cmd_tools,
    "ui": cmd_ui,
}
