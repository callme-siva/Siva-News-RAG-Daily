"""CLI commands that use the workspace store: run, search, data (REQUIREMENTS FR23-FR25,
FR38-FR43, FR16)."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from newsrag.config import load_config
from newsrag.embeddings import CrossEncoderReranker, make_embedder
from newsrag.engines import EngineUnavailable, select_engine
from newsrag.ingest import ingest, reindex
from newsrag.logging_setup import setup_logging
from newsrag.pipeline import collect
from newsrag.search import SearchFilters, search
from newsrag.secrets import KEYS
from newsrag.settings import Settings, load_settings
from newsrag.store import Store, backup_workspace, restore_workspace
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


def catchup_hours(settings: Settings, last_run: datetime | None, now: datetime) -> int:
    """FR25: look back to the last run, at least max_age_hours, at most max_catchup_days."""
    base = settings.sources.max_age_hours
    if last_run is None:
        return base
    since = int((now - last_run).total_seconds() // 3600) + 1
    return max(base, min(since, settings.sources.max_catchup_days * 24))


def cmd_run(args: argparse.Namespace) -> int:
    ws = require_workspace(args.workspace)
    settings = load_settings(ws.root)
    cfg = load_config()
    KEYS.load_from_env()
    log = setup_logging(ws.logs_dir)
    embedder = _embedder_for(settings, ws)

    async def run() -> int:
        started = datetime.now(UTC)
        with Store(ws.db_path) as store:
            hours = catchup_hours(settings, store.last_run(), started)
            if hours > settings.sources.max_age_hours:
                print(f"Catching up: looking back {hours} hours since the last run")
            fetch_settings = settings.model_copy(deep=True)
            fetch_settings.sources.max_age_hours = hours
            report = await collect(cfg, fetch_settings, KEYS, now=started)
            items = report.items[: args.limit] if args.limit else report.items
            try:
                choice = await select_engine(settings.llm, cfg, KEYS, task="processing")
            except EngineUnavailable as exc:
                print(f"LLM not available: {exc}")
                return 1
            print(f"Engine: {choice.engine.label} | {choice.reason}")

            def progress(done: int, total: int) -> None:
                print(f"\r  processed {done}/{total}", end="", flush=True)

            try:
                result = await ingest(
                    items,
                    store,
                    choice.engine,
                    embedder,
                    settings,
                    KEYS,
                    now=started,
                    on_progress=progress,
                )
            finally:
                await choice.aclose()
            print()
            finished = datetime.now(UTC)
            failed_sources = [s.source for s in report.sources if s.status == "error"]
            summary = {
                "fetched": report.fetched,
                "kept_after_filters": len(report.items),
                "processed_limit": args.limit,
                "engine": choice.engine.label,
                "engine_reason": choice.reason,
                "failed_sources": failed_sources,
                **result.model_dump(),
            }
            store.record_run(started, finished, summary)
            log.info("Run finished: %s", {k: v for k, v in summary.items() if k != "errors"})
            print(
                f"New {result.new} | skipped as seen {result.skipped_seen} | merged duplicates "
                f"{result.merged_duplicate} | updated {result.updated} | not relevant "
                f"{result.irrelevant} | failed {result.failed}"
            )
            if result.by_engine:
                print("Written by: " + ", ".join(f"{k} {v}" for k, v in result.by_engine.items()))
            if failed_sources:
                print("Sources with errors: " + ", ".join(failed_sources))
            if settings.data.auto_cleanup and settings.data.retention_days:
                c = store.cleanup(settings.data.retention_days, now=finished, apply=True)
                if c.items:
                    print(f"Auto cleanup removed {c.items} items older than {c.cutoff.date()}")
        return 0

    return asyncio.run(run())


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


HANDLERS = {"run": cmd_run, "search": cmd_search, "data": cmd_data}
