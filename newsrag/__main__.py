"""Command line: `python -m newsrag <command>` (REQUIREMENTS FR23, FR26, FR28)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from newsrag import __version__, cli_store
from newsrag.config import load_config
from newsrag.engines import EngineUnavailable, process_all, select_engine
from newsrag.logging_setup import setup_logging
from newsrag.pipeline import collect
from newsrag.pipeline.fetch import fetch_all
from newsrag.secrets import KEYS
from newsrag.settings import Settings, load_settings, save_settings
from newsrag.workspace import (
    WorkspaceError,
    open_workspace,
    recent_workspaces,
    remember_workspace,
)

NOT_BUILT = {"chat": 5, "ui": 6}


def _resolve_workspace(arg: str | None) -> Path | None:
    if arg:
        return Path(arg)
    recent = recent_workspaces()
    return recent[0] if recent else None


def _cmd_workspace(args: argparse.Namespace) -> int:
    root = _resolve_workspace(args.workspace)
    if root is None:
        print("No workspace yet. Create one with: newsrag --workspace PATH workspace")
        return 1
    ws = open_workspace(root)
    remember_workspace(ws.root)
    settings = load_settings(ws.root)
    save_settings(ws.root, settings)
    setup_logging(ws.logs_dir).info("Opened workspace %s", ws.root)
    print(f"Workspace: {ws.root}")
    print(f"Schema:    {ws.meta.schema_version}")
    print(f"Embedding: {ws.meta.embedding_model or 'not indexed yet'}")
    return 0


def _cmd_config(_: argparse.Namespace) -> int:
    cfg = load_config()
    print("Regions:    " + ", ".join(f"{r.code} ({r.name})" for r in cfg.regions))
    print("Categories: " + ", ".join(c.name for c in cfg.categories))
    print(f"Sources:    {len(cfg.sources)} configured")
    found = KEYS.load_from_env()
    print("Keys found: " + (", ".join(found) if found else "none (rules mode will be used)"))
    return 0


def _settings_for(args: argparse.Namespace) -> Settings:
    root = _resolve_workspace(args.workspace)
    return load_settings(open_workspace(root, create=False).root) if root else Settings()


def _cmd_sources(args: argparse.Namespace) -> int:
    cfg = load_config()
    KEYS.load_from_env()
    if not args.check:
        for s in cfg.sources:
            state = "on " if s.enabled else "off"
            where = s.url or s.query or s.api
            print(f"{state} {s.region} {s.category:<10} {s.type:<5} {s.name:<24} {where}")
        return 0
    sources = [s for s in cfg.sources if s.enabled or args.all]
    results = asyncio.run(fetch_all(sources, KEYS, max_age_hours=24 * 30))
    failed = 0
    for src, res in zip(sources, results, strict=True):
        failed += res.status == "error"
        detail = res.error or f"{res.count} items"
        print(f"{res.status:<7} {src.region} {src.name:<24} {res.elapsed_s:>5.1f}s  {detail}")
    return 1 if failed else 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    cfg = load_config()
    settings = _settings_for(args)
    KEYS.load_from_env()
    report = asyncio.run(collect(cfg, settings, KEYS))
    for r in report.sources:
        if r.status != "ok":
            print(f"  {r.status:<7} {r.source}: {r.error or 'no items'}")
    f = report.filter
    print(
        f"Fetched {report.fetched} | too old {f.too_old} | excluded {f.excluded} | "
        f"off-topic {f.not_relevant} | merged duplicates {report.merged_duplicates} | "
        f"over cap {report.capped} | kept {len(report.items)}"
    )
    print("(dry run: use `newsrag run` to store)")
    for item in report.items[: args.show]:
        also = f" (+{len(item.also_reported_by)})" if item.also_reported_by else ""
        when = item.published_at.strftime("%d %b %H:%M")
        print(f"  {item.region} {item.category:<10} {when}  {item.title[:80]}{also}")
    return 0


def _cmd_process(args: argparse.Namespace) -> int:
    cfg = load_config()
    settings = _settings_for(args)
    if args.mode:
        settings.llm.mode = args.mode
    KEYS.load_from_env()

    async def run() -> int:
        report = await collect(cfg, settings, KEYS)
        items = report.items[: args.limit]
        try:
            choice = await select_engine(settings.llm, cfg, KEYS, task="processing")
        except EngineUnavailable as exc:
            print(f"LLM not available: {exc}")
            return 1
        print(f"Engine: {choice.engine.label} | {choice.reason}")
        try:
            results = await process_all(choice.engine, items, concurrency=settings.llm.concurrency)
        finally:
            await choice.aclose()
        for item, p in zip(items, results, strict=True):
            tag = p.engine.value + (f", fallback: {p.fallback_reason}" if p.fallback_reason else "")
            flag = "" if p.relevant else "  [not relevant]"
            print(f"\n[{tag}] {item.region} {p.category}: {item.title[:90]}{flag}")
            print(f"  {p.summary[:300]}")
            if p.key_facts:
                print(f"  facts: {' | '.join(f[:80] for f in p.key_facts[:3])}")
            names = p.entities.flat()
            if names:
                print(f"  entities: {', '.join(names[:8])}")
        print("\n(dry run: use `newsrag run` to store)")
        return 0

    return asyncio.run(run())


def _cmd_recent(_: argparse.Namespace) -> int:
    for path in recent_workspaces():
        print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="newsrag", description=__doc__)
    parser.add_argument("--version", action="version", version=f"newsrag {__version__}")
    parser.add_argument("--workspace", help="workspace folder (default: most recent)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("workspace", help="create or open a workspace and show its details")
    sub.add_parser("recent", help="list recent workspaces")
    sub.add_parser("config", help="show regions, categories, sources and detected keys")
    p_src = sub.add_parser("sources", help="list sources, or --check them live")
    p_src.add_argument("--check", action="store_true", help="fetch each source and report status")
    p_src.add_argument("--all", action="store_true", help="with --check, include disabled sources")
    p_fetch = sub.add_parser("fetch", help="fetch, filter, dedupe and rank (dry run, no storage)")
    p_fetch.add_argument("--show", type=int, default=20, help="how many items to print")
    p_proc = sub.add_parser("process", help="fetch then summarise and tag items (dry run)")
    p_proc.add_argument("--limit", type=int, default=5, help="how many items to process")
    p_proc.add_argument("--mode", choices=["auto", "rules", "llm"], help="override engine mode")
    cli_store.add_parsers(sub)
    for name in NOT_BUILT:
        sub.add_parser(name, help=f"(available from stage {NOT_BUILT[name]})")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "workspace": _cmd_workspace,
        "config": _cmd_config,
        "recent": _cmd_recent,
        "sources": _cmd_sources,
        "fetch": _cmd_fetch,
        "process": _cmd_process,
        **cli_store.HANDLERS,
    }
    if args.command in NOT_BUILT:
        print(f"'{args.command}' is not built yet (stage {NOT_BUILT[args.command]}).")
        return 2
    try:
        return handlers[args.command](args)
    except WorkspaceError as exc:
        print(f"Workspace error: {KEYS.redact(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
