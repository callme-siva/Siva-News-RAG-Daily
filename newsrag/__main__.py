"""Command line: `python -m newsrag <command>` (REQUIREMENTS FR23, FR26, FR28)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from newsrag import __version__
from newsrag.config import load_config
from newsrag.logging_setup import setup_logging
from newsrag.secrets import KEYS
from newsrag.settings import load_settings, save_settings
from newsrag.workspace import (
    WorkspaceError,
    open_workspace,
    recent_workspaces,
    remember_workspace,
)

NOT_BUILT = {"run": 2, "chat": 5, "ui": 6}


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
    for name in NOT_BUILT:
        sub.add_parser(name, help=f"(available from stage {NOT_BUILT[name]})")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"workspace": _cmd_workspace, "config": _cmd_config, "recent": _cmd_recent}
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
