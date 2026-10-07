"""Workspace folders (REQUIREMENTS FR27 to FR31).

A workspace is one folder that holds everything the app writes for one user profile.
Outside it, the app only remembers the list of recent workspace paths.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = 1
META_FILE = "workspace.json"
SUBDIRS = ("vectors", "digests", "backups", "logs")
MAX_RECENT = 10


class WorkspaceError(Exception):
    pass


class WorkspaceMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    schema_version: int = SCHEMA_VERSION
    created_at: datetime
    embedding_model: str | None = None


class Workspace:
    """An opened workspace. All paths the app writes are derived from `root`."""

    def __init__(self, root: Path, meta: WorkspaceMeta) -> None:
        self.root = root
        self.meta = meta

    @property
    def db_path(self) -> Path:
        return self.root / "newsrag.db"

    @property
    def vectors_dir(self) -> Path:
        return self.root / "vectors"

    @property
    def digests_dir(self) -> Path:
        return self.root / "digests"

    @property
    def backups_dir(self) -> Path:
        return self.root / "backups"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def save_meta(self) -> None:
        (self.root / META_FILE).write_text(self.meta.model_dump_json(indent=2), "utf-8")

    def check_embedding_model(self, model: str) -> None:
        """Record the embedding model on first use; refuse to mix models later (FR29)."""
        if self.meta.embedding_model is None:
            self.meta.embedding_model = model
            self.save_meta()
        elif self.meta.embedding_model != model:
            raise WorkspaceError(
                f"This workspace was indexed with {self.meta.embedding_model!r}, "
                f"not {model!r}. Rebuild the index to switch models."
            )


def open_workspace(root: Path, create: bool = True) -> Workspace:
    """Open the workspace at `root`, creating its layout if needed."""
    root = root.expanduser().resolve()
    meta_path = root / META_FILE
    if meta_path.exists():
        meta = WorkspaceMeta.model_validate_json(meta_path.read_text("utf-8"))
        if meta.schema_version > SCHEMA_VERSION:
            raise WorkspaceError(
                f"Workspace schema {meta.schema_version} is newer than this app "
                f"({SCHEMA_VERSION}). Update newsrag to open it."
            )
    elif not create:
        raise WorkspaceError(f"No workspace at {root}")
    else:
        if root.exists() and any(p.name != ".DS_Store" for p in root.iterdir()):
            raise WorkspaceError(
                f"{root} is not empty and is not a workspace. Choose an empty folder."
            )
        root.mkdir(parents=True, exist_ok=True)
        meta = WorkspaceMeta(name=root.name, created_at=datetime.now(UTC))
        (root / META_FILE).write_text(meta.model_dump_json(indent=2), "utf-8")

    for sub in SUBDIRS:
        (root / sub).mkdir(exist_ok=True)
    return Workspace(root, meta)


def app_config_dir() -> Path:
    """Per-user folder for the recent-workspaces list. Overridable for tests."""
    override = os.environ.get("NEWSRAG_CONFIG_DIR")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "newsrag"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home())) / "newsrag"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "newsrag"


def _recent_file() -> Path:
    return app_config_dir() / "recent.json"


def recent_workspaces() -> list[Path]:
    path = _recent_file()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [Path(p) for p in data.get("recent", []) if isinstance(p, str)]


def remember_workspace(root: Path) -> None:
    """Put `root` first in the recent list. Stores paths only."""
    root = root.expanduser().resolve()
    items = [root, *(p for p in recent_workspaces() if p != root)][:MAX_RECENT]
    path = _recent_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"recent": [str(p) for p in items]}, indent=2), "utf-8")
