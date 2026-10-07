from __future__ import annotations

import json
from pathlib import Path

import pytest

from newsrag.__main__ import main
from newsrag.workspace import (
    SUBDIRS,
    WorkspaceError,
    app_config_dir,
    open_workspace,
    recent_workspaces,
    remember_workspace,
)


def test_create_and_reopen(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "personal")
    assert ws.meta.name == "personal"
    for sub in SUBDIRS:
        assert (ws.root / sub).is_dir()
    again = open_workspace(ws.root, create=False)
    assert again.meta == ws.meta


def test_refuses_non_empty_non_workspace_folder(tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "notes.txt").write_text("hi")
    with pytest.raises(WorkspaceError, match="not empty"):
        open_workspace(tmp_path / "x")


def test_missing_workspace_without_create(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError):
        open_workspace(tmp_path / "nope", create=False)


def test_embedding_model_is_locked(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "w")
    ws.check_embedding_model("model-a")
    ws.check_embedding_model("model-a")
    with pytest.raises(WorkspaceError, match="Rebuild the index"):
        ws.check_embedding_model("model-b")
    assert open_workspace(ws.root).meta.embedding_model == "model-a"


def test_recent_list_stores_paths_only_and_dedupes(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    remember_workspace(a)
    remember_workspace(b)
    remember_workspace(a)
    assert recent_workspaces() == [a.resolve(), b.resolve()]
    data = json.loads((app_config_dir() / "recent.json").read_text())
    assert data == {"recent": [str(a.resolve()), str(b.resolve())]}


def test_cli_writes_only_inside_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Acceptance criterion 10: everything the app writes stays in the workspace,
    except the remembered path in the app-config folder."""
    root = tmp_path / "ws"
    before = {p for p in tmp_path.rglob("*")}
    assert main(["--workspace", str(root), "workspace"]) == 0
    after = {p for p in tmp_path.rglob("*")}
    appcfg = app_config_dir().resolve()
    outside = [
        p
        for p in after - before
        if not p.resolve().is_relative_to(root.resolve())
        and not p.resolve().is_relative_to(appcfg)
        and p.resolve() != appcfg
    ]
    assert outside == []
    assert (root / "settings.json").exists() and (root / "workspace.json").exists()
    assert "Workspace:" in capsys.readouterr().out


def test_cli_unbuilt_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ui"]) == 2
    assert "not built yet" in capsys.readouterr().out
