"""Acceptance 13: every tool runs with no LLM and has a valid JSON schema (AG2-AG5)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from newsrag import runner as runner_mod
from newsrag.ingest import IngestReport
from newsrag.tools import REGISTRY, call_tool, tool_schemas
from tests.conftest_ctx import tool_context

NOW = datetime.now(UTC)
TODAY = NOW.date()


def test_registry_has_the_required_tools_and_flags() -> None:
    assert set(REGISTRY) == {
        "search_news",
        "get_article",
        "list_sources",
        "stats",
        "compare_periods",
        "topic_brief",
        "fetch_now",
        "cleanup",
    }
    changing = {n for n, t in REGISTRY.items() if t.changes_data}
    assert changing == {"fetch_now", "cleanup"}


def test_schemas_are_valid_json_objects() -> None:
    for schema in tool_schemas():
        text = json.dumps(schema)
        assert json.loads(text) == schema
        s = schema["input_schema"]
        assert s["type"] == "object" and "properties" in s
        assert schema["description"]
    search = next(s for s in tool_schemas() if s["name"] == "search_news")
    assert search["input_schema"]["required"] == ["query"]


def _call(ctx: Any, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(call_tool(ctx, name, args))


def test_read_only_tools_run_without_llm(tmp_path: Path) -> None:
    with tool_context(tmp_path, NOW) as ctx:
        res = _call(ctx, "search_news", {"query": "interest rates", "regions": ["IN"]})
        assert res["hits"] and {h["region"] for h in res["hits"]} == {"IN"}

        item_id = res["hits"][0]["item_id"]
        art = _call(ctx, "get_article", {"item_id": item_id})
        assert art["found"] and art["article"]["item_id"] == item_id
        assert _call(ctx, "get_article", {"item_id": "nope"}) == {"found": False, "article": None}

        src = _call(ctx, "list_sources", {"region": "IN", "category": "Finance"})
        assert src["sources"] and all(s["region"] == "IN" for s in src["sources"])

        st = _call(ctx, "stats", {})
        assert st["total"] == 6 and st["by_region"]["IN"] == 2 and st["by_engine"] == {"rules": 6}
        recent = _call(ctx, "stats", {"date_from": (TODAY - timedelta(days=3)).isoformat()})
        assert recent["total"] == 5

        cmp = _call(
            ctx,
            "compare_periods",
            {
                "query": "energy policy",
                "period_a": {"date_from": str(TODAY - timedelta(days=2)), "date_to": str(TODAY)},
                "period_b": {
                    "date_from": str(TODAY - timedelta(days=30)),
                    "date_to": str(TODAY - timedelta(days=10)),
                },
            },
        )
        old_titles = {h["title"] for h in cmp["period_b"]["hits"]}
        assert "Old story about energy policy from last month" in old_titles
        assert cmp["count_b"] == len(cmp["period_b"]["hits"])


def test_invalid_arguments_are_rejected(tmp_path: Path) -> None:
    with tool_context(tmp_path, NOW) as ctx:
        with pytest.raises(ValueError):
            _call(ctx, "search_news", {"query": ""})
        with pytest.raises(ValueError):
            _call(ctx, "search_news", {"query": "x", "unexpected": 1})


def test_cleanup_tool_is_a_dry_run_by_default(tmp_path: Path) -> None:
    with tool_context(tmp_path, NOW) as ctx:
        preview = _call(ctx, "cleanup", {"retention_days": 10})
        assert preview["items"] == 1 and preview["applied"] is False
        assert _call(ctx, "stats", {})["total"] == 6
        applied = _call(ctx, "cleanup", {"retention_days": 10, "dry_run": False})
        assert applied["applied"] and _call(ctx, "stats", {})["total"] == 5


def test_fetch_now_uses_the_shared_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_run(store, cfg, settings, keys, embedder, **kw):  # type: ignore[no-untyped-def]
        seen.update(kw)
        return runner_mod.RunOutcome(
            started_at=NOW,
            finished_at=NOW,
            lookback_hours=48,
            fetched=0,
            kept_after_filters=0,
            engine="Rules",
            engine_reason="test",
            ingest=IngestReport(),
        )

    monkeypatch.setattr("newsrag.tools.core.run_ingest", fake_run)
    with tool_context(tmp_path, NOW) as ctx:
        out = _call(ctx, "fetch_now", {"regions": ["IN"], "limit": 3})
    assert out["engine"] == "Rules" and seen == {"regions": ["IN"], "limit": 3}
