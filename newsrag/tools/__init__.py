"""Tool registry (REQUIREMENTS AG3, AG4). Schemas are generated from the pydantic models, so
the description an LLM sees can never drift from the real function signature."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from newsrag.tools import core
from newsrag.tools.context import ToolContext

__all__ = ["REGISTRY", "Tool", "ToolContext", "call_tool", "tool_schemas"]


@dataclass(frozen=True)
class Tool:
    name: str
    fn: Callable[..., Any]
    input_model: type[BaseModel]
    changes_data: bool

    @property
    def description(self) -> str:
        return (self.fn.__doc__ or "").strip().splitlines()[0]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "changes_data": self.changes_data,
            "input_schema": self.input_model.model_json_schema(),
        }


REGISTRY: dict[str, Tool] = {
    t.name: t
    for t in (
        Tool("search_news", core.search_news, core.SearchNewsInput, False),
        Tool("get_article", core.get_article, core.GetArticleInput, False),
        Tool("list_sources", core.list_sources, core.ListSourcesInput, False),
        Tool("stats", core.stats, core.StatsInput, False),
        Tool("compare_periods", core.compare_periods, core.ComparePeriodsInput, False),
        Tool("topic_brief", core.topic_brief, core.TopicBriefInput, False),
        Tool("fetch_now", core.fetch_now, core.FetchNowInput, True),
        Tool("cleanup", core.cleanup, core.CleanupInput, True),
    )
}


def tool_schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in REGISTRY.values()]


async def call_tool(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate arguments, run the tool, return its output as JSON-safe data."""
    tool = REGISTRY[name]
    args = tool.input_model.model_validate(arguments)
    result = tool.fn(ctx, args)
    if inspect.isawaitable(result):
        result = await result
    assert isinstance(result, BaseModel)
    return result.model_dump(mode="json")
