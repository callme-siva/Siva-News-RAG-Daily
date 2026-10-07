"""Engines turn items into summaries and tags. See base.Engine."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from newsrag.engines.base import Engine
from newsrag.engines.rules import RuleEngine
from newsrag.engines.select import EngineChoice, EngineUnavailable, select_engine
from newsrag.models import Item, Processed

__all__ = [
    "Engine",
    "EngineChoice",
    "EngineUnavailable",
    "RuleEngine",
    "process_all",
    "select_engine",
]

Progress = Callable[[int, int], None]


async def process_all(
    engine: Engine,
    items: list[Item],
    *,
    concurrency: int = 1,
    on_progress: Progress | None = None,
    stop: asyncio.Event | None = None,
) -> list[Processed]:
    """Process items with bounded concurrency, reporting progress (FR11e).
    When `stop` is set, items not yet started are skipped; results keep input order
    for the items that were processed."""
    gate = asyncio.Semaphore(max(1, concurrency))
    results: list[Processed | None] = [None] * len(items)
    done = 0

    async def one(i: int, item: Item) -> None:
        nonlocal done
        async with gate:
            if stop is not None and stop.is_set():
                return
            results[i] = await engine.process(item)
            done += 1
            if on_progress is not None:
                on_progress(done, len(items))

    await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    return [r for r in results if r is not None]
