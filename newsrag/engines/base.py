"""The engine interface shared by RuleEngine and LLMEngine (REQUIREMENTS R1, FR8)."""

from __future__ import annotations

from typing import Protocol

from newsrag.models import EngineName, Item, Processed


class Engine(Protocol):
    name: EngineName
    label: str  # shown in the UI badge, e.g. "Rules" or "LLM · Ollama (llama3.2)"

    async def process(self, item: Item) -> Processed:
        """Summarise, tag and judge relevance for one item. Never raises for one bad item:
        LLMEngine falls back to rules and records `fallback_reason`."""
        ...
