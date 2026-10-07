"""Choose the engine once per run (REQUIREMENTS FR11, 5.3.1).

mode = rules -> RuleEngine.
mode = auto  -> the configured provider if it works; else Anthropic if a key is present;
                else RuleEngine. The reason is always reported.
mode = llm   -> the configured provider, or EngineUnavailable with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from newsrag.config import AppConfig
from newsrag.engines.base import Engine
from newsrag.engines.llm import LLMEngine
from newsrag.engines.rules import RuleEngine
from newsrag.llm import LLMClient, LLMUnavailable, Task, build_client
from newsrag.secrets import KeyStore
from newsrag.settings import LLMSettings


class EngineUnavailable(Exception):
    pass


@dataclass
class EngineChoice:
    engine: Engine
    reason: str
    client: LLMClient | None = None

    async def aclose(self) -> None:
        if self.client is not None:
            await self.client.aclose()


async def _try(
    s: LLMSettings, keys: KeyStore, task: Task, provider: str
) -> tuple[LLMClient | None, str]:
    try:
        client = await build_client(s, keys, task, provider=provider)
    except LLMUnavailable as exc:
        return None, f"{provider}: {exc}"
    try:
        await client.check()
    except LLMUnavailable as exc:
        await client.aclose()
        return None, f"{provider}: {exc}"
    return client, ""


async def select_engine(
    s: LLMSettings, cfg: AppConfig, keys: KeyStore, task: Task = "processing"
) -> EngineChoice:
    rules = RuleEngine(cfg)
    if s.mode == "rules":
        return EngineChoice(rules, "rules mode selected in Settings")

    reasons: list[str] = []
    candidates = [s.provider]
    if s.mode == "auto" and s.provider != "anthropic" and keys.has("anthropic"):
        candidates.append("anthropic")
    for provider in candidates:
        client, why = await _try(s, keys, task, provider)
        if client is not None:
            engine = LLMEngine(client, cfg, rules, keys)
            note = "" if provider == s.provider else f" ({'; '.join(reasons)})"
            return EngineChoice(engine, f"using {engine.label}{note}", client)
        reasons.append(keys.redact(why))

    if s.mode == "llm":
        raise EngineUnavailable("; ".join(reasons))
    return EngineChoice(rules, "no LLM available, using rules (" + "; ".join(reasons) + ")")
