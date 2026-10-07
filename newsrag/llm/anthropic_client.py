"""Claude via the official `anthropic` SDK (optional dependency: `pip install newsrag[anthropic]`).

- JSON tasks use structured outputs (`output_config.format`), so replies are schema-valid JSON.
- Effort is set per task: `low` for per-article processing, `medium` for chat and digest.
- `temperature` is never sent: current Claude models reject sampling parameters.
- On models that support it, server-side refusal fallbacks are enabled; a final refusal
  raises LLMError so the engine falls back to rules for that item.
"""

from __future__ import annotations

from typing import Any

from newsrag.llm.base import LLMError, LLMUnavailable, Task

DEFAULT_MODEL = "claude-opus-5-5"

# Models that accept output_config.effort.
_EFFORT_MODELS = (
    "claude-fable-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
)
# Models that accept fallbacks="default" on the Claude API.
_FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_EFFORT: dict[Task, str] = {"processing": "low", "chat": "medium", "digest": "medium"}


def _supports_effort(model: str) -> bool:
    return model.startswith(_EFFORT_MODELS)


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, api_key: str, model: str | None, timeout_s: float, sdk: Any = None) -> None:
        self.model = model or DEFAULT_MODEL
        if sdk is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMUnavailable(
                    "the anthropic package is not installed (pip install newsrag[anthropic])"
                ) from exc
            sdk = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_s, max_retries=2)
        self._sdk = sdk

    def request(
        self, *, system: str, user: str, schema: dict[str, Any] | None, max_tokens: int, task: Task
    ) -> dict[str, Any]:
        """Build the request kwargs. Separate so tests can assert on the exact shape."""
        output_config: dict[str, Any] = {}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if _supports_effort(self.model):
            output_config["effort"] = _EFFORT[task]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if output_config:
            kwargs["output_config"] = output_config
        if self.model in _FALLBACK_MODELS:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    async def complete(
        self, *, system: str, user: str, schema: dict[str, Any] | None, max_tokens: int, task: Task
    ) -> str:
        kwargs = self.request(
            system=system, user=user, schema=schema, max_tokens=max_tokens, task=task
        )
        try:
            response = await self._sdk.beta.messages.create(**kwargs)
        except Exception as exc:
            raise LLMError(_describe(exc)) from exc
        if response.stop_reason == "refusal":
            raise LLMError("the model declined this item (refusal)")
        if response.stop_reason == "max_tokens":
            raise LLMError("reply hit max_tokens before finishing")
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        if not text.strip():
            raise LLMError("empty reply")
        return text

    async def check(self) -> None:
        try:
            await self._sdk.models.retrieve(self.model)
        except Exception as exc:
            raise LLMUnavailable(_describe(exc)) from exc

    async def aclose(self) -> None:
        close = getattr(self._sdk, "close", None)
        if close is not None:
            await close()


def _describe(exc: Exception) -> str:
    """Short, key-free description. Status codes and class names only, never request data."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if status == 401:
        return "authentication failed: check the Anthropic API key"
    if status == 404:
        return "model not found: check the model name"
    if status == 429:
        return "rate limited by Anthropic"
    return f"{name} (HTTP {status})" if status else name
