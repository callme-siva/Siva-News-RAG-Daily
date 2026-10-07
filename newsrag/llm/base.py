"""Provider-neutral LLM client interface (REQUIREMENTS FR10, FR11, 5.3.1)."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol

Task = Literal["processing", "chat", "digest"]


class LLMError(Exception):
    """A call failed (HTTP error, timeout, refusal, empty output). Message is safe to log."""


class LLMUnavailable(LLMError):
    """The provider cannot be used right now (no key, server down, model missing)."""


class LLMClient(Protocol):
    provider: str
    model: str

    async def complete(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any] | None,
        max_tokens: int,
        task: Task,
    ) -> str:
        """Return the model's text. With `schema`, the text should be a JSON object."""
        ...

    async def check(self) -> None:
        """Raise LLMUnavailable if the provider or model cannot be reached."""
        ...

    async def aclose(self) -> None: ...


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model text, tolerating code fences or surrounding prose."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = _OBJECT.search(cleaned)
        if not match:
            raise ValueError("no JSON object in the reply") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("reply is JSON but not an object")
    return value
