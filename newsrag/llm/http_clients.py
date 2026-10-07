"""LLM providers reached over plain HTTP: Ollama, OpenAI-compatible servers, Google Gemini."""

from __future__ import annotations

from typing import Any

import httpx

from newsrag.llm.base import LLMError, LLMUnavailable, Task


class _HttpLLM:
    provider = ""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        timeout_s: float,
        temperature: float,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self._http = httpx.AsyncClient(
            timeout=timeout_s, headers=headers or {}, transport=transport
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._http.post(url, json=body)
        except httpx.TimeoutException as exc:
            raise LLMError(f"{self.provider} timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"{self.provider} unreachable: {type(exc).__name__}") from exc
        if resp.status_code >= 400:
            raise LLMError(f"{self.provider} returned HTTP {resp.status_code}")
        data = resp.json()
        if not isinstance(data, dict):
            raise LLMError(f"{self.provider} returned an unexpected response")
        return data

    async def _get(self, url: str) -> dict[str, Any]:
        try:
            resp = await self._http.get(url)
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"{self.provider} unreachable at {self.base_url}") from exc
        if resp.status_code in (401, 403):
            raise LLMUnavailable(f"{self.provider} rejected the API key")
        if resp.status_code >= 400:
            raise LLMUnavailable(f"{self.provider} returned HTTP {resp.status_code}")
        data = resp.json()
        return data if isinstance(data, dict) else {}


class OllamaClient(_HttpLLM):
    """Ollama's native API. `format` accepts a JSON schema for structured output."""

    provider = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120,
        temperature: float = 0.2,
        context_length: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            timeout_s=timeout_s,
            temperature=temperature,
            transport=transport,
        )
        self.context_length = context_length

    async def list_models(self) -> list[str]:
        data = await self._get(f"{self.base_url}/api/tags")
        return [m["name"] for m in data.get("models", []) if isinstance(m, dict) and "name" in m]

    async def check(self) -> None:
        models = await self.list_models()
        if not any(m == self.model or m.split(":")[0] == self.model for m in models):
            listed = ", ".join(models) or "none"
            raise LLMUnavailable(f"model {self.model!r} is not installed in Ollama ({listed})")

    async def complete(
        self, *, system: str, user: str, schema: dict[str, Any] | None, max_tokens: int, task: Task
    ) -> str:
        options: dict[str, Any] = {"temperature": self.temperature, "num_predict": max_tokens}
        if self.context_length:
            options["num_ctx"] = self.context_length
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": options,
        }
        if schema is not None:
            body["format"] = schema
        data = await self._post(f"{self.base_url}/api/chat", body)
        text = (data.get("message") or {}).get("content", "")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("ollama returned an empty reply")
        return text


class OpenAICompatibleClient(_HttpLLM):
    """Any /v1/chat/completions server: hosted (with key) or local (LM Studio, llama.cpp, vLLM)."""

    provider = "openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 120,
        temperature: float = 0.2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        super().__init__(
            model=model,
            base_url=base_url,
            timeout_s=timeout_s,
            temperature=temperature,
            headers=headers,
            transport=transport,
        )

    async def check(self) -> None:
        data = await self._get(f"{self.base_url}/models")
        ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
        if ids and self.model not in ids:
            raise LLMUnavailable(f"model {self.model!r} not offered by {self.base_url}")

    async def complete(
        self, *, system: str, user: str, schema: dict[str, Any] | None, max_tokens: int, task: Task
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema, "strict": True},
            }
        data = await self._post(f"{self.base_url}/chat/completions", body)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected chat/completions response shape") from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMError("empty reply")
        return text


class GeminiClient(_HttpLLM):
    """Google Gemini generateContent. The key goes in the x-goog-api-key header, not the URL."""

    provider = "gemini"
    API = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout_s: float = 60,
        temperature: float = 0.2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=self.API,
            timeout_s=timeout_s,
            temperature=temperature,
            headers={"x-goog-api-key": api_key},
            transport=transport,
        )

    async def check(self) -> None:
        await self._get(f"{self.base_url}/models/{self.model}")

    async def complete(
        self, *, system: str, user: str, schema: dict[str, Any] | None, max_tokens: int, task: Task
    ) -> str:
        config: dict[str, Any] = {"temperature": self.temperature, "maxOutputTokens": max_tokens}
        if schema is not None:
            config["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": config,
        }
        data = await self._post(f"{self.base_url}/models/{self.model}:generateContent", body)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected Gemini response shape (possibly blocked)") from exc
        if not text.strip():
            raise LLMError("empty reply")
        return text
