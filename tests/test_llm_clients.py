from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from newsrag.llm import LLMError, LLMUnavailable, build_client, parse_json_object
from newsrag.llm.anthropic_client import FALLBACK_BETA, AnthropicClient
from newsrag.llm.http_clients import GeminiClient, OllamaClient, OpenAICompatibleClient
from newsrag.secrets import KeyStore
from newsrag.settings import LLMSettings

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


def test_parse_json_object_variants() -> None:
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"a": 1} thanks') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("[1, 2]")
    with pytest.raises(ValueError):
        parse_json_object("nothing here")


# ---------- Anthropic (fake SDK object, no network) ----------


class FakeSDK:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.kwargs: dict[str, Any] = {}
        self._response = response
        self._error = error
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(retrieve=self._retrieve)

    async def _create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self._error:
            raise self._error
        return self._response

    async def _retrieve(self, model: str) -> Any:
        if self._error:
            raise self._error
        return SimpleNamespace(id=model)


def _response(text: str, stop: str = "end_turn") -> Any:
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])


def test_anthropic_request_shape_default_model() -> None:
    sdk = FakeSDK(_response('{"a": "x"}'))
    client = AnthropicClient("k", None, 30, sdk=sdk)
    out = asyncio.run(
        client.complete(system="S", user="U", schema=SCHEMA, max_tokens=500, task="processing")
    )
    assert out == '{"a": "x"}'
    kw = sdk.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"] == {
        "format": {"type": "json_schema", "schema": SCHEMA},
        "effort": "low",
    }
    assert kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default"
    assert "temperature" not in kw
    assert kw["system"] == "S" and kw["messages"] == [{"role": "user", "content": "U"}]


def test_anthropic_haiku_gets_no_effort_or_fallbacks() -> None:
    client = AnthropicClient("k", "claude-haiku-4-5", 30, sdk=FakeSDK())
    kw = client.request(system="S", user="U", schema=None, max_tokens=10, task="chat")
    assert "output_config" not in kw and "betas" not in kw and "fallbacks" not in kw


def test_anthropic_chat_effort_is_medium() -> None:
    client = AnthropicClient("k", "claude-sonnet-5-5", 30, sdk=FakeSDK())
    kw = client.request(system="S", user="U", schema=None, max_tokens=10, task="chat")
    assert kw["output_config"] == {"effort": "medium"}


@pytest.mark.parametrize(
    ("stop", "message"), [("refusal", "declined"), ("max_tokens", "max_tokens")]
)
def test_anthropic_bad_stop_reasons_raise(stop: str, message: str) -> None:
    client = AnthropicClient("k", None, 30, sdk=FakeSDK(_response("{}", stop)))
    with pytest.raises(LLMError, match=message):
        asyncio.run(client.complete(system="", user="", schema=None, max_tokens=5, task="chat"))


def test_anthropic_errors_are_described_without_request_data() -> None:
    err = Exception("boom with sk-ant-api03-SECRETSECRETSECRET")
    err.status_code = 401  # type: ignore[attr-defined]
    client = AnthropicClient("k", None, 30, sdk=FakeSDK(error=err))
    with pytest.raises(LLMError) as info:
        asyncio.run(client.complete(system="", user="", schema=None, max_tokens=5, task="chat"))
    assert "authentication failed" in str(info.value) and "SECRET" not in str(info.value)
    with pytest.raises(LLMUnavailable):
        asyncio.run(client.check())


# ---------- HTTP providers (mock transport) ----------


def _transport(routes: dict[str, Any], seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        key = f"{request.method} {request.url.path}"
        value = routes[key]
        if isinstance(value, httpx.Response):
            return value
        return httpx.Response(200, json=value)

    return httpx.MockTransport(handler)


def test_ollama_complete_and_check() -> None:
    seen: list[httpx.Request] = []
    t = _transport(
        {
            "POST /api/chat": {"message": {"role": "assistant", "content": '{"a": "x"}'}},
            "GET /api/tags": {"models": [{"name": "llama3.2:latest"}]},
        },
        seen,
    )
    client = OllamaClient(model="llama3.2", transport=t, context_length=8192, temperature=0.1)

    async def run() -> str:
        await client.check()
        out = await client.complete(
            system="S", user="U", schema=SCHEMA, max_tokens=300, task="processing"
        )
        await client.aclose()
        return out

    assert asyncio.run(run()) == '{"a": "x"}'
    body = json.loads(seen[-1].content)
    assert body["format"] == SCHEMA and body["stream"] is False
    assert body["options"] == {"temperature": 0.1, "num_predict": 300, "num_ctx": 8192}


def test_ollama_missing_model_and_down_server() -> None:
    seen: list[httpx.Request] = []
    t = _transport({"GET /api/tags": {"models": [{"name": "mistral:latest"}]}}, seen)
    with pytest.raises(LLMUnavailable, match="not installed"):
        asyncio.run(OllamaClient(model="llama3.2", transport=t).check())

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailable, match="unreachable"):
        asyncio.run(OllamaClient(model="x", transport=httpx.MockTransport(down)).check())


def test_openai_compatible_request_and_auth_header() -> None:
    seen: list[httpx.Request] = []
    t = _transport(
        {"POST /v1/chat/completions": {"choices": [{"message": {"content": '{"a": "y"}'}}]}},
        seen,
    )
    client = OpenAICompatibleClient(
        model="local-model", base_url="http://localhost:1234/v1", api_key="k123", transport=t
    )
    out = asyncio.run(
        client.complete(system="S", user="U", schema=SCHEMA, max_tokens=100, task="processing")
    )
    assert out == '{"a": "y"}'
    body = json.loads(seen[0].content)
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert seen[0].headers["authorization"] == "Bearer k123"


def test_gemini_key_in_header_not_url() -> None:
    seen: list[httpx.Request] = []
    path = "POST /v1beta/models/gemini-test:generateContent"
    t = _transport({path: {"candidates": [{"content": {"parts": [{"text": '{"a": 1}'}]}}]}}, seen)
    client = GeminiClient(model="gemini-test", api_key="gkey-123456", transport=t)
    out = asyncio.run(
        client.complete(system="S", user="U", schema=SCHEMA, max_tokens=50, task="processing")
    )
    assert out == '{"a": 1}'
    assert "gkey-123456" not in str(seen[0].url)
    assert seen[0].headers["x-goog-api-key"] == "gkey-123456"


def test_http_error_status_is_llm_error() -> None:
    seen: list[httpx.Request] = []
    t = _transport({"POST /api/chat": httpx.Response(500)}, seen)
    with pytest.raises(LLMError, match="HTTP 500"):
        asyncio.run(
            OllamaClient(model="m", transport=t).complete(
                system="", user="", schema=None, max_tokens=5, task="chat"
            )
        )


# ---------- build_client ----------


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        (LLMSettings(provider="anthropic"), "no Anthropic API key"),
        (LLMSettings(provider="gemini"), "no Gemini API key"),
        (LLMSettings(provider="openai_compatible"), "no API key"),
        (LLMSettings(provider="local_openai", base_url=None), "base URL"),
        (LLMSettings(provider="local_openai", base_url="http://x/v1"), "choose a model"),
    ],
)
def test_build_client_reasons(settings: LLMSettings, match: str) -> None:
    with pytest.raises(LLMUnavailable, match=match):
        asyncio.run(build_client(settings, KeyStore(), "processing"))


def test_build_client_gemini_needs_model() -> None:
    keys = KeyStore()
    keys.set("gemini", "AIza" + "x" * 35)
    with pytest.raises(LLMUnavailable, match="choose a Gemini model"):
        asyncio.run(build_client(LLMSettings(provider="gemini"), keys, "processing"))


def test_build_client_uses_per_task_model() -> None:
    keys = KeyStore()
    keys.set("anthropic", "sk-ant-api03-" + "x" * 30)
    s = LLMSettings(provider="anthropic", model_processing="claude-haiku-4-5")

    async def run() -> tuple[str, str]:
        a = await build_client(s, keys, "processing")
        b = await build_client(s, keys, "chat")
        models = (a.model, b.model)
        await a.aclose()
        await b.aclose()
        return models

    assert asyncio.run(run()) == ("claude-haiku-4-5", "claude-opus-5-5")
