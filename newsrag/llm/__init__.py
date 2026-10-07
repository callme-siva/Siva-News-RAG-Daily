"""Build an LLM client from settings and the in-memory KeyStore."""

from __future__ import annotations

from newsrag.llm.anthropic_client import AnthropicClient
from newsrag.llm.base import LLMClient, LLMError, LLMUnavailable, Task, parse_json_object
from newsrag.llm.http_clients import GeminiClient, OllamaClient, OpenAICompatibleClient
from newsrag.secrets import KeyStore
from newsrag.settings import LLMSettings

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMUnavailable",
    "Task",
    "build_client",
    "model_for",
    "parse_json_object",
]

KEY_NAMES = {
    "anthropic": "anthropic",
    "gemini": "gemini",
    "openai_compatible": "openai_compatible",
    "local_openai": "openai_compatible",
}
LOCAL_PROVIDERS = frozenset({"ollama", "local_openai"})


def model_for(s: LLMSettings, task: Task) -> str | None:
    return {"processing": s.model_processing, "chat": s.model_chat, "digest": s.model_digest}[task]


async def build_client(
    s: LLMSettings, keys: KeyStore, task: Task, provider: str | None = None
) -> LLMClient:
    """Create the client for `provider` (default: the configured one). Raises LLMUnavailable
    with a user-facing reason when it cannot be built. Does not make a network call except
    for Ollama with no model set, where the first installed model is chosen."""
    provider = provider or s.provider
    model = model_for(s, task)
    timeout = float(s.timeout_s)

    if provider == "anthropic":
        key = keys.get("anthropic")
        if not key:
            raise LLMUnavailable("no Anthropic API key")
        return AnthropicClient(key, model, timeout)

    if provider == "gemini":
        key = keys.get("gemini")
        if not key:
            raise LLMUnavailable("no Gemini API key")
        if not model:
            raise LLMUnavailable("choose a Gemini model in Settings")
        return GeminiClient(model=model, api_key=key, timeout_s=timeout, temperature=s.temperature)

    if provider in ("openai_compatible", "local_openai"):
        key = keys.get(KEY_NAMES[provider])
        if provider == "openai_compatible" and not key:
            raise LLMUnavailable("no API key for the OpenAI-compatible provider")
        if not s.base_url:
            raise LLMUnavailable("set the server base URL in Settings")
        if not model:
            raise LLMUnavailable("choose a model in Settings")
        return OpenAICompatibleClient(
            model=model,
            base_url=s.base_url,
            api_key=key,
            timeout_s=timeout,
            temperature=s.temperature,
        )

    if provider == "ollama":
        base = s.base_url or "http://localhost:11434"
        if not model:
            probe = OllamaClient(model="", base_url=base, timeout_s=5)
            try:
                installed = await probe.list_models()
            finally:
                await probe.aclose()
            if not installed:
                raise LLMUnavailable("Ollama has no models installed (try: ollama pull llama3.2)")
            model = installed[0]
        return OllamaClient(
            model=model,
            base_url=base,
            timeout_s=timeout,
            temperature=s.temperature,
            context_length=s.context_length,
        )

    raise LLMUnavailable(f"unknown provider {provider!r}")
