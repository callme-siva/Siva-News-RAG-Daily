"""In-memory API key store and redaction (REQUIREMENTS R3).

Keys come from environment variables or the UI password field, live only in this
process's memory, and are scrubbed from any text that is logged or shown as an error.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping

REDACTED = "[REDACTED]"

ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai_compatible": ("OPENAI_API_KEY",),
    "gnews": ("GNEWS_API_KEY",),
    "newsdata": ("NEWSDATA_API_KEY",),
    "thenewsapi": ("THENEWSAPI_KEY",),
    "currents": ("CURRENTS_API_KEY",),
    "newsapi": ("NEWSAPI_KEY",),
    "smtp": ("NEWSRAG_SMTP_PASSWORD",),
}

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)\b(api[_-]?key|apikey|token|x-api-key)=([^&\s]{8,})"),
]
_MIN_SECRET_LEN = 8


def looks_like_secret(text: str) -> bool:
    """True if `text` contains something shaped like a known API key format."""
    return any(p.search(text) for p in _SECRET_PATTERNS)


class KeyStore:
    """Holds API keys in memory only. Never serialisable, never printed."""

    def __init__(self) -> None:
        self._keys: dict[str, str] = {}

    def set(self, name: str, key: str | None) -> None:
        key = (key or "").strip()
        if key:
            self._keys[name] = key
        else:
            self._keys.pop(name, None)

    def get(self, name: str) -> str | None:
        return self._keys.get(name)

    def has(self, name: str) -> bool:
        return name in self._keys

    def names(self) -> list[str]:
        return sorted(self._keys)

    def values(self) -> list[str]:
        return [v for v in self._keys.values() if len(v) >= _MIN_SECRET_LEN]

    def clear(self) -> None:
        self._keys.clear()

    def load_from_env(self, environ: Mapping[str, str] | None = None) -> list[str]:
        """Load any keys present in the environment. Returns the provider names found."""
        env = os.environ if environ is None else environ
        found = []
        for name, variables in ENV_VARS.items():
            for var in variables:
                if env.get(var, "").strip():
                    self.set(name, env[var])
                    found.append(name)
                    break
        return found

    def redact(self, text: str) -> str:
        return redact(text, self.values())

    def __repr__(self) -> str:
        return f"KeyStore(names={self.names()})"

    __str__ = __repr__

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("KeyStore cannot be pickled or serialised")


def redact(text: str, known_secrets: list[str] | None = None) -> str:
    """Replace known key values and key-shaped strings in `text` with [REDACTED]."""
    out = text
    for secret in sorted(known_secrets or [], key=len, reverse=True):
        if len(secret) >= _MIN_SECRET_LEN:
            out = out.replace(secret, REDACTED)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            out = pattern.sub(lambda m: f"{m.group(1)}={REDACTED}", out)
        else:
            out = pattern.sub(REDACTED, out)
    return out


class RedactingFilter(logging.Filter):
    """Logging filter that scrubs secrets from messages, args and exception text."""

    def __init__(self, store: KeyStore) -> None:
        super().__init__()
        self._store = store

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if record.exc_info:
            formatter = logging.Formatter()
            message = f"{message}\n{formatter.formatException(record.exc_info)}"
            record.exc_info = None
            record.exc_text = None
        record.msg = self._store.redact(message)
        record.args = None
        return True


# One store per process. The UI and CLI share it.
KEYS = KeyStore()
