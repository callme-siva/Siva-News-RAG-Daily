"""Deterministic identities: URL normalisation and stable IDs (REQUIREMENTS DD1, DD4, DD5).

Every ID is derived from content, so re-running any step produces the same IDs and
database constraints can reject duplicates.
"""

from __future__ import annotations

import hashlib
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAMS = frozenset(
    {
        "ref",
        "ref_src",
        "fbclid",
        "gclid",
        "dclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "igshid",
        "ito",
        "cmpid",
        "ncid",
        "spm",
        "smid",
        "maca",  # DW feeds
    }
)
TRACKING_PREFIXES = ("utm_",)


def _is_tracking(param: str) -> bool:
    p = param.lower()
    return p in TRACKING_PARAMS or p.startswith(TRACKING_PREFIXES)


def normalise_url(url: str) -> str:
    """Return a canonical form of `url` so spelling variants of one article compare equal.

    Lowercases scheme and host, prefers https, drops `www.`, default ports, fragments,
    tracking parameters and trailing slashes, and sorts the remaining query parameters.
    Network work such as resolving redirects happens in the source adapters, not here.
    """
    raw = url.strip()
    if not raw:
        raise ValueError("URL is empty")
    parts = urlsplit(raw)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"Not an http(s) URL: {raw!r}")

    host = parts.hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    port = parts.port
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"

    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"

    query_pairs = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking(k)
    ]
    query = urlencode(sorted(query_pairs))

    return urlunsplit(("https", netloc, path, query, ""))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def item_id_for_url(url: str) -> str:
    """Stable item ID for a URL-identified item (news): sha256 of the normalised URL."""
    return _sha256(normalise_url(url))


def item_id_for_key(*parts: str) -> str:
    """Stable item ID from any natural key (DD6), e.g. ("gold", "2026-10-07", "source")."""
    if not parts or any(p == "" for p in parts):
        raise ValueError("Natural key parts must be non-empty")
    return _sha256("\x1f".join(parts))


def chunk_id(item_id: str, chunk_index: int) -> str:
    """Stable chunk ID shared by the chunks table, FTS5 and the vector store (DD4)."""
    if chunk_index < 0:
        raise ValueError("chunk_index must be >= 0")
    return f"{item_id}:{chunk_index}"


def content_hash(text: str) -> str:
    """Hash of cleaned text, used to detect updated articles (DD5). Whitespace-insensitive."""
    return _sha256(" ".join(text.split()))
