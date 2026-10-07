"""Text cleaning and date parsing shared by the adapters.

Real feeds are messy: RBI dates have no timezone, SEBI uses "05 Oct, 2026 +0530",
GDELT uses "20261006T083000Z", and PIB has no dates at all.
"""

from __future__ import annotations

import calendar
import html
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAGS = re.compile(r"<[^>]+>")
_TRUNCATION = re.compile(r"\[\+\d+ chars\]")
_SPACES = re.compile(r"\s+")

_FORMATS = (
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S",
    "%d %b, %Y %z",
    "%d %b %Y %H:%M:%S %z",
    "%d %b %Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y%m%dT%H%M%SZ",
)


def clean_text(raw: str | None) -> str:
    """Strip HTML, decode entities, remove API truncation markers, collapse whitespace."""
    if not raw:
        return ""
    text = _SCRIPT_STYLE.sub(" ", raw)
    text = _TAGS.sub(" ", text)
    text = html.unescape(text)
    text = _TRUNCATION.sub("", text)
    return _SPACES.sub(" ", text).strip()


def parse_date(value: str | None, default_tz: str | None = None) -> datetime | None:
    """Parse a date string in any format seen in our sources. Naive results get
    `default_tz` (IANA name) or UTC. Returns None if nothing matches."""
    if not value or not value.strip():
        return None
    s = value.strip()
    dt: datetime | None = None
    try:
        dt = parsedate_to_datetime(s)
    except (TypeError, ValueError, IndexError):
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            dt = None
    if dt is None:
        for fmt in _FORMATS:
            try:
                dt = datetime.strptime(s, fmt)
            except ValueError:
                continue
            if fmt.endswith("Z"):
                dt = dt.replace(tzinfo=UTC)
            break
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(default_tz) if default_tz else UTC)
    return dt.astimezone(UTC)


def from_struct_time(value: time.struct_time | None) -> datetime | None:
    """feedparser's *_parsed fields are UTC struct_time values."""
    if value is None:
        return None
    return datetime.fromtimestamp(calendar.timegm(value), tz=UTC)
