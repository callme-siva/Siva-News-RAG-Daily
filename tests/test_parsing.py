from __future__ import annotations

from datetime import UTC, datetime

import pytest

from newsrag.sources.parsing import clean_text, parse_date


def test_clean_text() -> None:
    raw = (
        "<p>Rates <b>held</b> &amp; steady.</p><script>x()</script><style>p{}</style> [+512 chars]"
    )
    assert clean_text(raw) == "Rates held & steady."
    assert clean_text(None) == ""


@pytest.mark.parametrize(
    ("value", "tz", "expected"),
    [
        ("Wed, 07 Oct 2026 08:00:00 GMT", None, datetime(2026, 10, 7, 8, 0, tzinfo=UTC)),
        # RBI style: no timezone, IST applied -> 07:20 UTC
        ("Wed, 07 Oct 2026 12:50:00", "Asia/Kolkata", datetime(2026, 10, 7, 7, 20, tzinfo=UTC)),
        # SEBI style
        ("05 Oct, 2026 +0530", None, datetime(2026, 10, 4, 18, 30, tzinfo=UTC)),
        # GDELT style
        ("20261006T083000Z", None, datetime(2026, 10, 6, 8, 30, tzinfo=UTC)),
        # ISO with Z
        ("2026-10-06T08:30:00Z", None, datetime(2026, 10, 6, 8, 30, tzinfo=UTC)),
        # NewsData style, naive, UTC by default
        ("2026-10-06 08:30:00", None, datetime(2026, 10, 6, 8, 30, tzinfo=UTC)),
    ],
)
def test_parse_date_formats(value: str, tz: str | None, expected: datetime) -> None:
    assert parse_date(value, tz) == expected


@pytest.mark.parametrize("value", [None, "", "   ", "yesterday-ish", "32 Foo 2026"])
def test_parse_date_unparseable(value: str | None) -> None:
    assert parse_date(value) is None
