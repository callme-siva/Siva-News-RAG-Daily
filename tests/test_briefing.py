from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest

from newsrag.briefing import (
    Digest,
    EmailNotConfigured,
    llm_digest,
    render_html,
    render_markdown,
    rule_digest,
    select_articles,
    send_email,
    write_digest,
)
from newsrag.secrets import KeyStore
from newsrag.settings import BriefingSettings
from tests.conftest_ctx import tool_context
from tests.test_engines import FakeLLM

NOW = datetime.now(UTC)
REGIONS = ["US", "EU", "IN"]


def _base(tmp_path: Path) -> Digest:
    with tool_context(tmp_path, NOW) as ctx:
        arts, start = select_articles(ctx.store, ctx.cfg, now=NOW, per_group=5)
        return rule_digest(arts, ctx.cfg, REGIONS, now=NOW, start=start)


def test_selection_window_order_and_numbering(tmp_path: Path) -> None:
    d = _base(tmp_path)
    titles = [a.title for a in d.articles]
    assert "Old story about energy policy from last month" not in titles  # outside 24h
    assert [a.ref for a in d.articles] == list(range(1, len(d.articles) + 1))
    regions = [a.region for a in d.articles]
    assert regions == sorted(regions, key=REGIONS.index)  # US, EU, IN order from config


def test_rule_digest_structure(tmp_path: Path) -> None:
    d = _base(tmp_path)
    assert d.engine == "rules" and d.writer == "Template"
    assert len(d.top) == 3 and len(d.sections) == 9
    in_fin = next(s for s in d.sections if (s.region, s.category) == ("IN", "Finance"))
    assert len(in_fin.lines) == 2
    empty = next(s for s in d.sections if (s.region, s.category) == ("IN", "Politics"))
    assert empty.lines == []
    md = render_markdown(d)
    assert "No new stories today." in md and "Links added by code" in md
    assert all(a.url in md for a in d.articles)


def test_html_is_escaped(tmp_path: Path) -> None:
    d = _base(tmp_path)
    evil = d.articles[0].model_copy(update={"title": "<script>alert(1)</script>"})
    d = d.model_copy(update={"articles": [evil, *d.articles[1:]]})
    d.sections[0].lines[0].headline = "<b>x</b>"
    page = render_html(d)
    assert "<script>" not in page and "&lt;b&gt;x&lt;/b&gt;" in page


def _llm_reply(d: Digest, **override: Any) -> str:
    sections = []
    for s in d.sections:
        refs = [a.ref for a in d.articles if a.region == s.region and a.category == s.category]
        if refs:
            sections.append(
                {
                    "region": s.region,
                    "category": s.category,
                    "lines": [{"headline": f"LLM {s.category}", "why": "Because.", "refs": refs}],
                }
            )
    reply: dict[str, Any] = {
        "top": [{"headline": "LLM top", "why": "Matters.", "refs": [1]}],
        "sections": sections,
    }
    reply.update(override)
    return json.dumps(reply)


def test_llm_digest_happy_path(tmp_path: Path) -> None:
    base = _base(tmp_path)
    fake = FakeLLM([_llm_reply(base)])
    d = asyncio.run(llm_digest(fake, base, BriefingSettings(), KeyStore()))
    assert d.engine == "llm" and d.writer.startswith("LLM · fake")
    assert d.top[0].headline == "LLM top" and d.notes == []
    assert fake.calls[0]["task"] == "digest" and "[1]" in fake.calls[0]["user"]
    assert "Written by LLM" in render_markdown(d)


def test_llm_invalid_refs_are_dropped_and_gaps_filled(tmp_path: Path) -> None:
    base = _base(tmp_path)
    in_fin = [a.ref for a in base.articles if (a.region, a.category) == ("IN", "Finance")]
    us_tech = [a.ref for a in base.articles if (a.region, a.category) == ("US", "Technology")]
    reply = _llm_reply(
        base,
        top=[{"headline": "Made up", "why": "x", "refs": [999]}],
        sections=[
            # cites an article from another section: not allowed
            {
                "region": "IN",
                "category": "Finance",
                "lines": [
                    {"headline": "Wrong section", "why": "x", "refs": us_tech},
                    {"headline": "Right section", "why": "y", "refs": [*in_fin, 999]},
                ],
            },
        ],
    )
    d = asyncio.run(llm_digest(FakeLLM([reply]), base, BriefingSettings(), KeyStore()))
    sec = next(s for s in d.sections if (s.region, s.category) == ("IN", "Finance"))
    assert [line.headline for line in sec.lines] == ["Right section"]
    assert sec.lines[0].refs == in_fin
    assert d.top == base.top  # invalid top replaced by the template
    assert any("Template used for" in n for n in d.notes)


def test_llm_failure_falls_back_to_template(tmp_path: Path) -> None:
    from newsrag.llm import LLMError

    base = _base(tmp_path)
    d = asyncio.run(
        llm_digest(FakeLLM([LLMError("timeout")]), base, BriefingSettings(), KeyStore())
    )
    assert d.engine == "rules" and any("LLM unavailable" in n for n in d.notes)


def test_write_digest_files(tmp_path: Path) -> None:
    d = _base(tmp_path)
    md, page = write_digest(d, tmp_path / "digests")
    assert md.read_text().startswith("# Today's Briefing")
    assert page.read_text().startswith("<!doctype html>")
    assert (tmp_path / "digests" / f"{d.date}.json").exists()


class FakeSMTP:
    sent: ClassVar[list[Any]] = []

    def __init__(self, host: str, port: int, timeout: int) -> None:
        self.host, self.port = host, port

    def __enter__(self) -> FakeSMTP:
        return self

    def __exit__(self, *a: object) -> None: ...

    def starttls(self) -> None: ...

    def login(self, user: str, password: str) -> None:
        self.creds = (user, password)

    def send_message(self, msg: Any) -> None:
        FakeSMTP.sent.append((self, msg))


def test_email_needs_config_and_password(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _base(tmp_path)
    with pytest.raises(EmailNotConfigured, match="NEWSRAG_SMTP_HOST"):
        send_email(d, KeyStore(), environ={})
    monkeypatch.setattr("newsrag.briefing.smtplib.SMTP", FakeSMTP)
    keys = KeyStore()
    keys.set("smtp", "app-password-123")
    env = {
        "NEWSRAG_SMTP_HOST": "smtp.example.com",
        "NEWSRAG_SMTP_USER": "me@example.com",
        "NEWSRAG_SMTP_TO": "me@example.com",
    }
    assert send_email(d, keys, environ=env) == "me@example.com"
    smtp, msg = FakeSMTP.sent[-1]
    assert smtp.port == 587 and smtp.creds == ("me@example.com", "app-password-123")
    assert msg["Subject"].startswith("Today's Briefing")
    assert "app-password-123" not in msg.as_string()


def test_top_of_day_spreads_regions_and_truncates_cleanly(tmp_path: Path) -> None:
    d = _base(tmp_path)
    regions = [d.article(line.refs[0]).region for line in d.top]  # type: ignore[union-attr]
    assert len(set(regions)) == 3
    from newsrag.briefing import WHY_MAX_CHARS, _first_sentence

    long = "word " * 100 + "end."
    cut = _first_sentence(long)
    assert cut.endswith("…") and len(cut) <= WHY_MAX_CHARS + 1 and " wor…" not in cut
