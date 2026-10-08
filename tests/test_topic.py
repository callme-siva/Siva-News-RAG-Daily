from __future__ import annotations

import asyncio
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from newsrag.llm import LLMError
from newsrag.secrets import KeyStore
from newsrag.tools import call_tool
from newsrag.topic import (
    BriefLine,
    build_brief,
    llm_brief,
    no_news_text,
    numbers_supported,
    render_html,
    render_markdown,
    slug,
    write_brief,
)
from tests.conftest_ctx import tool_context
from tests.test_engines import FakeLLM

# Pinned to midday UTC: the fixtures are published 2-6 hours before "now", and with the real
# clock between 00:00 and 05:00 UTC they straddled midnight, which changed the ordering by day.
NOW = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
TODAY = NOW.date()


def _brief(tmp_path: Path, topic: str = "interest rates inflation", **kw: Any):  # type: ignore[no-untyped-def]
    with tool_context(tmp_path, NOW) as ctx:
        return asyncio.run(build_brief(ctx, topic, today=TODAY, **kw))


def test_rules_brief_has_all_four_sections(tmp_path: Path) -> None:
    b = _brief(tmp_path)
    assert b.found and b.engine == "rules" and b.writer == "Template"
    # "Rupee weakens after rate decision" says "rate", not "rates": no stemming, so 2 match.
    assert len(b.articles) == 2 and [a.ref for a in b.articles] == [1, 2]
    assert b.overview and all(line.refs for line in b.overview)
    assert {line.region for line in b.by_region} == {"IN", "US"}  # two regions covered it
    assert any("6.75%" in line.text for line in b.key_numbers)


def test_timeline_is_oldest_first_with_dates_from_articles(tmp_path: Path) -> None:
    b = _brief(tmp_path, "policy bill parliament energy", days=60)
    dates = [line.date for line in b.timeline]
    assert len(set(dates)) > 1 and dates == sorted(dates) and all(dates)
    by_ref = {a.ref: a.published_date for a in b.articles}
    assert all(line.date == by_ref[line.refs[0]] for line in b.timeline)


def test_single_region_has_no_by_region_section(tmp_path: Path) -> None:
    b = _brief(tmp_path, regions=["IN"])
    assert b.found and {a.region for a in b.articles} == {"IN"}
    assert b.by_region == []


def test_nothing_found_is_a_code_reply(tmp_path: Path) -> None:
    fake = FakeLLM([])
    with tool_context(tmp_path, NOW) as ctx:
        b = asyncio.run(build_brief(ctx, "zzqx unknownterm", today=TODAY, client=fake))
    assert not b.found and fake.calls == []
    assert "don't have news on 'zzqx unknownterm'" in render_markdown(b)
    assert html.escape(no_news_text(b)) in render_html(b)


def test_time_range_filters_articles(tmp_path: Path) -> None:
    old = _brief(tmp_path, "energy", days=60)
    assert [a.title for a in old.articles] == ["Old story about energy policy from last month"]
    assert not _brief(tmp_path / "b", "energy", days=3).found


def _llm_reply(**override: Any) -> str:
    reply: dict[str, Any] = {
        "overview": [{"text": "The central bank raised rates to 6.75%.", "refs": [1]}],
        "timeline": [
            {"text": "Bank raised its policy rate.", "refs": [1]},
            {"text": "Fed held rates steady.", "refs": [2]},
        ],
        "by_region": [{"text": "India saw a hike, the US a hold.", "refs": [1, 2], "region": "IN"}],
        "key_numbers": [{"text": "Policy rate now 6.75%.", "refs": [1]}],
    }
    reply.update(override)
    return json.dumps(reply)


def _llm(tmp_path: Path, reply: str | Exception):  # type: ignore[no-untyped-def]
    fake = FakeLLM([reply])
    with tool_context(tmp_path, NOW) as ctx:
        b = asyncio.run(build_brief(ctx, "interest rates inflation", today=TODAY, client=fake))
    return b, fake


def test_llm_brief_happy_path_and_dates_come_from_articles(tmp_path: Path) -> None:
    b, fake = _llm(tmp_path, _llm_reply())
    assert b.engine == "llm" and b.writer.startswith("LLM · fake")
    call = fake.calls[0]
    assert call["task"] == "chat" and "never calculate" in " ".join(call["system"].split())
    assert all(line.date for line in b.timeline)
    cited = {a.ref: a.published_date for a in b.articles}
    for line in b.timeline:
        assert line.date == min(cited[r] for r in line.refs)
    assert "Dates and links added by code" in render_markdown(b)


def test_invalid_refs_dropped_and_gaps_filled(tmp_path: Path) -> None:
    reply = _llm_reply(
        overview=[{"text": "Made up.", "refs": [99]}],
        timeline=[{"text": "No citation.", "refs": []}],
        key_numbers=[{"text": "Policy rate now 6.75%.", "refs": [1, 99]}],
    )
    b, _ = _llm(tmp_path, reply)
    assert b.overview and all(1 <= r <= 2 for line in b.overview for r in line.refs)
    assert b.key_numbers[0].refs == [1]
    assert any("Template used for: overview, timeline" in n for n in b.notes)


def test_fabricated_figures_are_rejected(tmp_path: Path) -> None:
    reply = _llm_reply(
        key_numbers=[
            {"text": "The rate rose to 7.25% overall.", "refs": [1]},  # 7.25% is not in the text
            {"text": "The policy rate is now 6.75%.", "refs": [1]},  # 6.75% is in article 1
            {"text": "The rate is 6.75% in the US too.", "refs": [2]},  # not in article 2
        ]
    )
    b, _ = _llm(tmp_path, reply)
    texts = [line.text for line in b.key_numbers]
    assert texts == ["The policy rate is now 6.75%."]


def test_numbers_supported_rules() -> None:
    evidence = "Rates rose 25 basis points to 6.75% and the budget was $1,200 million."
    assert numbers_supported("Rate is 6.75%", evidence)
    assert numbers_supported("Budget of 1,200 million", evidence)
    assert not numbers_supported("Rate is 6.85%", evidence)
    assert not numbers_supported("Budget of 1,300", evidence)
    assert numbers_supported("On the 7th, 3 officials spoke", evidence)  # small integers unchecked
    assert not numbers_supported("Inflation hit 9%", evidence)


def test_llm_failure_falls_back_to_template(tmp_path: Path) -> None:
    b, _ = _llm(tmp_path, LLMError("timeout"))
    assert b.engine == "rules" and b.found and any("LLM unavailable" in n for n in b.notes)
    secret = "sk-ant-api03-" + "Z" * 30
    keys = KeyStore()
    keys.set("anthropic", secret)
    with tool_context(tmp_path / "k", NOW) as ctx:
        base = asyncio.run(build_brief(ctx, "interest rates inflation", today=TODAY))
        out = asyncio.run(llm_brief(FakeLLM([LLMError(f"bad key {secret}")]), base, keys))
    assert secret not in " ".join(out.notes)


def test_invalid_json_falls_back(tmp_path: Path) -> None:
    b, _ = _llm(tmp_path, "not json")
    assert b.engine == "rules" and any("LLM unavailable" in n for n in b.notes)


def test_render_escapes_html_and_lists_sources(tmp_path: Path) -> None:
    b = _brief(tmp_path)
    b.overview[0].text = "<script>x()</script>"
    page = render_html(b)
    assert "<script>" not in page and "&lt;script&gt;" in page
    md = render_markdown(b)
    assert all(a.url in md for a in b.articles) and "## Sources (2 articles)" in md


def test_write_brief_files(tmp_path: Path) -> None:
    b = _brief(tmp_path / "w")
    md, page = write_brief(b, tmp_path / "briefs")
    assert md.name == f"interest-rates-inflation-{TODAY.isoformat()}.md"
    assert page.read_text().startswith("<!doctype html>")
    assert (tmp_path / "briefs" / md.with_suffix(".json").name).exists()
    assert slug("RBI: rates & inflation!") == "rbi-rates-inflation"


def test_topic_brief_tool_runs_without_llm(tmp_path: Path) -> None:
    with tool_context(tmp_path, NOW) as ctx:
        out = asyncio.run(call_tool(ctx, "topic_brief", {"topic": "interest rates", "days": 7}))
    assert out["found"] and out["engine"] == "rules" and out["articles"]
    assert "evidence" not in out["articles"][0]  # internal field is never serialised


def test_briefline_ignores_extra_keys() -> None:
    assert BriefLine.model_validate({"text": "x", "refs": [1], "extra": 1}).text == "x"


def test_topic_terms_and_coverage() -> None:
    from newsrag.topic import coverage, topic_terms

    terms = topic_terms("What did the RBI decide on interest rates?")
    assert terms == ["rbi", "decide", "interest", "rate"]
    assert coverage(terms, "RBI decides: interest rate rise") == 1.0
    assert coverage(terms, "Ola Electric rights issue") == 0.0
    assert coverage([], "anything") == 1.0  # nothing to match on: no filtering


def test_loosely_related_articles_are_left_out(tmp_path: Path) -> None:
    """Regression (seen live): articles sharing one word with the topic flooded the brief."""
    b = _brief(tmp_path, "chipmaker interest rates inflation")
    titles = [a.title for a in b.articles]
    assert titles and not any("Chipmaker" in t for t in titles)
    assert all(a.relevance >= 0.5 for a in b.articles)
    assert any("loosely related" in n and "chipmaker" in n for n in b.notes)


def test_years_are_not_key_numbers() -> None:
    from newsrag.topic import _checkable_number

    assert not _checkable_number("The firm was founded in 2024 by two people.")
    assert _checkable_number("Revenue rose 12% in 2024.")
    assert _checkable_number("The rate is 6.75.")
    assert _checkable_number("Spent $1,200 million.")


def test_same_day_stories_are_ordered_by_relevance(tmp_path: Path) -> None:
    b = _brief(tmp_path, "interest rates inflation")
    same_day = [a for a in b.articles if a.published_date == b.articles[0].published_date]
    rel = [a.relevance for a in same_day]
    assert rel == sorted(rel, reverse=True)


def test_number_pattern_does_not_swallow_punctuation() -> None:
    """Regression (seen live): '2023,' was read as a number, so the year check missed it."""
    from newsrag.topic import _NUMBER, _checkable_number

    assert _NUMBER.findall("since February 2023, the rupee fell 1.6%, to 4,096.13.") == [
        "2023",
        "1.6%",
        "4,096.13",
    ]
    assert not _checkable_number("Despite raising the rate since February 2023, the rupee fell.")


def test_notes_are_separate_paragraphs_in_markdown(tmp_path: Path) -> None:
    """Regression (seen in the UI): a note right after the source list joined its last item."""
    b = _brief(tmp_path, "chipmaker interest rates inflation")
    md = render_markdown(b)
    assert "\n\n_Note: " in md and not md.rstrip().split("\n")[-2].startswith("_Note")
