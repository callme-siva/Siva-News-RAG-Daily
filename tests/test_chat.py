from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from newsrag.chat import ChatFilters, ChatSession, ask, check_citations, format_answer
from newsrag.llm import LLMError
from tests.conftest_ctx import tool_context
from tests.test_engines import FakeLLM

NOW = datetime.now(UTC)
TODAY = NOW.date()


def test_check_citations() -> None:
    text, used, removed = check_citations("Rates rose [1]. Rupee fell [3] [9].", 3)
    assert used == [1, 3] and removed == 1 and "[9]" not in text


def test_rules_answer_is_cited_list(tmp_path: Path) -> None:
    with tool_context(tmp_path, NOW) as ctx:
        a = asyncio.run(ask(ctx, ChatSession(), "interest rates inflation", None, TODAY))
    assert a.engine == "rules" and a.found and a.sources
    assert a.text.startswith("Here is what I found")
    assert all(f"[{s.n}]" in a.text for s in a.sources)
    assert "Rules (article text)" in format_answer(a)


def test_nothing_found_is_said_by_code_and_llm_not_called(tmp_path: Path) -> None:
    fake = FakeLLM([])
    with tool_context(tmp_path, NOW) as ctx:
        s = ChatSession(filters=ChatFilters(regions=["IN"], days=1))
        a = asyncio.run(ask(ctx, s, "zzqx unknownterm", fake, TODAY))
    assert not a.found and a.text.startswith("I don't have news on that")
    assert "regions IN" in a.text and fake.calls == []


def test_sticky_filters_apply(tmp_path: Path) -> None:
    with tool_context(tmp_path, NOW) as ctx:
        s = ChatSession(filters=ChatFilters(regions=["US"]))
        a = asyncio.run(ask(ctx, s, "interest rates", None, TODAY))
    assert a.sources and all("Federal Reserve" in x.title or "FXCHIP" in x.title for x in a.sources)


def test_llm_answer_grounded_and_sources_built_by_code(tmp_path: Path) -> None:
    reply = json.dumps({"answerable": True, "answer": "The bank raised rates to 6.75% [1]."})
    fake = FakeLLM([reply])
    with tool_context(tmp_path, NOW) as ctx:
        a = asyncio.run(ask(ctx, ChatSession(), "what did the central bank do", fake, TODAY))
    assert a.engine == "llm" and a.found and [s.n for s in a.sources] == [1]
    out = format_answer(a)
    assert "Sources:" in out and a.sources[0].url in out
    call = fake.calls[0]
    assert call["task"] == "chat" and "[1]" in call["user"]
    assert "Never add facts" in call["system"]


def test_llm_invalid_citations_removed_or_fallback(tmp_path: Path) -> None:
    bad_only = json.dumps({"answerable": True, "answer": "Something happened [42]."})
    mixed = json.dumps({"answerable": True, "answer": "Rates rose [1]. Also [42]."})
    with tool_context(tmp_path, NOW) as ctx:
        a = asyncio.run(ask(ctx, ChatSession(), "interest rates", FakeLLM([bad_only]), TODAY))
        b = asyncio.run(ask(ctx, ChatSession(), "interest rates", FakeLLM([mixed]), TODAY))
    assert a.engine == "rules" and any("no valid citations" in n for n in a.notes)
    assert b.engine == "llm" and "[42]" not in b.text
    assert any("removed 1 citation" in n for n in b.notes)


def test_llm_says_unanswerable(tmp_path: Path) -> None:
    reply = json.dumps({"answerable": False, "answer": "The articles do not cover sport."})
    with tool_context(tmp_path, NOW) as ctx:
        a = asyncio.run(ask(ctx, ChatSession(), "interest rates", FakeLLM([reply]), TODAY))
    assert not a.found and a.text.startswith("I don't have news on that") and a.sources == []


def test_llm_error_falls_back_to_rules(tmp_path: Path) -> None:
    with tool_context(tmp_path, NOW) as ctx:
        a = asyncio.run(
            ask(ctx, ChatSession(), "interest rates", FakeLLM([LLMError("down")]), TODAY)
        )
    assert a.engine == "rules" and a.found and any("LLM unavailable" in n for n in a.notes)


def test_history_is_sent_and_trimmed(tmp_path: Path) -> None:
    replies = [json.dumps({"answerable": True, "answer": f"Answer {i} [1]."}) for i in range(4)]
    fake = FakeLLM(replies)  # type: ignore[arg-type]
    session = ChatSession(memory_turns=2)
    with tool_context(tmp_path, NOW) as ctx:
        for i in range(4):
            asyncio.run(ask(ctx, session, f"interest rates question {i}", fake, TODAY))
    assert len(session.history) == 2
    last_prompt = fake.calls[-1]["user"]
    assert "question 2" in last_prompt and "question 0" not in last_prompt
    assert "not a source" in last_prompt
