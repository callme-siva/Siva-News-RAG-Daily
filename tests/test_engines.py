from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from newsrag.config import load_config
from newsrag.engines import RuleEngine, process_all, select_engine
from newsrag.engines.llm import LLMEngine, output_schema
from newsrag.engines.rules import extract_entities, extractive_summary, key_facts
from newsrag.engines.select import EngineUnavailable
from newsrag.llm import LLMError, LLMUnavailable, Task
from newsrag.models import EngineName
from newsrag.secrets import KeyStore
from newsrag.settings import LLMSettings
from tests.helpers import item

CFG = load_config()
BODY = (
    "The Fixture Reserve Bank raised its policy rate by 25 basis points to 6.75% on Wednesday. "
    "Governor Asha Mehra said inflation remained above target. "
    "Markets in India fell 1.2% after the announcement. "
    "Analysts at Fixture Capital Ltd expect one more increase this year."
)
GOOD = {
    "relevant": True,
    "category": "Finance",
    "summary": "The fixture central bank raised its policy rate to 6.75%.",
    "key_facts": ["Policy rate raised by 25 basis points to 6.75%"],
    "entities": {
        "companies": ["Fixture Capital Ltd"],
        "people": ["Asha Mehra"],
        "organisations": ["Fixture Reserve Bank"],
        "countries": ["India"],
    },
}


class FakeLLM:
    provider = "fake"
    model = "fake-1"

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        self.checked = False
        self.closed = False

    async def complete(
        self, *, system: str, user: str, schema: dict[str, Any] | None, max_tokens: int, task: Task
    ) -> str:
        self.calls.append({"system": system, "user": user, "schema": schema, "task": task})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def check(self) -> None:
        self.checked = True

    async def aclose(self) -> None:
        self.closed = True


def _engine(replies: list[str | Exception], keys: KeyStore | None = None) -> LLMEngine:
    keys = keys or KeyStore()
    return LLMEngine(FakeLLM(replies), CFG, RuleEngine(CFG), keys)


# ---------- RuleEngine ----------


def test_extractive_summary_uses_only_article_sentences() -> None:
    summary = extractive_summary("Headline", BODY)
    assert summary.startswith("The Fixture Reserve Bank raised")
    assert len(summary) <= 400
    assert all(part in BODY for part in summary.split(". ") if part)
    assert extractive_summary("Only a headline", "") == "Only a headline"


def test_key_facts_are_numeric_sentences_not_in_summary() -> None:
    summary = extractive_summary("H", BODY)
    facts = key_facts(BODY, summary)
    assert facts and all(f in BODY and f not in summary for f in facts)
    assert any("1.2%" in f for f in facts)


def test_entity_rules() -> None:
    ents = extract_entities(BODY)
    assert "Fixture Reserve Bank" in ents.organisations
    assert "Fixture Capital Ltd" in ents.companies
    assert "India" in ents.countries
    assert "Asha Mehra" in ents.people
    assert "Wednesday" not in ents.flat()


def test_rule_engine_process() -> None:
    p = asyncio.run(RuleEngine(CFG).process(item("Bank raises interest rates", body=BODY)))
    assert p.engine == EngineName.RULES and p.relevant and p.fallback_reason is None
    off = asyncio.run(RuleEngine(CFG).process(item("Celebrity wedding photos", body="Gowns.")))
    assert not off.relevant


# ---------- LLMEngine ----------


def test_llm_engine_happy_path() -> None:
    engine = _engine([json.dumps(GOOD)])
    p = asyncio.run(engine.process(item("Bank raises interest rates", body=BODY)))
    assert p.engine == EngineName.LLM and p.fallback_reason is None
    assert p.summary == GOOD["summary"] and p.entities.people == ["Asha Mehra"]
    fake = engine._client
    assert isinstance(fake, FakeLLM)
    call = fake.calls[0]
    assert call["task"] == "processing"
    assert call["schema"] == output_schema(["Technology", "Finance", "Politics"])
    assert "Bank raises interest rates" in call["user"]


def test_llm_reply_in_code_fence_is_accepted() -> None:
    engine = _engine(["```json\n" + json.dumps(GOOD) + "\n```"])
    assert asyncio.run(engine.process(item("x y z", body=BODY))).engine == EngineName.LLM


def test_invalid_json_retries_once_then_succeeds() -> None:
    engine = _engine(["not json at all", json.dumps(GOOD)])
    p = asyncio.run(engine.process(item("x y z", body=BODY)))
    assert p.engine == EngineName.LLM
    fake = engine._client
    assert isinstance(fake, FakeLLM)
    assert len(fake.calls) == 2 and "previous reply was not valid" in fake.calls[1]["user"]


def test_invalid_twice_falls_back_to_rules_for_that_item() -> None:
    bad = json.dumps({**GOOD, "summary": ""})
    p = asyncio.run(_engine([bad, bad]).process(item("Bank raises rates", body=BODY)))
    assert p.engine == EngineName.RULES
    assert p.fallback_reason and "summary" in p.fallback_reason


def test_llm_error_falls_back_and_reason_is_redacted() -> None:
    keys = KeyStore()
    keys.set("anthropic", "sk-ant-api03-" + "Z" * 30)
    secret = keys.get("anthropic")
    assert secret
    engine = _engine([LLMError(f"auth failed for {secret}")], keys)
    p = asyncio.run(engine.process(item("Bank raises rates", body=BODY)))
    assert p.engine == EngineName.RULES
    assert p.fallback_reason and secret not in p.fallback_reason


def test_unknown_category_from_llm_is_ignored() -> None:
    engine = _engine([json.dumps({**GOOD, "category": "Sports"})])
    p = asyncio.run(engine.process(item("x y z", body=BODY, category="Finance")))
    assert p.category == "Finance"


def test_more_than_five_facts_are_trimmed() -> None:
    engine = _engine([json.dumps({**GOOD, "key_facts": [f"fact {i}" for i in range(9)]})])
    assert len(asyncio.run(engine.process(item("x y z", body=BODY))).key_facts) == 5


def test_extra_entity_keys_from_llm_are_ignored() -> None:
    reply = {**GOOD, "entities": {**GOOD["entities"], "places": ["Mumbai"]}, "extra": 1}  # type: ignore[dict-item]
    p = asyncio.run(_engine([json.dumps(reply)]).process(item("x y z", body=BODY)))
    assert p.engine == EngineName.LLM


# ---------- process_all ----------


def test_process_all_progress_and_stop() -> None:
    items = [item(f"Bank rates story number {i}", body=BODY) for i in range(5)]
    seen: list[tuple[int, int]] = []
    out = asyncio.run(
        process_all(
            RuleEngine(CFG), items, concurrency=2, on_progress=lambda d, t: seen.append((d, t))
        )
    )
    assert len(out) == 5 and seen[-1] == (5, 5)
    assert [p.item_id for p in out] == [i.item_id for i in items]

    stop = asyncio.Event()
    stop.set()
    assert asyncio.run(process_all(RuleEngine(CFG), items, stop=stop)) == []


# ---------- select_engine ----------


def _patch_build(monkeypatch: pytest.MonkeyPatch, behaviour: dict[str, Any]) -> list[str]:
    tried: list[str] = []

    async def fake_build(s: LLMSettings, keys: KeyStore, task: Task, provider: str | None = None):  # type: ignore[no-untyped-def]
        name = provider or s.provider
        tried.append(name)
        outcome = behaviour[name]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("newsrag.engines.select.build_client", fake_build)
    return tried


def test_select_rules_mode_never_touches_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    tried = _patch_build(monkeypatch, {})
    choice = asyncio.run(select_engine(LLMSettings(mode="rules"), CFG, KeyStore()))
    assert choice.engine.name == EngineName.RULES and tried == []


def test_select_auto_with_working_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLLM([])
    _patch_build(monkeypatch, {"ollama": fake})
    choice = asyncio.run(select_engine(LLMSettings(mode="auto"), CFG, KeyStore()))
    assert choice.engine.name == EngineName.LLM and fake.checked
    asyncio.run(choice.aclose())
    assert fake.closed


def test_select_auto_falls_back_to_anthropic_then_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = KeyStore()
    keys.set("anthropic", "sk-ant-api03-" + "Q" * 30)
    claude = FakeLLM([])
    tried = _patch_build(
        monkeypatch, {"ollama": LLMUnavailable("not running"), "anthropic": claude}
    )
    choice = asyncio.run(select_engine(LLMSettings(mode="auto"), CFG, keys))
    assert tried == ["ollama", "anthropic"] and choice.engine.name == EngineName.LLM
    assert "not running" in choice.reason

    tried = _patch_build(
        monkeypatch,
        {"ollama": LLMUnavailable("not running"), "anthropic": LLMUnavailable("no key")},
    )
    choice = asyncio.run(select_engine(LLMSettings(mode="auto"), CFG, keys))
    assert choice.engine.name == EngineName.RULES and "not running" in choice.reason


def test_select_llm_mode_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_build(monkeypatch, {"ollama": LLMUnavailable("not running")})
    with pytest.raises(EngineUnavailable, match="not running"):
        asyncio.run(select_engine(LLMSettings(mode="llm"), CFG, KeyStore()))


def test_select_check_failure_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class Failing(FakeLLM):
        async def check(self) -> None:
            raise LLMUnavailable("model missing")

    bad = Failing([])
    _patch_build(monkeypatch, {"ollama": bad})
    choice = asyncio.run(select_engine(LLMSettings(mode="auto"), CFG, KeyStore()))
    assert choice.engine.name == EngineName.RULES and bad.closed


def test_split_sentences_keeps_abbreviations() -> None:
    from newsrag.engines.rules import split_sentences

    text = "The U.S. Supreme Court will hear the case. Mr. Smith said yes. Done."
    assert split_sentences(text) == [
        "The U.S. Supreme Court will hear the case.",
        "Mr. Smith said yes.",
        "Done.",
    ]
