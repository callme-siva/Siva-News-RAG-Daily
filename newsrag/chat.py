"""Grounded chat over stored news (REQUIREMENTS FR20-FR22, R4, R5).

1. Retrieval always runs first, with the session's sticky filters (default: last 7 days).
2. Nothing found -> a code-written "I don't have news on that ..." reply. The LLM is not asked.
3. Rules mode -> the retrieved articles as a cited list (no generated prose).
4. LLM mode -> the model answers from numbered passages only. Code keeps only citations that
   point to retrieved passages and builds the Sources list itself. An answer with no valid
   citation falls back to the rules list, with a note.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from newsrag.llm import LLMClient, LLMError, parse_json_object
from newsrag.search import SearchHit, SearchResult
from newsrag.secrets import KeyStore
from newsrag.tools.context import ToolContext
from newsrag.tools.core import SearchNewsInput, search_news

log = logging.getLogger("newsrag.chat")

DEFAULT_DAYS = 7
MAX_ANSWER_TOKENS = 1500
_CITE = re.compile(r"\[(\d{1,3})\]")


class ChatFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regions: list[str] | None = None
    categories: list[str] | None = None
    days: int = Field(DEFAULT_DAYS, ge=1, le=3650)

    def date_from(self, today: date) -> date:
        return today - timedelta(days=self.days - 1)

    def describe(self) -> str:
        parts = [f"the last {self.days} days" if self.days > 1 else "today"]
        if self.regions:
            parts.append("regions " + ", ".join(self.regions))
        if self.categories:
            parts.append(", ".join(self.categories))
        return "; ".join(parts)


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int
    title: str
    source: str
    published_date: str
    url: str


class ChatAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    text: str
    sources: list[SourceRef] = Field(default_factory=list)
    engine: str  # rules | llm
    found: bool
    retrieved: int
    filters: ChatFilters
    notes: list[str] = Field(default_factory=list)


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class ChatSession:
    memory_turns: int = 6
    filters: ChatFilters = field(default_factory=ChatFilters)
    history: list[Turn] = field(default_factory=list)

    def remember(self, question: str, answer: str) -> None:
        self.history.append(Turn(question, answer))
        if self.memory_turns == 0:
            self.history.clear()
        else:
            del self.history[: -self.memory_turns]

    def clear(self) -> None:
        self.history.clear()


def no_news_text(filters: ChatFilters) -> str:
    return f"I don't have news on that in my knowledge base for {filters.describe()}."


def _source(n: int, h: SearchHit) -> SourceRef:
    return SourceRef(
        n=n, title=h.title, source=h.source, published_date=h.published_date, url=h.url
    )


def _dedupe_hits(hits: list[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    out = []
    for h in hits:
        if h.item_id not in seen:
            seen.add(h.item_id)
            out.append(h)
    return out


def rules_answer(question: str, result: SearchResult, filters: ChatFilters) -> ChatAnswer:
    if not result.hits:
        return ChatAnswer(
            question=question,
            text=no_news_text(filters),
            engine="rules",
            found=False,
            retrieved=0,
            filters=filters,
            notes=list(result.notes),
        )
    hits = _dedupe_hits(result.hits)
    lines = [f"Here is what I found for {filters.describe()} (no LLM; article summaries):", ""]
    for n, h in enumerate(hits, start=1):
        lines.append(f"[{n}] {h.title} ({h.source}, {h.published_date})")
        if h.summary and h.summary != h.title:
            lines.append(f"    {h.summary[:300]}")
    return ChatAnswer(
        question=question,
        text="\n".join(lines),
        sources=[_source(n, h) for n, h in enumerate(hits, start=1)],
        engine="rules",
        found=True,
        retrieved=len(result.hits),
        filters=filters,
        notes=list(result.notes),
    )


SYSTEM = """\
You are a personal news analyst. Today is {today}. Answer ONLY from the numbered news
passages provided. Never add facts, figures, dates or names from your own knowledge.
Cite every factual statement with the passage number in square brackets, like [2].
If the passages do not answer the question, set "answerable" to false and say briefly what
is missing. If passages disagree, say so and cite both.
Style: open with a 1-2 sentence direct answer, then short bullet points, stating dates
explicitly. Under 250 words. Do not write a sources list; it is added for you."""

USER = """\
Time range: {range}

Passages:
{passages}

{history}Question: {question}"""


def _schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "answerable": {"type": "boolean"},
            "answer": {"type": "string"},
        },
        "required": ["answerable", "answer"],
        "additionalProperties": False,
    }


class _LLMReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answerable: bool
    answer: str = Field(min_length=1)


def _passages(hits: list[SearchHit]) -> str:
    return "\n\n".join(f"[{n}] {h.text[:1800]}" for n, h in enumerate(hits, start=1))


def _history(session: ChatSession) -> str:
    if not session.history:
        return ""
    turns = "\n".join(f"Q: {t.question}\nA: {t.answer[:600]}" for t in session.history)
    return f"Earlier in this conversation (for context only, not a source):\n{turns}\n\n"


def check_citations(text: str, allowed: int) -> tuple[str, list[int], int]:
    """Remove citations outside 1..allowed. Returns (clean text, used numbers, removed)."""
    used: list[int] = []
    removed = 0

    def fix(m: re.Match[str]) -> str:
        nonlocal removed
        n = int(m.group(1))
        if 1 <= n <= allowed:
            if n not in used:
                used.append(n)
            return m.group(0)
        removed += 1
        return ""

    clean = _CITE.sub(fix, text)
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
    return clean, used, removed


async def llm_answer(
    client: LLMClient,
    question: str,
    result: SearchResult,
    session: ChatSession,
    keys: KeyStore,
    today: date,
) -> ChatAnswer:
    filters = session.filters
    if not result.hits:
        return rules_answer(question, result, filters)
    hits = result.hits
    user = USER.format(
        range=f"{filters.date_from(today).isoformat()} to {today.isoformat()} "
        f"({filters.describe()})",
        passages=_passages(hits),
        history=_history(session),
        question=question,
    )
    try:
        text = await client.complete(
            system=SYSTEM.format(today=today.isoformat()),
            user=user,
            schema=_schema(),
            max_tokens=MAX_ANSWER_TOKENS,
            task="chat",
        )
        reply = _LLMReply.model_validate(parse_json_object(text))
    except (LLMError, ValueError, ValidationError) as exc:
        fallback = rules_answer(question, result, filters)
        fallback.notes.append(
            f"LLM unavailable, showing articles instead ({keys.redact(str(exc))[:150]})"
        )
        return fallback

    if not reply.answerable:
        return ChatAnswer(
            question=question,
            text=f"{no_news_text(filters)} {reply.answer}".strip(),
            engine="llm",
            found=False,
            retrieved=len(hits),
            filters=filters,
            notes=list(result.notes),
        )
    clean, used, removed = check_citations(reply.answer, len(hits))
    notes = list(result.notes)
    if removed:
        notes.append(f"removed {removed} citation(s) that did not match a retrieved article")
    if not used:
        fallback = rules_answer(question, result, filters)
        fallback.notes.append("LLM answer had no valid citations, showing articles instead")
        return fallback
    return ChatAnswer(
        question=question,
        text=clean,
        sources=[_source(n, hits[n - 1]) for n in sorted(used)],
        engine="llm",
        found=True,
        retrieved=len(hits),
        filters=filters,
        notes=notes,
    )


async def ask(
    ctx: ToolContext,
    session: ChatSession,
    question: str,
    client: LLMClient | None,
    today: date,
) -> ChatAnswer:
    """Retrieve with the session filters, answer (LLM or rules), remember the turn."""
    f = session.filters
    result = search_news(
        ctx,
        SearchNewsInput(
            query=question,
            regions=f.regions,
            categories=f.categories,
            date_from=f.date_from(today),
            date_to=today,
        ),
    )
    if client is None:
        answer = rules_answer(question, result, f)
    else:
        answer = await llm_answer(client, question, result, session, ctx.keys, today)
    session.remember(question, answer.text)
    return answer


def format_answer(a: ChatAnswer) -> str:
    out = [a.text]
    if a.sources and a.engine == "llm":
        out += ["", "Sources:"]
        out += [f"[{s.n}] {s.title} - {s.source}, {s.published_date} - {s.url}" for s in a.sources]
    elif a.sources:
        out += ["", "Links:"]
        out += [f"[{s.n}] {s.url}" for s in a.sources]
    label = "LLM (GENERATED from cited articles)" if a.engine == "llm" else "Rules (article text)"
    out += ["", f"({label}; {a.retrieved} passages retrieved)"]
    out += [f"Note: {n}" for n in a.notes]
    return "\n".join(out)
