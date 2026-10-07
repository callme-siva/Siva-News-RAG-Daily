"""Topic brief: a structured summary of everything stored about a topic over a time range.

Code does the retrieval, ordering, dates, numbers and links. The writer (rules or LLM) only
produces short lines tied to article numbers. Code then:
  - drops lines citing articles that do not exist,
  - sets every timeline date from the cited articles (the model never writes a date),
  - drops any line whose figures do not appear in the articles it cites,
  - fills empty sections from the rules version, with a note,
  - and adds all links itself.
"""

from __future__ import annotations

import html
import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from newsrag.briefing import REGION_NAMES
from newsrag.engines.rules import split_sentences
from newsrag.llm import LLMClient, LLMError, parse_json_object
from newsrag.search import SearchFilters, SearchHit, search
from newsrag.secrets import KeyStore
from newsrag.settings import RetrievalSettings
from newsrag.store import STOPWORDS, Store

if TYPE_CHECKING:  # tools.core imports this module, so a runtime import would be circular
    from newsrag.tools.context import ToolContext

log = logging.getLogger("newsrag.topic")

MIN_TERM_COVERAGE = 0.5  # an article must mention at least half of the topic's words
MAX_ARTICLES = 25
MAX_TIMELINE = 12
MAX_NUMBERS = 8
SENTENCE_MAX = 260

_NUMBER = re.compile(r"\d(?:[\d,]*\d)?(?:\.\d+)?%?")  # never ends in a comma or full stop


class BriefArticle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: int
    item_id: str
    title: str
    url: str
    source: str
    region: str
    category: str
    published_date: str
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    outlets: int = 1
    relevance: float = 1.0
    evidence: str = Field(default="", exclude=True)


class BriefLine(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1)
    refs: list[int] = Field(default_factory=list)
    date: str | None = None
    region: str | None = None


class TopicBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str
    date_from: str
    date_to: str
    regions: list[str] | None = None
    found: bool
    engine: str  # rules | llm
    writer: str
    overview: list[BriefLine] = Field(default_factory=list)
    timeline: list[BriefLine] = Field(default_factory=list)
    by_region: list[BriefLine] = Field(default_factory=list)
    key_numbers: list[BriefLine] = Field(default_factory=list)
    articles: list[BriefArticle] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def article(self, ref: int) -> BriefArticle | None:
        return next((a for a in self.articles if a.ref == ref), None)


# ---------- retrieval (code) ----------


def gather(
    ctx: ToolContext,
    topic: str,
    *,
    regions: list[str] | None,
    categories: list[str] | None,
    days: int,
    max_articles: int,
    today: date,
) -> tuple[list[BriefArticle], list[str]]:
    """Hybrid search for the topic; one numbered article per story, oldest to newest."""
    settings: RetrievalSettings = ctx.settings.retrieval.model_copy(
        update={
            "top_k": max_articles,
            "candidates_k": max(max_articles * 2, ctx.settings.retrieval.candidates_k),
            "max_chunks_per_article": 1,
        }
    )
    result = search(
        ctx.store,
        topic,
        SearchFilters(
            regions=regions,
            categories=categories,
            date_from=today - timedelta(days=days - 1),
            date_to=today,
        ),
        settings,
        ctx.embedder,
        ctx.reranker,
    )
    terms = topic_terms(topic)
    scored: list[tuple[SearchHit, BriefArticle]] = []
    dropped = 0
    for h in _one_per_article(result.hits):
        art = _article(ctx.store, 0, h)
        art.relevance = coverage(terms, f"{h.title} {h.summary} {' '.join(art.key_facts)}")
        if art.relevance < MIN_TERM_COVERAGE:
            dropped += 1
            continue
        scored.append((h, art))
    # Oldest first; within a day the most relevant story comes first.
    scored.sort(key=lambda p: (p[0].published_date, -p[1].relevance, p[0].title))
    articles = [a.model_copy(update={"ref": n}) for n, (_, a) in enumerate(scored, start=1)]
    notes = list(result.notes)
    if dropped:
        notes.append(
            f"{dropped} loosely related article(s) left out "
            f"(they mention fewer than half of: {', '.join(terms)})"
        )
    return articles, notes


_WORD = re.compile(r"[a-z0-9][a-z0-9.%$'-]*")


def topic_terms(topic: str) -> list[str]:
    """Significant words of the topic, reduced to a stem so 'rates' also matches 'rate'."""
    words = [w.strip(".'-") for w in _WORD.findall(topic.lower())]
    stems = [w.rstrip("s") if len(w) > 3 else w for w in words if w and w not in STOPWORDS]
    return list(dict.fromkeys(stems))


def coverage(terms: list[str], text: str) -> float:
    """Share of the topic's words that start a word in `text` (1.0 when there are none)."""
    if not terms:
        return 1.0
    lower = text.lower()
    found = sum(1 for t in terms if re.search(rf"\b{re.escape(t)}", lower))
    return found / len(terms)


def _one_per_article(hits: list[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    out = []
    for h in hits:
        if h.item_id not in seen:
            seen.add(h.item_id)
            out.append(h)
    return out


def _article(store: Store, ref: int, h: SearchHit) -> BriefArticle:
    row = store.item(h.item_id) or {}
    facts = json.loads(row.get("key_facts", "[]"))
    outlets = 1 + len(json.loads(row.get("also_reported_by", "[]")))
    return BriefArticle(
        ref=ref,
        item_id=h.item_id,
        title=h.title,
        url=h.url,
        source=h.source,
        region=h.region,
        category=h.category,
        published_date=h.published_date,
        summary=h.summary,
        key_facts=facts,
        outlets=outlets,
        evidence=f"{h.title}\n{h.summary}\n{' '.join(facts)}\n{h.text}",
    )


# ---------- rules writer ----------


def _first(text: str) -> str:
    sentences = split_sentences(text)
    first = sentences[0] if sentences else text
    if len(first) <= SENTENCE_MAX:
        return first
    return first[:SENTENCE_MAX].rsplit(" ", 1)[0].rstrip(" ,;:-") + "…"


def rule_brief(
    topic: str,
    articles: list[BriefArticle],
    *,
    date_from: date,
    date_to: date,
    regions: list[str] | None,
    notes: list[str] | None = None,
) -> TopicBrief:
    brief = TopicBrief(
        topic=topic,
        date_from=date_from.isoformat(),
        date_to=date_to.isoformat(),
        regions=regions,
        found=bool(articles),
        engine="rules",
        writer="Template",
        articles=articles,
        notes=list(notes or []),
    )
    if not articles:
        return brief
    ranked = sorted(articles, key=lambda a: (-a.relevance, -a.outlets, a.published_date, a.ref))
    brief.overview = [
        BriefLine(text=_first(a.summary) or a.title, refs=[a.ref]) for a in ranked[:2]
    ]
    brief.timeline = [
        BriefLine(text=a.title, refs=[a.ref], date=a.published_date)
        for a in articles[-MAX_TIMELINE:]
    ]
    by_region: dict[str, list[BriefArticle]] = {}
    for a in ranked:
        by_region.setdefault(a.region, []).append(a)
    if len(by_region) > 1:
        brief.by_region = [
            BriefLine(
                text=" | ".join(a.title for a in group[:2]),
                refs=[a.ref for a in group[:2]],
                region=region,
            )
            for region, group in sorted(by_region.items())
        ]
    seen: set[str] = set()
    for a in ranked:
        # Stored key facts exclude sentences already in the summary, so for short articles
        # the numeric sentence lives in the summary. Look at both.
        for sentence in [*a.key_facts, *split_sentences(a.summary)]:
            key = " ".join(sentence.lower().split())
            if key in seen or not _checkable_number(sentence):
                continue
            seen.add(key)
            brief.key_numbers.append(BriefLine(text=sentence[:SENTENCE_MAX], refs=[a.ref]))
            if len(brief.key_numbers) == MAX_NUMBERS:
                return brief
    return brief


# ---------- LLM writer ----------

SYSTEM = """\
You write a short brief about one topic from numbered news articles. Use only facts stated in
the articles. Do not add facts, figures, dates or names from your own knowledge, and never
calculate new figures. Every line must cite the numbers of the articles it is based on in
"refs". Do not write dates; they are added for you. Reply with JSON only."""

USER = """\
Topic: {topic}
Period: {date_from} to {date_to}

Write:
- "overview": 2 or 3 lines saying what happened.
- "timeline": up to {timeline} lines for the key developments, oldest first. One event per line.
- "by_region": one line per region that covered the topic, saying how that region's articles
  covered it (leave empty if only one region is present). Set "region" to the region code.
- "key_numbers": up to {numbers} lines, each stating one concrete figure from the articles
  (rate, amount, percentage, count) with what it measures.

Articles (oldest first):
{blocks}"""


def _line_schema(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    props: dict[str, Any] = {
        "text": {"type": "string"},
        "refs": {"type": "array", "items": {"type": "integer"}},
    }
    props.update(extra or {})
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def _schema() -> dict[str, Any]:
    plain = _line_schema()
    region = _line_schema({"region": {"type": "string"}})
    return {
        "type": "object",
        "properties": {
            "overview": {"type": "array", "items": plain},
            "timeline": {"type": "array", "items": plain},
            "by_region": {"type": "array", "items": region},
            "key_numbers": {"type": "array", "items": plain},
        },
        "required": ["overview", "timeline", "by_region", "key_numbers"],
        "additionalProperties": False,
    }


class _LLMBrief(BaseModel):
    model_config = ConfigDict(extra="ignore")

    overview: list[BriefLine] = Field(default_factory=list)
    timeline: list[BriefLine] = Field(default_factory=list)
    by_region: list[BriefLine] = Field(default_factory=list)
    key_numbers: list[BriefLine] = Field(default_factory=list)


def _blocks(articles: list[BriefArticle]) -> str:
    return "\n\n".join(
        f"[{a.ref}] {a.title} ({a.source}, {a.region}, {a.published_date})\n"
        f"    {a.summary[:500]}" + (f"\n    Facts: {'; '.join(a.key_facts)}" if a.key_facts else "")
        for a in articles
    )


def _norm_number(token: str) -> str:
    return token.replace(",", "").rstrip("%").rstrip(".")


def _is_checkable(token: str) -> bool:
    core = _norm_number(token)
    return token.endswith("%") or "." in core or len(core) >= 3


def _is_year(token: str) -> bool:
    return token.isdigit() and 1900 <= int(token) <= 2100


def _checkable_number(text: str) -> bool:
    """A sentence worth listing under key numbers: has a figure that is not just a year."""
    return any(_is_checkable(t) and not _is_year(t) for t in _NUMBER.findall(text))


def numbers_supported(text: str, evidence: str) -> bool:
    """Every checkable figure in `text` must appear in the cited articles' text.
    Checkable: has a % sign, a decimal point, or three or more digits. Plain one- or
    two-digit integers (days of the month, small counts) are not checked."""
    haystack = evidence.replace(",", "")
    return all(
        _norm_number(token) in haystack for token in _NUMBER.findall(text) if _is_checkable(token)
    )


def _valid(line: BriefLine, brief: TopicBrief, allowed: set[int]) -> BriefLine | None:
    refs = [r for r in dict.fromkeys(line.refs) if r in allowed]
    text = line.text.strip()
    if not refs or not text:
        return None
    cited = [a for r in refs if (a := brief.article(r)) is not None]
    if not numbers_supported(text, "\n".join(a.evidence for a in cited)):
        return None
    when = min((a.published_date for a in cited), default=None)
    return BriefLine(text=text, refs=refs, date=when, region=line.region)


async def llm_brief(client: LLMClient, base: TopicBrief, keys: KeyStore) -> TopicBrief:
    """Upgrade a rules brief with LLM-written lines. Falls back to `base` on any failure."""
    if not base.articles:
        return base
    user = USER.format(
        topic=base.topic,
        date_from=base.date_from,
        date_to=base.date_to,
        timeline=MAX_TIMELINE,
        numbers=MAX_NUMBERS,
        blocks=_blocks(base.articles),
    )
    try:
        text = await client.complete(
            system=SYSTEM, user=user, schema=_schema(), max_tokens=3000, task="chat"
        )
        parsed = _LLMBrief.model_validate(parse_json_object(text))
    except (LLMError, ValueError, ValidationError) as exc:
        reason = keys.redact(str(exc))[:200]
        log.info("Topic brief falls back to the template: %s", reason)
        return base.model_copy(update={"notes": [*base.notes, f"LLM unavailable ({reason})"]})

    all_refs = {a.ref for a in base.articles}
    regions_present = {a.region for a in base.articles}

    def keep(lines: list[BriefLine], limit: int) -> list[BriefLine]:
        return [v for line in lines if (v := _valid(line, base, all_refs))][:limit]

    overview = keep(parsed.overview, 3)
    timeline = sorted(keep(parsed.timeline, MAX_TIMELINE), key=lambda ln: ln.date or "")
    by_region = [ln for ln in keep(parsed.by_region, 6) if ln.region in regions_present]
    key_numbers = keep(parsed.key_numbers, MAX_NUMBERS)

    notes = list(base.notes)
    filled: list[str] = []
    if not overview:
        overview = base.overview
        filled.append("overview")
    if not timeline:
        timeline = base.timeline
        filled.append("timeline")
    if not by_region and base.by_region:
        by_region = base.by_region
        filled.append("by region")
    if not key_numbers and base.key_numbers:
        key_numbers = base.key_numbers
        filled.append("key numbers")
    if filled:
        notes.append("Template used for: " + ", ".join(filled))
    return base.model_copy(
        update={
            "engine": "llm",
            "writer": f"LLM · {client.provider} ({client.model})",
            "overview": overview,
            "timeline": timeline,
            "by_region": by_region,
            "key_numbers": key_numbers,
            "notes": notes,
        }
    )


# ---------- entry point ----------


async def build_brief(
    ctx: ToolContext,
    topic: str,
    *,
    regions: list[str] | None = None,
    categories: list[str] | None = None,
    days: int = 30,
    max_articles: int = MAX_ARTICLES,
    client: LLMClient | None = None,
    today: date | None = None,
) -> TopicBrief:
    """Retrieve, build the rules brief, and upgrade it with `client` if one is given."""
    today = today or date.today()
    articles, notes = gather(
        ctx,
        topic,
        regions=regions,
        categories=categories,
        days=days,
        max_articles=max_articles,
        today=today,
    )
    base = rule_brief(
        topic,
        articles,
        date_from=today - timedelta(days=days - 1),
        date_to=today,
        regions=regions,
        notes=notes,
    )
    if client is None or not base.found:
        return base
    return await llm_brief(client, base, ctx.keys)


# ---------- rendering and saving ----------


def _label(b: TopicBrief) -> str:
    if b.engine == "llm":
        return (
            f"Written by {b.writer} from the listed articles. "
            "Dates and links added by code; figures checked against the articles."
        )
    return "Written by the template from article text (no LLM). Dates and links added by code."


def _cite(b: TopicBrief, refs: list[int]) -> str:
    links = [f"[{a.source}, {a.published_date}]({a.url})" for r in refs if (a := b.article(r))]
    return " · ".join(links)


def no_news_text(b: TopicBrief) -> str:
    scope = f"{b.date_from} to {b.date_to}" + (
        f", regions {', '.join(b.regions)}" if b.regions else ""
    )
    return f"I don't have news on '{b.topic}' in my knowledge base for {scope}."


def render_markdown(b: TopicBrief) -> str:
    out = [f"# Topic brief: {b.topic}", "", f"_{b.date_from} to {b.date_to}. {_label(b)}_", ""]
    if not b.found:
        return "\n".join([*out, no_news_text(b), ""])

    def bullets(lines: list[BriefLine], with_date: bool = False, with_region: bool = False) -> None:
        for ln in lines:
            prefix = f"**{ln.date}** " if with_date and ln.date else ""
            if with_region and ln.region:
                prefix = f"**{REGION_NAMES.get(ln.region, ln.region)}:** "
            out.append(f"- {prefix}{ln.text} ({_cite(b, ln.refs)})")

    out += ["## Overview", ""]
    bullets(b.overview)
    out += ["", "## Timeline", ""]
    bullets(b.timeline, with_date=True)
    if b.by_region:
        out += ["", "## By region", ""]
        bullets(b.by_region, with_region=True)
    if b.key_numbers:
        out += ["", "## Key numbers", ""]
        bullets(b.key_numbers)
    out += ["", f"## Sources ({len(b.articles)} articles)", ""]
    out += [f"{a.ref}. [{a.title}]({a.url}), {a.source}, {a.published_date}" for a in b.articles]
    if b.notes:
        out += ["", *(f"_Note: {n}_" for n in b.notes)]  # blank line: else it joins the list
    return "\n".join(out) + "\n"


def render_html(b: TopicBrief) -> str:
    e = html.escape

    def cite(refs: list[int]) -> str:
        links = [
            f'<a href="{e(a.url, quote=True)}">{e(a.source)}, {e(a.published_date)}</a>'
            for r in refs
            if (a := b.article(r))
        ]
        return " &middot; ".join(links)

    def items(lines: list[BriefLine], with_date: bool = False, with_region: bool = False) -> str:
        rows = []
        for ln in lines:
            prefix = f"<strong>{e(ln.date)}</strong> " if with_date and ln.date else ""
            if with_region and ln.region:
                prefix = f"<strong>{e(REGION_NAMES.get(ln.region, ln.region))}:</strong> "
            rows.append(f"<li>{prefix}{e(ln.text)} <span>({cite(ln.refs)})</span></li>")
        return "<ul>" + "".join(rows) + "</ul>"

    parts = [
        '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:720px;'
        'line-height:1.5;color:#1f1f1f">',
        f'<h1 style="font-size:22px">Topic brief: {e(b.topic)}</h1>',
        f'<p style="color:#555;font-size:13px"><em>{e(b.date_from)} to {e(b.date_to)}. '
        f"{e(_label(b))}</em></p>",
    ]
    if not b.found:
        parts.append(f"<p>{e(no_news_text(b))}</p></div>")
        return "\n".join(parts)
    parts += ['<h2 style="font-size:18px">Overview</h2>', items(b.overview)]
    parts += ['<h2 style="font-size:18px">Timeline</h2>', items(b.timeline, with_date=True)]
    if b.by_region:
        parts += ['<h2 style="font-size:18px">By region</h2>', items(b.by_region, with_region=True)]
    if b.key_numbers:
        parts += ['<h2 style="font-size:18px">Key numbers</h2>', items(b.key_numbers)]
    parts.append(f'<h2 style="font-size:18px">Sources ({len(b.articles)} articles)</h2><ol>')
    parts += [
        f'<li><a href="{e(a.url, quote=True)}">{e(a.title)}</a>, {e(a.source)}, '
        f"{e(a.published_date)}</li>"
        for a in b.articles
    ]
    parts.append("</ol>")
    parts += [f'<p style="color:#555;font-size:12px"><em>Note: {e(n)}</em></p>' for n in b.notes]
    parts.append("</div>")
    return "\n".join(parts)


def slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:60] or "topic"


def write_brief(b: TopicBrief, briefs_dir: Path) -> tuple[Path, Path]:
    briefs_dir.mkdir(parents=True, exist_ok=True)
    base = f"{slug(b.topic)}-{b.date_to}"
    md, page = briefs_dir / f"{base}.md", briefs_dir / f"{base}.html"
    md.write_text(render_markdown(b), "utf-8")
    page.write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(b.topic)}'
        f"</title></head><body>{render_html(b)}</body></html>",
        "utf-8",
    )
    (briefs_dir / f"{base}.json").write_text(b.model_dump_json(indent=2), "utf-8")
    return md, page
