"""Today's briefing (REQUIREMENTS FR17-FR19).

Code selects and numbers the articles. The writer (template or LLM) only produces lines of
{headline, why, refs}. Code validates every ref, drops lines with none, and attaches all
links itself, so a link can never be invented. If the LLM fails or leaves a section empty,
the template fills it and the digest says so.
"""

from __future__ import annotations

import html
import json
import logging
import os
import smtplib
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from newsrag.config import AppConfig
from newsrag.engines.rules import split_sentences
from newsrag.llm import LLMClient, LLMError, parse_json_object
from newsrag.secrets import KeyStore
from newsrag.settings import BriefingSettings
from newsrag.store import Store

log = logging.getLogger("newsrag.briefing")

TOP_COUNT = 3
WINDOW_HOURS = 24
WHY_MAX_CHARS = 240


class DigestArticle(BaseModel):
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
    outlets: int


class DigestLine(BaseModel):
    model_config = ConfigDict(extra="ignore")

    headline: str = Field(min_length=1)
    why: str = ""
    refs: list[int] = Field(default_factory=list)


class DigestSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str
    category: str
    lines: list[DigestLine] = Field(default_factory=list)


class Digest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    window_from: str
    window_to: str
    engine: str  # "rules" or "llm"
    writer: str  # e.g. "Template" or "LLM · anthropic (claude-opus-5-5)"
    top: list[DigestLine] = Field(default_factory=list)
    sections: list[DigestSection] = Field(default_factory=list)
    articles: list[DigestArticle] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def article(self, ref: int) -> DigestArticle | None:
        return next((a for a in self.articles if a.ref == ref), None)


# ---------- selection (code) ----------


def select_articles(
    store: Store,
    cfg: AppConfig,
    *,
    now: datetime,
    per_group: int,
    window_hours: int = WINDOW_HOURS,
) -> tuple[list[DigestArticle], datetime]:
    start = now - timedelta(hours=window_hours)
    rows = store.recent_items(int(start.timestamp()), int(now.timestamp()))
    region_order = {code: i for i, code in enumerate(cfg.region_codes())}
    cat_order = {c.name: i for i, c in enumerate(cfg.categories)}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["region"], r["category"]), []).append(r)
    chosen: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: (region_order.get(k[0], 99), cat_order.get(k[1], 99))):
        ranked = sorted(
            groups[key],
            key=lambda r: (-len(json.loads(r["also_reported_by"])), -r["published_ts"]),
        )
        chosen.extend(ranked[:per_group])
    articles = [
        DigestArticle(
            ref=i,
            item_id=r["item_id"],
            title=r["title"],
            url=r["url"],
            source=r["source"],
            region=r["region"],
            category=r["category"],
            published_date=datetime.fromtimestamp(r["published_ts"], tz=UTC).date().isoformat(),
            summary=r["summary"],
            outlets=1 + len(json.loads(r["also_reported_by"])),
        )
        for i, r in enumerate(chosen, start=1)
    ]
    return articles, start


def _skeleton(cfg: AppConfig, regions: list[str]) -> list[DigestSection]:
    return [
        DigestSection(region=r.code, category=c.name)
        for r in cfg.regions
        if r.code in regions
        for c in cfg.categories
    ]


def _first_sentence(text: str) -> str:
    sentences = split_sentences(text)
    first = sentences[0] if sentences else text
    if len(first) <= WHY_MAX_CHARS:
        return first
    cut = first[:WHY_MAX_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return f"{cut}…"


def _rule_line(a: DigestArticle) -> DigestLine:
    why = _first_sentence(a.summary)
    return DigestLine(headline=a.title, why="" if why == a.title else why, refs=[a.ref])


def _rule_top(articles: list[DigestArticle]) -> list[DigestLine]:
    """Most widely reported first, taking one story per region before any second one."""
    ranked = sorted(articles, key=lambda a: (-a.outlets, a.ref))
    picked: list[DigestArticle] = []
    regions_used: set[str] = set()
    for a in ranked:
        if a.region not in regions_used:
            picked.append(a)
            regions_used.add(a.region)
        if len(picked) == TOP_COUNT:
            break
    for a in ranked:
        if len(picked) == TOP_COUNT:
            break
        if a not in picked:
            picked.append(a)
    return [_rule_line(a) for a in picked]


def rule_digest(
    articles: list[DigestArticle],
    cfg: AppConfig,
    regions: list[str],
    *,
    now: datetime,
    start: datetime,
) -> Digest:
    sections = _skeleton(cfg, regions)
    for s in sections:
        s.lines = [
            _rule_line(a) for a in articles if a.region == s.region and a.category == s.category
        ]
    return Digest(
        date=now.date().isoformat(),
        window_from=start.isoformat(timespec="minutes"),
        window_to=now.isoformat(timespec="minutes"),
        engine="rules",
        writer="Template",
        top=_rule_top(articles),
        sections=sections,
        articles=articles,
    )


# ---------- LLM writer ----------

SYSTEM = """\
You write a short morning news briefing from numbered articles. Use only facts stated in
the articles. Do not add facts, numbers, dates or names from your own knowledge. Every line
must cite the numbers of the articles it is based on in "refs". Reply with JSON only."""

USER = """\
Write the briefing for {date}.

Return:
- "top": exactly {top} lines covering the most important stories across all regions.
- "sections": for each region and category below, {per_group_text} lines.
Each line has "headline" (a short bold-style headline in your own words), "why" ({why_len}
explaining why it matters, based only on the article) and "refs" (article numbers).
A section's lines may only cite articles listed under that section.

{blocks}"""


def _schema() -> dict[str, Any]:
    line = {
        "type": "object",
        "properties": {
            "headline": {"type": "string"},
            "why": {"type": "string"},
            "refs": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["headline", "why", "refs"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "top": {"type": "array", "items": line},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "region": {"type": "string"},
                        "category": {"type": "string"},
                        "lines": {"type": "array", "items": line},
                    },
                    "required": ["region", "category", "lines"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["top", "sections"],
        "additionalProperties": False,
    }


class _LLMDigest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    top: list[DigestLine] = Field(default_factory=list)
    sections: list[DigestSection] = Field(default_factory=list)


def _blocks(digest: Digest) -> str:
    parts = []
    for s in digest.sections:
        arts = [a for a in digest.articles if a.region == s.region and a.category == s.category]
        if not arts:
            continue
        lines = "\n".join(
            f"[{a.ref}] {a.title} ({a.source}, {a.published_date})\n    {a.summary[:600]}"
            for a in arts
        )
        parts.append(f"REGION {s.region} / CATEGORY {s.category}\n{lines}")
    return "\n\n".join(parts)


def _valid_line(line: DigestLine, allowed: set[int]) -> DigestLine | None:
    refs = [r for r in dict.fromkeys(line.refs) if r in allowed]
    if not refs or not line.headline.strip():
        return None
    return DigestLine(headline=line.headline.strip(), why=line.why.strip(), refs=refs)


async def llm_digest(
    client: LLMClient, base: Digest, settings: BriefingSettings, keys: KeyStore
) -> Digest:
    """Ask the LLM to write the lines; validate refs; fill gaps from `base` (the template)."""
    if not base.articles:
        return base
    user = USER.format(
        date=base.date,
        top=min(TOP_COUNT, len(base.articles)),
        per_group_text=f"up to {settings.items_per_category}",
        why_len="one sentence" if settings.style == "brief" else "one or two sentences",
        blocks=_blocks(base),
    )
    try:
        text = await client.complete(
            system=SYSTEM, user=user, schema=_schema(), max_tokens=4000, task="digest"
        )
        parsed = _LLMDigest.model_validate(parse_json_object(text))
    except (LLMError, ValueError, ValidationError) as exc:
        reason = keys.redact(str(exc))[:200]
        log.info("Digest falls back to the template: %s", reason)
        return base.model_copy(update={"notes": [*base.notes, f"LLM unavailable ({reason})"]})

    all_refs = {a.ref for a in base.articles}
    top = [v for line in parsed.top if (v := _valid_line(line, all_refs))][:TOP_COUNT]
    llm_sections = {(s.region, s.category): s for s in parsed.sections}
    sections: list[DigestSection] = []
    filled: list[str] = []
    for s in base.sections:
        allowed = {
            a.ref for a in base.articles if a.region == s.region and a.category == s.category
        }
        got = llm_sections.get((s.region, s.category))
        lines = [v for line in (got.lines if got else []) if (v := _valid_line(line, allowed))]
        if allowed and not lines:
            lines = s.lines
            filled.append(f"{s.region}/{s.category}")
        sections.append(s.model_copy(update={"lines": lines}))
    notes = list(base.notes)
    if not top:
        top = base.top
        notes.append("Top of the day written by the template (LLM gave no valid lines)")
    if filled:
        notes.append("Template used for: " + ", ".join(filled))
    return base.model_copy(
        update={
            "engine": "llm",
            "writer": f"LLM · {client.provider} ({client.model})",
            "top": top,
            "sections": sections,
            "notes": notes,
        }
    )


# ---------- rendering ----------

REGION_NAMES = {"US": "United States", "EU": "Europe", "IN": "India"}


def _label(d: Digest) -> str:
    if d.engine == "llm":
        return f"Written by {d.writer} from the listed articles. Links added by code."
    return "Written by the template from article text (no LLM). Links added by code."


def render_markdown(d: Digest) -> str:
    def cite(refs: list[int]) -> str:
        links = [f"[{a.source}, {a.published_date}]({a.url})" for r in refs if (a := d.article(r))]
        return " · ".join(links)

    out = [f"# Today's Briefing: {d.date}", "", f"_{_label(d)}_", ""]
    if not d.articles:
        out += ["No new stories in this period.", ""]
        return "\n".join(out)
    out += ["## Top of the day", ""]
    out += [
        f"- **{line.headline}** {line.why} ({cite(line.refs)})".replace("  ", " ") for line in d.top
    ]
    current = None
    for s in d.sections:
        if s.region != current:
            current = s.region
            out += ["", f"## {REGION_NAMES.get(s.region, s.region)}"]
        out += ["", f"### {s.category}", ""]
        if not s.lines:
            out.append("No new stories today.")
        for line in s.lines:
            out.append(f"- **{line.headline}** {line.why} ({cite(line.refs)})".replace("  ", " "))
    if d.notes:
        out += ["", "---", *(f"_Note: {n}_" for n in d.notes)]
    return "\n".join(out) + "\n"


def render_html(d: Digest) -> str:
    e = html.escape

    def cite(refs: list[int]) -> str:
        links = [
            f'<a href="{e(a.url, quote=True)}">{e(a.source)}, {e(a.published_date)}</a>'
            for r in refs
            if (a := d.article(r))
        ]
        return " &middot; ".join(links)

    def item(line: DigestLine) -> str:
        why = f" {e(line.why)}" if line.why else ""
        return f"<li><strong>{e(line.headline)}</strong>{why} <span>({cite(line.refs)})</span></li>"

    parts = [
        '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:720px;'
        'line-height:1.5;color:#1f1f1f">',
        f'<h1 style="font-size:22px">Today\'s Briefing: {e(d.date)}</h1>',
        f'<p style="color:#555;font-size:13px"><em>{e(_label(d))}</em></p>',
    ]
    if not d.articles:
        parts.append("<p>No new stories in this period.</p></div>")
        return "\n".join(parts)
    parts += ['<h2 style="font-size:18px">Top of the day</h2>', "<ul>"]
    parts += [item(line) for line in d.top]
    parts.append("</ul>")
    current = None
    for s in d.sections:
        if s.region != current:
            current = s.region
            parts.append(
                f'<h2 style="font-size:18px">{e(REGION_NAMES.get(s.region, s.region))}</h2>'
            )
        parts.append(f'<h3 style="font-size:16px">{e(s.category)}</h3>')
        if not s.lines:
            parts.append("<p>No new stories today.</p>")
        else:
            parts += ["<ul>", *(item(line) for line in s.lines), "</ul>"]
    for n in d.notes:
        parts.append(f'<p style="color:#555;font-size:12px"><em>Note: {e(n)}</em></p>')
    parts.append("</div>")
    return "\n".join(parts)


def write_digest(d: Digest, digests_dir: Path) -> tuple[Path, Path]:
    digests_dir.mkdir(parents=True, exist_ok=True)
    md = digests_dir / f"{d.date}.md"
    page = digests_dir / f"{d.date}.html"
    md.write_text(render_markdown(d), "utf-8")
    page.write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>Briefing {html.escape(d.date)}'
        f"</title></head><body>{render_html(d)}</body></html>",
        "utf-8",
    )
    (digests_dir / f"{d.date}.json").write_text(d.model_dump_json(indent=2), "utf-8")
    return md, page


# ---------- email (optional, FR19) ----------

SMTP_ENV = {
    "host": "NEWSRAG_SMTP_HOST",
    "port": "NEWSRAG_SMTP_PORT",
    "user": "NEWSRAG_SMTP_USER",
    "sender": "NEWSRAG_SMTP_FROM",
    "to": "NEWSRAG_SMTP_TO",
}


class EmailNotConfigured(Exception):
    pass


def send_email(d: Digest, keys: KeyStore, environ: dict[str, str] | None = None) -> str:
    """Send the digest by SMTP (STARTTLS). The password comes from the KeyStore ('smtp'),
    never from settings. Returns the recipient address."""
    env = dict(os.environ) if environ is None else environ
    cfg = {k: env.get(v, "").strip() for k, v in SMTP_ENV.items()}
    password = keys.get("smtp")
    missing = [SMTP_ENV[k] for k in ("host", "user", "to") if not cfg[k]]
    if missing or not password:
        need = missing + ([] if password else ["NEWSRAG_SMTP_PASSWORD"])
        raise EmailNotConfigured("set " + ", ".join(need))
    msg = EmailMessage()
    msg["Subject"] = f"Today's Briefing - {d.date}"
    msg["From"] = cfg["sender"] or cfg["user"]
    msg["To"] = cfg["to"]
    msg.set_content(render_markdown(d))
    msg.add_alternative(render_html(d), subtype="html")
    with smtplib.SMTP(cfg["host"], int(cfg["port"] or 587), timeout=30) as smtp:
        smtp.starttls()
        smtp.login(cfg["user"], password)
        smtp.send_message(msg)
    return cfg["to"]
