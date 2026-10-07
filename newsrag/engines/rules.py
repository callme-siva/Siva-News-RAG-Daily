"""RuleEngine: pure-Python processing that works with no key and no model (FR9).

Summaries are extractive (the lead sentences), key facts are sentences carrying numbers,
and entities come from simple capitalisation and keyword rules. Nothing is invented:
every output string is copied from the article text.
"""

from __future__ import annotations

import re

from newsrag.config import AppConfig
from newsrag.models import EngineName, Entities, Item, Processed
from newsrag.pipeline.filter import is_official

SUMMARY_MAX_CHARS = 400
SUMMARY_MAX_SENTENCES = 2
MAX_KEY_FACTS = 3
MAX_ENTITIES_PER_TYPE = 8

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(])")
_NUMBERISH = re.compile(r"\d|%|\b(per ?cent|billion|million|crore|lakh|trillion)\b", re.I)
_WORD = r"(?:[A-Z][a-zA-Z&'-]+|[A-Z]{2,})"
_CAPS_RUN = re.compile(rf"\b{_WORD}(?:\s+(?:of|and|for|the|de|&)?\s*{_WORD})*")
_ACRONYM = re.compile(r"^[A-Z]{2,6}$")

_ORG_WORDS = frozenset(
    """bank ministry commission court council agency authority board reserve parliament
    congress senate government union party fund exchange institute university department
    office committee assembly bureau nations organisation organization federation police""".split()
)
_COMPANY_WORDS = frozenset(
    """inc ltd limited corp corporation plc group holdings llc ag sa co technologies motors
    industries airlines pharma labs""".split()
)
_COUNTRIES = frozenset(
    """India|United States|US|USA|UK|United Kingdom|Britain|China|Japan|Germany|France|Italy|
    Spain|Netherlands|Belgium|Ireland|Poland|Ukraine|Russia|Canada|Mexico|Brazil|Australia|
    Pakistan|Bangladesh|Sri Lanka|Nepal|Israel|Iran|Saudi Arabia|Turkey|Greece|Sweden|Norway|
    Denmark|Finland|Austria|Switzerland|Portugal|South Korea|Taiwan|Singapore|Indonesia|
    Vietnam|Europe|European Union|EU""".replace("\n", "")
    .replace("    ", "")
    .split("|")
)
_STOP_STARTS = frozenset(
    """The A An This That These Those It He She They We I In On At For But And Or As If When
    While After Before Monday Tuesday Wednesday Thursday Friday Saturday Sunday January
    February March April May June July August September October November December Read Also
    Photo Watch Live Updates""".split()
)
_TITLES = frozenset(
    """Mr Mrs Ms Dr Prof Governor President Minister Chancellor Senator Judge Justice Chief
    CEO Chairman Chairwoman Chair Secretary Commissioner Director Analyst""".split()
)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def extractive_summary(title: str, body: str) -> str:
    """The lead sentences of the body (at most 2, at most 400 characters)."""
    sentences = split_sentences(body)
    if not sentences:
        return title
    out = ""
    for s in sentences[:SUMMARY_MAX_SENTENCES]:
        candidate = f"{out} {s}".strip()
        if out and len(candidate) > SUMMARY_MAX_CHARS:
            break
        out = candidate
    return out[:SUMMARY_MAX_CHARS].rstrip()


def key_facts(body: str, summary: str) -> list[str]:
    """Sentences with numbers that are not already in the summary."""
    facts = []
    for s in split_sentences(body):
        if s in summary or not _NUMBERISH.search(s) or len(s) > 300:
            continue
        facts.append(s)
        if len(facts) == MAX_KEY_FACTS:
            break
    return facts


def _strip_edges(words: list[str]) -> list[str]:
    while words and words[0] in _STOP_STARTS:
        words = words[1:]
    return words


def extract_entities(text: str) -> Entities:
    ents = Entities()
    seen: set[str] = set()
    for sentence in split_sentences(text):
        for match in _CAPS_RUN.finditer(sentence):
            words = _strip_edges(match.group(0).split())
            phrase = " ".join(words).strip(" ,'-")
            if len(phrase) < 2 or phrase in seen:
                continue
            seen.add(phrase)
            lower = {w.lower() for w in words}
            if phrase in _COUNTRIES:
                ents.countries.append(phrase)
            elif lower & _COMPANY_WORDS:
                ents.companies.append(phrase)
            elif lower & _ORG_WORDS or _ACRONYM.match(phrase):
                ents.organisations.append(phrase)
            else:
                name = [w for w in words if w not in _TITLES]
                if 2 <= len(name) <= 3 and all(w[0].isupper() and w[1:].islower() for w in name):
                    person = " ".join(name)
                    if person not in ents.people:
                        ents.people.append(person)
    for field in ("companies", "people", "organisations", "countries"):
        setattr(ents, field, getattr(ents, field)[:MAX_ENTITIES_PER_TYPE])
    return ents


class RuleEngine:
    name = EngineName.RULES
    label = "Rules"

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg

    def is_relevant(self, item: Item) -> bool:
        try:
            include = self._cfg.category(item.category).include_re()
        except KeyError:
            return False
        if include is None or is_official(item.url, self._cfg.official_domains):
            return True
        return bool(include.search(f"{item.title} {item.body}"))

    def process_sync(self, item: Item, fallback_reason: str | None = None) -> Processed:
        summary = extractive_summary(item.title, item.body)
        return Processed(
            item_id=item.item_id,
            relevant=self.is_relevant(item),
            category=item.category,
            summary=summary,
            key_facts=key_facts(item.body, summary),
            entities=extract_entities(f"{item.title}. {item.body}"),
            engine=EngineName.RULES,
            fallback_reason=fallback_reason,
        )

    async def process(self, item: Item) -> Processed:
        return self.process_sync(item)
