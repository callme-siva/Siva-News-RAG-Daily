"""LLMEngine: an LLM writes the summary, facts and entities; code validates everything (FR10).

Per item: ask for JSON, validate it with pydantic, retry once with the validation error,
then fall back to RuleEngine for that item. Code still owns category membership and the
item's identity; the LLM can only choose among configured categories.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from newsrag.config import AppConfig
from newsrag.engines.rules import RuleEngine
from newsrag.llm import LLMClient, LLMError, parse_json_object
from newsrag.models import EngineName, Entities, Item, Processed
from newsrag.secrets import KeyStore

log = logging.getLogger("newsrag.engines.llm")

BODY_MAX_CHARS = 6000
MAX_TOKENS = 1200

SYSTEM_PROMPT = """\
You process one news article for a personal news digest covering the United States, Europe
and India. Work only from the article text you are given. Do not add facts, figures, dates or
names that are not in the text. Reply with a single JSON object that matches the schema.
No markdown, no commentary."""

USER_TEMPLATE = """Categories and what belongs in them:
{rules}

Decide whether the article is relevant to its assigned category ("relevant": false for
shopping deals, lifestyle, sport, celebrity or other off-topic content). Keep the assigned
category unless the article clearly belongs to another listed category.

Write:
- "summary": 2-3 factual sentences using only the article text.
- "key_facts": up to 5 short statements with the concrete numbers, dates or decisions from the text.
- "entities": names that appear in the text, grouped as companies, people, organisations, countries.

Article
Assigned category: {category}
Region: {region}
Headline: {title}
Source: {source}
Published: {published}
URL: {url}
Text: {body}"""

CATEGORY_HINTS = {
    "Technology": "AI, software, internet, telecom, semiconductors, startups and funding, "
    "cybersecurity, big-tech business, tech regulation",
    "Finance": "markets, central banks and regulators, interest rates, inflation, GDP, banking, "
    "earnings, IPOs, M&A, commodities, currencies, budgets and tax policy",
    "Politics": "government decisions, legislation, parliaments, elections, court rulings on "
    "public policy, diplomacy, defence policy",
}


class _LLMEntities(BaseModel):
    model_config = ConfigDict(extra="ignore")

    companies: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    organisations: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)


class LLMProcessed(BaseModel):
    """What the model must return. Extra keys are ignored; missing ones fail validation."""

    model_config = ConfigDict(extra="ignore")

    relevant: bool
    category: str
    summary: str = Field(min_length=1)
    key_facts: list[str] = Field(default_factory=list)
    entities: _LLMEntities = Field(default_factory=_LLMEntities)


def output_schema(categories: list[str]) -> dict[str, Any]:
    """JSON schema sent to providers that support structured output."""
    str_list = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "relevant": {"type": "boolean"},
            "category": {"type": "string", "enum": categories},
            "summary": {"type": "string"},
            "key_facts": str_list,
            "entities": {
                "type": "object",
                "properties": {
                    "companies": str_list,
                    "people": str_list,
                    "organisations": str_list,
                    "countries": str_list,
                },
                "required": ["companies", "people", "organisations", "countries"],
                "additionalProperties": False,
            },
        },
        "required": ["relevant", "category", "summary", "key_facts", "entities"],
        "additionalProperties": False,
    }


class LLMEngine:
    name = EngineName.LLM

    def __init__(
        self, client: LLMClient, cfg: AppConfig, rules: RuleEngine, keys: KeyStore
    ) -> None:
        self._client = client
        self._cfg = cfg
        self._rules = rules
        self._keys = keys
        self._categories = [c.name for c in cfg.categories]
        self._schema = output_schema(self._categories)
        self._rules_text = "\n".join(
            f"- {c}: {CATEGORY_HINTS.get(c, 'news about ' + c.lower())}" for c in self._categories
        )
        self.label = f"LLM · {client.provider} ({client.model})"

    def prompt(self, item: Item) -> str:
        return USER_TEMPLATE.format(
            rules=self._rules_text,
            category=item.category,
            region=item.region,
            title=item.title,
            source=item.source,
            published=item.published_at.date().isoformat(),
            url=item.url,
            body=item.body[:BODY_MAX_CHARS] or "(no body text; use the headline only)",
        )

    async def _attempt(self, user: str) -> LLMProcessed:
        text = await self._client.complete(
            system=SYSTEM_PROMPT,
            user=user,
            schema=self._schema,
            max_tokens=MAX_TOKENS,
            task="processing",
        )
        return LLMProcessed.model_validate(parse_json_object(text))

    async def process(self, item: Item) -> Processed:
        user = self.prompt(item)
        try:
            try:
                out = await self._attempt(user)
            except (ValueError, ValidationError) as first:
                retry = (
                    f"{user}\n\nYour previous reply was not valid ({_short(first)}). "
                    "Reply again with only the JSON object."
                )
                out = await self._attempt(retry)
        except (LLMError, ValueError, ValidationError) as exc:
            reason = self._keys.redact(_short(exc))
            log.info("Falling back to rules for %s: %s", item.url, reason)
            return self._rules.process_sync(item, fallback_reason=reason)

        category = out.category if out.category in self._categories else item.category
        return Processed(
            item_id=item.item_id,
            relevant=out.relevant,
            category=category,
            summary=out.summary.strip(),
            key_facts=[f.strip() for f in out.key_facts if f.strip()][:5],
            entities=Entities.model_validate(out.entities.model_dump()),
            engine=EngineName.LLM,
        )


def _short(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        return f"field {'.'.join(str(p) for p in first['loc'])}: {first['msg']}"
    return str(exc)[:200] or type(exc).__name__
