"""Tool-shaped core operations (REQUIREMENTS 12.1, AG1-AG3).

Each tool takes the session context and one typed input model, returns one typed output
model, prints nothing and has no UI code. The UI, CLI and a future agent all call these.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time

from pydantic import BaseModel, ConfigDict, Field

from newsrag.runner import RunOutcome, run_ingest
from newsrag.search import SearchFilters, SearchResult, search
from newsrag.tools.context import ToolContext
from newsrag.topic import TopicBrief, build_brief


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- search_news ----------


class SearchNewsInput(_In):
    query: str = Field(min_length=1, description="Search text with entity names and topic words")
    regions: list[str] | None = Field(None, description="Region codes, e.g. ['IN', 'US']")
    categories: list[str] | None = Field(None, description="Category names, e.g. ['Finance']")
    date_from: date | None = Field(None, description="Earliest published date (inclusive)")
    date_to: date | None = Field(None, description="Latest published date (inclusive)")
    top_k: int | None = Field(None, ge=1, le=50, description="How many results (default: setting)")


def search_news(ctx: ToolContext, args: SearchNewsInput) -> SearchResult:
    """Hybrid search over stored news with region, category and date filters."""
    r = ctx.settings.retrieval.model_copy()
    if args.top_k:
        r.top_k = min(args.top_k, r.candidates_k)
    filters = SearchFilters(
        regions=args.regions,
        categories=args.categories,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    return search(ctx.store, args.query, filters, r, ctx.embedder, ctx.reranker)


# ---------- get_article ----------


class GetArticleInput(_In):
    item_id: str = Field(min_length=1)


class Article(_Out):
    item_id: str
    title: str
    url: str
    source: str
    region: str
    category: str
    published_date: str
    summary: str
    key_facts: list[str]
    entities: dict[str, list[str]]
    also_reported_by: list[str]
    also_reported_urls: list[str]
    engine: str
    body: str


class GetArticleOutput(_Out):
    found: bool
    article: Article | None = None


def get_article(ctx: ToolContext, args: GetArticleInput) -> GetArticleOutput:
    """The full stored article and its metadata."""
    row = ctx.store.item(args.item_id)
    if row is None:
        return GetArticleOutput(found=False)
    return GetArticleOutput(
        found=True,
        article=Article(
            item_id=row["item_id"],
            title=row["title"],
            url=row["url"],
            source=row["source"],
            region=row["region"],
            category=row["category"],
            published_date=datetime.fromtimestamp(row["published_ts"], tz=UTC).date().isoformat(),
            summary=row["summary"],
            key_facts=json.loads(row["key_facts"]),
            entities=json.loads(row["entities"]),
            also_reported_by=json.loads(row["also_reported_by"]),
            also_reported_urls=json.loads(row["also_reported_urls"]),
            engine=row["engine"],
            body=row["body"],
        ),
    )


# ---------- list_sources ----------


class ListSourcesInput(_In):
    region: str | None = None
    category: str | None = None


class SourceInfo(_Out):
    name: str
    type: str
    region: str
    category: str
    enabled: bool
    location: str
    last_status: str  # ok | error | unknown


class ListSourcesOutput(_Out):
    sources: list[SourceInfo]


def list_sources(ctx: ToolContext, args: ListSourcesInput) -> ListSourcesOutput:
    """Configured sources and their status in the last run."""
    last = ctx.store.last_run_summary()
    failed = set(last.get("failed_sources", [])) if last else set()
    out = []
    for s in ctx.cfg.sources:
        if args.region and s.region != args.region:
            continue
        if args.category and s.category != args.category:
            continue
        status = "unknown" if not last or not s.enabled else ("error" if s.name in failed else "ok")
        out.append(
            SourceInfo(
                name=s.name,
                type=s.type,
                region=s.region,
                category=s.category,
                enabled=s.enabled,
                location=s.url or s.query or s.api or "",
                last_status=status,
            )
        )
    return ListSourcesOutput(sources=out)


# ---------- stats ----------


class StatsInput(_In):
    date_from: date | None = None
    date_to: date | None = None


class StatsOutput(_Out):
    total: int
    by_region: dict[str, int]
    by_category: dict[str, int]
    by_source: dict[str, int]
    by_engine: dict[str, int]


def _ts(d: date | None, end: bool) -> int | None:
    if d is None:
        return None
    return int(datetime.combine(d, time.max if end else time.min, tzinfo=UTC).timestamp())


def stats(ctx: ToolContext, args: StatsInput) -> StatsOutput:
    """Counts of stored articles by region, category, source and engine. Computed in SQL."""
    counts = ctx.store.counts(_ts(args.date_from, False), _ts(args.date_to, True))
    return StatsOutput(**counts)


# ---------- compare_periods ----------


class Period(_In):
    date_from: date
    date_to: date


class ComparePeriodsInput(_In):
    query: str = Field(min_length=1)
    period_a: Period
    period_b: Period
    regions: list[str] | None = None
    categories: list[str] | None = None


class ComparePeriodsOutput(_Out):
    period_a: SearchResult
    period_b: SearchResult
    count_a: int
    count_b: int


def compare_periods(ctx: ToolContext, args: ComparePeriodsInput) -> ComparePeriodsOutput:
    """Run the same search over two date ranges and return both result sets."""

    def one(p: Period) -> SearchResult:
        return search_news(
            ctx,
            SearchNewsInput(
                query=args.query,
                regions=args.regions,
                categories=args.categories,
                date_from=p.date_from,
                date_to=p.date_to,
            ),
        )

    a, b = one(args.period_a), one(args.period_b)
    return ComparePeriodsOutput(period_a=a, period_b=b, count_a=len(a.hits), count_b=len(b.hits))


# ---------- topic_brief ----------


class TopicBriefInput(_In):
    topic: str = Field(min_length=1, description="Topic or keywords, e.g. 'RBI interest rates'")
    regions: list[str] | None = None
    categories: list[str] | None = None
    days: int = Field(30, ge=1, le=365, description="How many days back to look")
    max_articles: int = Field(25, ge=3, le=50)


async def topic_brief(ctx: ToolContext, args: TopicBriefInput) -> TopicBrief:
    """Structured brief of stored news on a topic (overview, timeline, regions, key numbers)."""
    return await build_brief(
        ctx,
        args.topic,
        regions=args.regions,
        categories=args.categories,
        days=args.days,
        max_articles=args.max_articles,
    )


# ---------- fetch_now (changes data) ----------


class FetchNowInput(_In):
    regions: list[str] | None = None
    limit: int = Field(0, ge=0, description="Process at most N items (0 = all)")


async def fetch_now(ctx: ToolContext, args: FetchNowInput) -> RunOutcome:
    """Fetch, process and store new articles (incremental; duplicates are skipped)."""
    return await run_ingest(
        ctx.store,
        ctx.cfg,
        ctx.settings,
        ctx.keys,
        ctx.embedder,
        regions=args.regions,
        limit=args.limit,
    )


# ---------- cleanup (changes data) ----------


class CleanupInput(_In):
    retention_days: int = Field(ge=1)
    dry_run: bool = True


class CleanupOutput(_Out):
    cutoff: str
    items: int
    chunks: int
    applied: bool


def cleanup(ctx: ToolContext, args: CleanupInput) -> CleanupOutput:
    """Preview (default) or apply removal of articles older than retention_days."""
    r = ctx.store.cleanup(args.retention_days, now=datetime.now(UTC), apply=not args.dry_run)
    return CleanupOutput(
        cutoff=r.cutoff.date().isoformat(), items=r.items, chunks=r.chunks, applied=r.applied
    )
