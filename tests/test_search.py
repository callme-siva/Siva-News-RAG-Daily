from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from newsrag.config import load_config
from newsrag.engines import RuleEngine
from newsrag.ingest import ingest
from newsrag.search import SearchFilters, rrf, search
from newsrag.secrets import KeyStore
from newsrag.settings import RetrievalSettings, Settings
from newsrag.store import Store, fts_query
from tests.fakes import BrokenReranker, FakeEmbedder, FakeReranker
from tests.helpers import NOW, item

CFG = load_config()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    s = Store(tmp_path / "newsrag.db")
    items = [
        item(
            "Central bank raises interest rates to curb inflation",
            body="The central bank raised rates by 25 basis points. Inflation stayed high.",
            region="IN",
            hours_ago=2,
        ),
        item(
            "Federal Reserve holds interest rates steady",
            body="The Federal Reserve kept rates unchanged and signalled patience on inflation.",
            region="US",
            hours_ago=30,
        ),
        item(
            "Chipmaker FXCHIP shares jump after earnings beat",
            body="Shares of FXCHIP rose 12% after quarterly earnings beat forecasts.",
            region="US",
            category="Technology",
            hours_ago=5,
        ),
        item(
            "European parliament debates energy prices",
            body="Lawmakers debated measures to lower household energy bills across Europe.",
            region="EU",
            category="Politics",
            hours_ago=24 * 10,
        ),
    ]
    settings = Settings()
    settings.sources.dedupe_window_days = 0
    asyncio.run(ingest(items, s, RuleEngine(CFG), FakeEmbedder(), settings, KeyStore(), now=NOW))
    yield s
    s.close()


def _settings(**kw: object) -> RetrievalSettings:
    return RetrievalSettings.model_validate(kw)


def test_fts_query_is_safe() -> None:
    # Every token is quoted, so FTS operators and punctuation in user text are inert.
    assert fts_query('rates" OR 1=1; DROP') == '"rates" OR "1" OR "DROP"'
    assert fts_query("!!!") is None
    assert fts_query("6.75% repo") == '"6.75%" OR "repo"'


def test_rrf_rewards_agreement() -> None:
    scores = rrf({"keyword": ["a", "b"], "semantic": ["b", "c"]}, {})
    assert max(scores, key=scores.__getitem__) == "b"


def test_exact_term_found_in_hybrid(store: Store) -> None:
    """Acceptance 14: a ticker that appears in only one article is found."""
    res = search(store, "FXCHIP", SearchFilters(), _settings(), FakeEmbedder(), FakeReranker())
    assert res.hits and res.hits[0].title.startswith("Chipmaker FXCHIP")
    assert "keyword" in res.hits[0].found_by


def test_rerank_scores_shown_and_order_follows_rerank(store: Store) -> None:
    res = search(
        store,
        "interest rates inflation",
        SearchFilters(),
        _settings(),
        FakeEmbedder(),
        FakeReranker(),
    )
    assert res.reranked and all(h.rerank_score is not None for h in res.hits)
    scores = [h.rerank_score or 0 for h in res.hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("reranker", [None, BrokenReranker()])
def test_search_works_without_or_with_broken_reranker(store: Store, reranker: object) -> None:
    res = search(
        store,
        "interest rates",
        SearchFilters(),
        _settings(),
        FakeEmbedder(),
        reranker,  # type: ignore[arg-type]
    )
    assert res.hits and not res.reranked and res.notes
    assert all(h.rerank_score is None for h in res.hits)


def test_rerank_off_has_no_note(store: Store) -> None:
    res = search(
        store, "interest rates", SearchFilters(), _settings(rerank=False), FakeEmbedder(), None
    )
    assert res.hits and res.notes == []


def test_region_and_date_filters_are_applied_in_the_query(store: Store) -> None:
    """Acceptance 5: only matching items come back."""
    res = search(
        store,
        "interest rates",
        SearchFilters(regions=["US"]),
        _settings(),
        FakeEmbedder(),
        FakeReranker(),
    )
    assert res.hits and {h.region for h in res.hits} == {"US"}

    recent = SearchFilters(date_from=(NOW - timedelta(days=1)).date())
    res = search(store, "energy prices parliament", recent, _settings(), FakeEmbedder(), None)
    assert all(h.published_date >= recent.date_from.isoformat() for h in res.hits)  # type: ignore[union-attr]
    assert "European parliament debates energy prices" not in {h.title for h in res.hits}


@pytest.mark.parametrize("mode", ["keyword", "semantic"])
def test_single_modes(store: Store, mode: str) -> None:
    res = search(
        store,
        "interest rates",
        SearchFilters(),
        _settings(search_mode=mode),
        FakeEmbedder(),
        None,
    )
    assert res.hits and all(h.found_by == [mode] for h in res.hits)


def test_no_embedder_falls_back_to_keyword(store: Store) -> None:
    res = search(store, "interest rates", SearchFilters(), _settings(), None, None)
    assert res.hits and any("semantic search unavailable" in n for n in res.notes)


def test_group_by_article_and_top_k(tmp_path: Path) -> None:
    s = Store(tmp_path / "g.db")
    long_body = " ".join(f"Interest rates sentence {i} about policy." for i in range(120))
    asyncio.run(
        ingest(
            [item("Interest rates explainer", body=long_body)],
            s,
            RuleEngine(CFG),
            FakeEmbedder(),
            Settings(),
            KeyStore(),
            now=NOW,
        )
    )
    res = search(
        s,
        "interest rates policy",
        SearchFilters(),
        _settings(max_chunks_per_article=2, top_k=8),
        FakeEmbedder(),
        None,
    )
    assert len(res.hits) == 2 and len({h.item_id for h in res.hits}) == 1
    s.close()


def test_min_score_drops_weak_reranked_hits(store: Store) -> None:
    res = search(
        store,
        "interest rates inflation",
        SearchFilters(),
        _settings(min_score=0.99),
        FakeEmbedder(),
        FakeReranker(),
    )
    assert all((h.rerank_score or 0) >= 0.99 for h in res.hits)


def test_relevance_floor_drops_unrelated_semantic_matches(store: Store) -> None:
    """Vector search always returns neighbours; unrelated ones must not reach the reader."""
    res = search(store, "zzqx unknownterm", SearchFilters(), _settings(), FakeEmbedder(), None)
    assert res.hits == [] and any("relevant enough" in n for n in res.notes)
    loose = search(
        store,
        "zzqx unknownterm",
        SearchFilters(),
        _settings(min_similarity=0.0, rerank=False),
        FakeEmbedder(),
        None,
    )
    assert loose.hits  # with the floor off, neighbours come back
    ok = search(store, "interest rates", SearchFilters(), _settings(), FakeEmbedder(), None)
    assert ok.hits and all(
        "keyword" in h.found_by or (h.semantic_score or 0) >= 0.30 for h in ok.hits
    )


def test_question_words_do_not_match_everything(store: Store) -> None:
    """Regression: 'what/did/the/on' joined with OR matched almost every article live."""
    assert fts_query("What did the RBI decide on interest rates?") == (
        '"RBI" OR "decide" OR "interest" OR "rates"'
    )
    assert fts_query("Tell me the latest news") is None
    assert fts_query("AI chips EU") == '"AI" OR "chips" OR "EU"'
    res = search(
        store, "What did the chipmaker do?", SearchFilters(), _settings(), FakeEmbedder(), None
    )
    assert all("Chipmaker" in h.title or (h.semantic_score or 0) >= 0.30 for h in res.hits)
