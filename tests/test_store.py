from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from newsrag import ids
from newsrag.config import load_config
from newsrag.engines import RuleEngine
from newsrag.ingest import IngestReport, ingest, reindex
from newsrag.models import Item
from newsrag.pipeline.chunk import build_chunks
from newsrag.secrets import KeyStore
from newsrag.settings import Settings
from newsrag.store import Store, backup_workspace, restore_workspace
from newsrag.workspace import open_workspace
from tests.fakes import FakeEmbedder
from tests.helpers import NOW, item

CFG = load_config()
BODY = (
    "The central bank raised its policy rate by 25 basis points to 6.75% on Wednesday. "
    "Officials said inflation remained above target. Bond yields rose 8 basis points."
)


def _items() -> list[Item]:
    return [
        item("Central bank raises policy interest rates", body=BODY, source="Outlet A"),
        item(
            "Chipmaker posts record quarterly revenue on data centre demand",
            body="Revenue rose 40% to $30 billion as AI chip sales grew.",
            source="Outlet B",
            category="Technology",
            region="US",
        ),
        item(
            "Parliament passes new data protection bill",
            body="The bill passed with a majority of 300 votes after a long debate.",
            source="Outlet C",
            category="Politics",
        ),
    ]


def _ingest(store: Store, items: list[Item], settings: Settings | None = None) -> IngestReport:
    return asyncio.run(
        ingest(
            items,
            store,
            RuleEngine(CFG),
            FakeEmbedder(),
            settings or Settings(),
            KeyStore(),
            now=NOW,
        )
    )


def _counts(store: Store) -> tuple[int, int, int, int]:
    q = store._conn.execute
    return (
        q("SELECT COUNT(*) FROM items").fetchone()[0],
        q("SELECT COUNT(*) FROM chunks").fetchone()[0],
        q("SELECT COUNT(*) FROM chunks_fts").fetchone()[0],
        q("SELECT COUNT(*) FROM embeddings").fetchone()[0],
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    s = Store(tmp_path / "newsrag.db")
    yield s
    s.close()


def test_ingest_twice_changes_nothing(store: Store) -> None:
    """Acceptance 15a: same fixtures twice -> counts unchanged, all 'skipped as seen'."""
    first = _ingest(store, _items())
    assert first.new == 3 and first.by_engine == {"rules": 3}
    before = _counts(store)
    assert before[0] == 3 and before[1] == before[2] == before[3] >= 3

    second = _ingest(store, _items())
    assert second.new == 0 and second.skipped_seen == 3
    assert _counts(store) == before


def test_url_variants_are_one_item(store: Store) -> None:
    """Acceptance 15b."""
    a = item("Central bank raises rates", url="https://www.example.com/x?utm_source=a", body=BODY)
    b = item("Central bank raises rates", url="http://example.com/x/#top", body=BODY)
    assert _ingest(store, [a]).new == 1
    assert _ingest(store, [b]).skipped_seen == 1
    assert _counts(store)[0] == 1


def test_same_story_next_day_from_another_outlet_is_merged(store: Store) -> None:
    """Acceptance 15c: one item with both sources listed."""
    day1 = item(
        "Central bank raises policy interest rates sharply",
        body=BODY,
        source="Outlet A",
        hours_ago=26,
    )
    day2 = item(
        "Central bank raises policy interest rates again",
        url="https://other.example.org/rates",
        source="Outlet Z",
        hours_ago=1,
    )
    _ingest(store, [day1])
    report = _ingest(store, [day2])
    assert report.merged_duplicate == 1 and report.new == 0
    stored = store.item(day1.item_id)
    assert stored is not None
    assert '"Outlet Z"' in stored["also_reported_by"]
    assert "other.example.org/rates" in stored["also_reported_urls"]
    assert store.seen_urls([day2.url]) == {day2.url}


def test_crash_during_vector_write_leaves_nothing_then_rerun_is_clean(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 15d: a failure mid-write rolls back the whole item."""
    original = Store._insert_vectors
    calls = {"n": 0}

    def flaky(self: Store, chunks, vectors) -> None:  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full (simulated)")
        original(self, chunks, vectors)

    monkeypatch.setattr(Store, "_insert_vectors", flaky)
    report = _ingest(store, _items())
    assert report.new == 2 and report.failed == 1
    assert "disk full" in report.errors[0]
    rep = store.verify(repair=False)
    assert rep.clean

    monkeypatch.setattr(Store, "_insert_vectors", original)
    rerun = _ingest(store, _items())
    assert rerun.new == 1 and rerun.skipped_seen == 2
    assert _counts(store)[0] == 3 and store.verify(repair=False).clean


def test_series_upsert_replaces_correction(store: Store) -> None:
    """Acceptance 15e: (series, date, source) is the natural key."""
    store.upsert_series("gold_inr_10g", "2026-10-07", "fixture-source", 75000.0, "INR")
    store.upsert_series("gold_inr_10g", "2026-10-07", "fixture-source", 75120.0, "INR")
    store.upsert_series("gold_inr_10g", "2026-10-08", "fixture-source", 75300.0, "INR")
    rows = store._conn.execute("SELECT date, value FROM series ORDER BY date").fetchall()
    assert [tuple(r) for r in rows] == [("2026-10-07", 75120.0), ("2026-10-08", 75300.0)]


def test_irrelevant_items_are_marked_seen_not_stored(store: Store) -> None:
    junk = item("Celebrity wedding photos gallery", body="Gowns and guests.")
    report = _ingest(store, [junk])
    assert report.irrelevant == 1 and _counts(store)[0] == 0
    assert _ingest(store, [junk]).skipped_seen == 1


def test_on_update_replace(store: Store) -> None:
    original = item("Central bank raises policy interest rates", body=BODY)
    _ingest(store, [original])
    edited = original.model_copy(update={"body": BODY + " Update: markets recovered by noon."})
    settings = Settings()
    settings.sources.on_update = "replace"
    assert _ingest(store, [edited]).skipped_seen == 1  # ignore is the default
    report = _ingest(store, [edited], settings)
    assert report.updated == 1
    assert store.content_hash(original.item_id) == edited.content_hash
    assert _counts(store)[0] == 1 and store.verify(repair=False).clean


def test_cleanup_removes_old_items_and_their_rows(store: Store) -> None:
    """Acceptance 11."""
    old = item("Old central bank story about rates", body=BODY, hours_ago=24 * 40)
    new = item("New central bank story about rates today", body=BODY, hours_ago=2)
    settings = Settings()
    settings.sources.dedupe_window_days = 0
    _ingest(store, [old, new], settings)
    preview = store.cleanup(30, now=NOW, apply=False)
    assert preview.items == 1 and preview.chunks >= 1 and _counts(store)[0] == 2
    done = store.cleanup(30, now=NOW, apply=True)
    assert done.items == 1 and _counts(store)[0] == 1
    assert store.item(new.item_id) is not None and store.item(old.item_id) is None
    assert store.verify(repair=False).clean
    assert store.seen_urls([old.url]) == {old.url}  # never fetched again


def test_verify_finds_and_repairs_orphans(store: Store) -> None:
    _ingest(store, _items())
    store._conn.execute("INSERT INTO chunks_fts (chunk_id, text) VALUES ('ghost:0', 'x')")
    victim = store._conn.execute(
        "SELECT c.chunk_id, c.item_id, i.url FROM chunks c JOIN items i USING (item_id) LIMIT 1"
    ).fetchone()
    assert store.seen_urls([victim["url"]]) == {victim["url"]}
    store._conn.execute("DELETE FROM embeddings WHERE chunk_id = ?", (victim["chunk_id"],))
    store._conn.commit()
    rep = store.verify(repair=False)
    assert rep.orphan_fts == 1 and rep.missing_vectors == 1 and not rep.clean
    fixed = store.verify(repair=True)
    assert fixed.repaired and store.verify(repair=False).clean
    assert store.item(victim["item_id"]) is None  # incomplete item removed, refetched later
    assert store.seen_urls([victim["url"]]) == set()  # so it is fetched and stored again


def test_chunks_carry_header_and_respect_size() -> None:
    long_body = " ".join(f"Sentence number {i} about the central bank decision." for i in range(80))
    it = item("Central bank raises rates", body=long_body)
    p = RuleEngine(CFG).process_sync(it)
    chunks = build_chunks(it, p, size=600, overlap=100)
    assert len(chunks) > 3
    assert all(c.text.startswith("Headline: Central bank raises rates") for c in chunks)
    assert all(it.url in c.text for c in chunks)
    assert [c.chunk_id for c in chunks] == [f"{it.item_id}:{i}" for i in range(len(chunks))]
    header_len = len(chunks[0].text.split("\n[Part")[0])
    assert all(len(c.text) <= header_len + 40 + 600 + 100 for c in chunks)


def test_reindex_applies_new_chunk_size_without_refetch(store: Store) -> None:
    long_body = " ".join(f"Sentence {i} about the central bank rate decision." for i in range(60))
    _ingest(store, [item("Central bank raises rates", body=long_body)])
    before = _counts(store)[1]
    settings = Settings()
    settings.retrieval.chunk_size = 400
    settings.retrieval.chunk_overlap = 50
    embedder = FakeEmbedder()
    assert reindex(store, embedder, settings) == 1
    after = _counts(store)
    assert after[1] > before and after[1] == after[2] == after[3]
    assert store.verify(repair=False).clean


def test_reindex_survives_url_normalisation_rule_changes(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: an item stored before a tracking parameter joined the strip list has an
    ID computed from the old URL. Reindex must keep using the stored ID."""
    it = item(
        "Central bank raises rates", url="https://example.com/rates?ref_campaign=x", body=BODY
    )
    _ingest(store, [it])
    stored_id = it.item_id
    monkeypatch.setattr(ids, "TRACKING_PARAMS", ids.TRACKING_PARAMS | {"ref_campaign"})
    assert ids.item_id_for_url(it.url) != stored_id  # the rule change alters the ID
    assert reindex(store, FakeEmbedder(), Settings()) == 1
    assert store.verify(repair=False).clean
    count = store._conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE item_id = ?", (stored_id,)
    ).fetchone()[0]
    assert count >= 1


def test_backup_and_restore(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    with Store(ws.db_path) as store:
        _ingest(store, _items())
        archive = backup_workspace(ws.root, ws.backups_dir, store)
    with Store(ws.db_path) as store:
        store.cleanup(0, now=NOW + timedelta(days=1), apply=True)
        assert _counts(store)[0] == 0
    restore_workspace(archive, ws.root)
    with Store(ws.db_path) as store:
        assert _counts(store)[0] == 3
