"""Workspace storage: one SQLite file with items, chunks, FTS5 index and vectors."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import zipfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from newsrag.embeddings import Vectors, from_blob, to_blob
from newsrag.models import Chunk, Entities, Item, Processed
from newsrag.store.schema import SCHEMA, SCHEMA_VERSION

_FTS_TOKEN = re.compile(r"[\w][\w.%$'-]*", re.UNICODE)

# Question words and function words. Joined with OR they would match almost every article,
# which also defeats the relevance floor (a keyword match counts as relevant).
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its
    itself just me more most my myself no nor not now of off on once only or other our ours out
    over own same she should so some such than that the their theirs them then there these they
    this those through to too under until up very was we were what when where which while who
    whom why will with would you your yours tell show give latest news today week happened
    happen anything something""".split()
)


class StoreError(Exception):
    pass


@dataclass
class Filters:
    regions: list[str] | None = None
    categories: list[str] | None = None
    from_ts: int | None = None
    to_ts: int | None = None

    def sql(self, alias: str = "c") -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if self.regions:
            clauses.append(f"{alias}.region IN ({','.join('?' * len(self.regions))})")
            params.extend(self.regions)
        if self.categories:
            clauses.append(f"{alias}.category IN ({','.join('?' * len(self.categories))})")
            params.extend(self.categories)
        if self.from_ts is not None:
            clauses.append(f"{alias}.published_ts >= ?")
            params.append(self.from_ts)
        if self.to_ts is not None:
            clauses.append(f"{alias}.published_ts <= ?")
            params.append(self.to_ts)
        return (" AND ".join(clauses) or "1=1"), params


@dataclass
class CleanupResult:
    cutoff: datetime
    items: int
    chunks: int
    applied: bool


@dataclass
class IntegrityReport:
    orphan_fts: int = 0
    orphan_chunks: int = 0
    missing_vectors: int = 0
    missing_fts: int = 0
    repaired: bool = False
    details: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (
            self.orphan_fts or self.orphan_chunks or self.missing_vectors or self.missing_fts
        )


def fts_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 query: each token quoted, joined with OR."""
    tokens = [t.strip(".'-") for t in _FTS_TOKEN.findall(text)]
    tokens = [t for t in tokens if t and t.lower() not in STOPWORDS]
    if not tokens:
        return None
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in dict.fromkeys(tokens))


def _now_ts() -> int:
    return int(datetime.now(UTC).timestamp())


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._conn.execute("PRAGMA foreign_keys = ON")
        version = self._meta("schema_version")
        if version is None:
            with self._conn:
                self._set_meta("schema_version", str(SCHEMA_VERSION))
        elif int(version) > SCHEMA_VERSION:
            raise StoreError(f"Database schema {version} is newer than this app")

    # ---------- lifecycle ----------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._conn:
            yield self._conn

    def _meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # ---------- duplicate checks (DD2, DD3, DD5) ----------

    def seen_urls(self, urls: Iterable[str]) -> set[str]:
        urls = list(dict.fromkeys(urls))
        found: set[str] = set()
        for i in range(0, len(urls), 500):
            batch = urls[i : i + 500]
            rows = self._conn.execute(
                f"SELECT url FROM seen_urls WHERE url IN ({','.join('?' * len(batch))})", batch
            )
            found.update(r["url"] for r in rows)
        return found

    def content_hash(self, item_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT content_hash FROM items WHERE item_id = ?", (item_id,)
        ).fetchone()
        return None if row is None else str(row["content_hash"])

    def recent_titles(self, category: str, since_ts: int) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT item_id, title FROM items WHERE category = ? AND published_ts >= ?",
            (category, since_ts),
        )
        return [(r["item_id"], r["title"]) for r in rows]

    def mark_seen(self, urls: Iterable[str], item_id: str | None, outcome: str) -> None:
        now = _now_ts()
        with self._conn:
            self._conn.executemany(
                "INSERT INTO seen_urls (url, item_id, first_seen_ts, outcome) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (url) DO NOTHING",
                [(u, item_id, now, outcome) for u in dict.fromkeys(urls)],
            )

    def attach_source(self, item_id: str, source: str, url: str) -> None:
        """Record another outlet for an existing item (cross-day near-duplicate)."""
        row = self._conn.execute(
            "SELECT source, also_reported_by, also_reported_urls, url FROM items WHERE item_id = ?",
            (item_id,),
        ).fetchone()
        if row is None:
            raise StoreError(f"unknown item {item_id}")
        sources = json.loads(row["also_reported_by"])
        urls = json.loads(row["also_reported_urls"])
        if source != row["source"] and source not in sources:
            sources.append(source)
        if url != row["url"] and url not in urls:
            urls.append(url)
        with self._conn:
            self._conn.execute(
                "UPDATE items SET also_reported_by = ?, also_reported_urls = ? WHERE item_id = ?",
                (json.dumps(sorted(sources)), json.dumps(sorted(urls)), item_id),
            )
            self._conn.execute(
                "INSERT INTO seen_urls (url, item_id, first_seen_ts, outcome) "
                "VALUES (?, ?, ?, 'merged') ON CONFLICT (url) DO NOTHING",
                (url, item_id, _now_ts()),
            )

    # ---------- writes (DD4) ----------

    def write_item(
        self,
        item: Item,
        processed: Processed,
        chunks: list[Chunk],
        vectors: Vectors,
        *,
        replace: bool = False,
    ) -> bool:
        """Store an item with its chunks, keyword rows and vectors in ONE transaction.
        Returns False if the item already exists and `replace` is False."""
        if len(chunks) != len(vectors):
            raise StoreError("chunks and vectors differ in length")
        with self._conn:
            exists = self._conn.execute(
                "SELECT 1 FROM items WHERE item_id = ?", (item.item_id,)
            ).fetchone()
            if exists and not replace:
                return False
            if exists:
                self._delete_items([item.item_id])
            self._insert_item(item, processed)
            self._insert_chunks(chunks)
            self._insert_fts(chunks)
            self._insert_vectors(chunks, vectors)
            urls = [item.url, *item.also_reported_urls]
            self._conn.executemany(
                "INSERT INTO seen_urls (url, item_id, first_seen_ts, outcome) "
                "VALUES (?, ?, ?, 'stored') ON CONFLICT (url) DO NOTHING",
                [(u, item.item_id, _now_ts()) for u in urls],
            )
        return True

    def _insert_item(self, item: Item, p: Processed) -> None:
        self._conn.execute(
            """INSERT INTO items (item_id, url, kind, title, body, source, region, category,
                published_ts, fetched_via, also_reported_by, also_reported_urls, tags,
                content_hash, summary, key_facts, entities, engine, fallback_reason, ingested_ts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                item.item_id,
                item.url,
                item.kind,
                item.title,
                item.body,
                item.source,
                item.region,
                p.category,
                int(item.published_at.timestamp()),
                item.fetched_via.value,
                json.dumps(list(item.also_reported_by)),
                json.dumps(list(item.also_reported_urls)),
                json.dumps(item.tags),
                item.content_hash,
                p.summary,
                json.dumps(p.key_facts),
                p.entities.model_dump_json(),
                p.engine.value,
                p.fallback_reason,
                _now_ts(),
            ),
        )

    def _insert_chunks(self, chunks: list[Chunk]) -> None:
        self._conn.executemany(
            """INSERT INTO chunks (chunk_id, item_id, chunk_index, text, region, category,
                source, published_ts) VALUES (?,?,?,?,?,?,?,?)""",
            [
                (
                    c.chunk_id,
                    c.item_id,
                    c.chunk_index,
                    c.text,
                    c.metadata["region"],
                    c.metadata["category"],
                    c.metadata["source"],
                    c.metadata["published_ts"],
                )
                for c in chunks
            ],
        )

    def _insert_fts(self, chunks: list[Chunk]) -> None:
        self._conn.executemany(
            "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
            [(c.chunk_id, c.text) for c in chunks],
        )

    def _insert_vectors(self, chunks: list[Chunk], vectors: Vectors) -> None:
        self._conn.executemany(
            "INSERT INTO embeddings (chunk_id, dim, vector) VALUES (?, ?, ?)",
            [
                (c.chunk_id, int(v.shape[0]), to_blob(v))
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )

    def _delete_items(self, item_ids: list[str]) -> int:
        """Delete items and everything hanging off them. Caller holds the transaction."""
        deleted = 0
        for i in range(0, len(item_ids), 500):
            batch = item_ids[i : i + 500]
            marks = ",".join("?" * len(batch))
            self._conn.execute(
                f"DELETE FROM chunks_fts WHERE chunk_id IN "
                f"(SELECT chunk_id FROM chunks WHERE item_id IN ({marks}))",
                batch,
            )
            cur = self._conn.execute(f"DELETE FROM items WHERE item_id IN ({marks})", batch)
            deleted += cur.rowcount
        return deleted

    def iter_stored(self) -> Iterator[tuple[Item, Processed]]:
        """Rebuild Item and Processed objects from stored rows (used by reindex)."""
        rows = self._conn.execute("SELECT * FROM items ORDER BY published_ts").fetchall()
        for r in rows:
            item = Item(
                title=r["title"],
                body=r["body"],
                url=r["url"],
                source=r["source"],
                region=r["region"],
                category=r["category"],
                published_at=datetime.fromtimestamp(r["published_ts"], tz=UTC),
                fetched_via=r["fetched_via"],
                kind=r["kind"],
                also_reported_by=tuple(json.loads(r["also_reported_by"])),
                also_reported_urls=tuple(json.loads(r["also_reported_urls"])),
                tags=json.loads(r["tags"]),
            )
            processed = Processed(
                item_id=r["item_id"],
                relevant=True,
                category=r["category"],
                summary=r["summary"],
                key_facts=json.loads(r["key_facts"]),
                entities=Entities.model_validate_json(r["entities"]),
                engine=r["engine"],
                fallback_reason=r["fallback_reason"],
            )
            yield item, processed

    def replace_chunks(self, item_id: str, chunks: list[Chunk], vectors: Vectors) -> None:
        """Swap an item's chunks, keyword rows and vectors in one transaction."""
        if len(chunks) != len(vectors):
            raise StoreError("chunks and vectors differ in length")
        with self._conn:
            self._conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE item_id = ?)",
                (item_id,),
            )
            self._conn.execute("DELETE FROM chunks WHERE item_id = ?", (item_id,))
            self._insert_chunks(chunks)
            self._insert_fts(chunks)
            self._insert_vectors(chunks, vectors)

    def upsert_series(self, series: str, date: str, source: str, value: float, unit: str) -> None:
        """Numeric series with natural key (series, date, source); corrections replace (DD6)."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO series (series, date, source, value, unit) VALUES (?,?,?,?,?) "
                "ON CONFLICT (series, date, source) DO UPDATE SET value = excluded.value, "
                "unit = excluded.unit",
                (series, date, source, value, unit),
            )

    # ---------- runs ----------

    def record_run(self, started: datetime, finished: datetime, summary: dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO runs (started_ts, finished_ts, summary) VALUES (?, ?, ?)",
                (int(started.timestamp()), int(finished.timestamp()), json.dumps(summary)),
            )

    def last_run(self) -> datetime | None:
        row = self._conn.execute("SELECT MAX(finished_ts) AS ts FROM runs").fetchone()
        return None if row["ts"] is None else datetime.fromtimestamp(row["ts"], tz=UTC)

    def last_run_summary(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT summary FROM runs ORDER BY finished_ts DESC, run_id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else dict(json.loads(row["summary"]))

    def counts(self, from_ts: int | None, to_ts: int | None) -> dict[str, Any]:
        where, params = Filters(from_ts=from_ts, to_ts=to_ts).sql("i")

        def group(col: str) -> dict[str, int]:
            rows = self._conn.execute(
                f"SELECT {col} AS k, COUNT(*) AS n FROM items i WHERE {where} "
                f"GROUP BY {col} ORDER BY n DESC, k",
                params,
            )
            return {r["k"]: r["n"] for r in rows}

        total = self._conn.execute(f"SELECT COUNT(*) FROM items i WHERE {where}", params)
        return {
            "total": total.fetchone()[0],
            "by_region": group("region"),
            "by_category": group("category"),
            "by_source": group("source"),
            "by_engine": group("engine"),
        }

    def recent_items(self, from_ts: int, to_ts: int) -> list[dict[str, Any]]:
        """Stored items published in [from_ts, to_ts], newest first (for the digest)."""
        rows = self._conn.execute(
            """SELECT item_id, title, url, source, region, category, published_ts, summary,
                      key_facts, also_reported_by, engine
               FROM items WHERE published_ts BETWEEN ? AND ? ORDER BY published_ts DESC""",
            (from_ts, to_ts),
        )
        return [dict(r) for r in rows]

    # ---------- reads for search ----------

    def keyword_candidates(self, query: str, filters: Filters, limit: int) -> list[str]:
        match = fts_query(query)
        if match is None:
            return []
        where, params = filters.sql("c")
        rows = self._conn.execute(
            f"""SELECT f.chunk_id FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.chunk_id
                WHERE chunks_fts MATCH ? AND {where} ORDER BY bm25(chunks_fts) LIMIT ?""",
            [match, *params, limit],
        )
        return [r["chunk_id"] for r in rows]

    def vectors(self, filters: Filters) -> tuple[list[str], Vectors]:
        where, params = filters.sql("c")
        rows = self._conn.execute(
            f"""SELECT e.chunk_id, e.vector FROM embeddings e JOIN chunks c
                ON c.chunk_id = e.chunk_id WHERE {where}""",
            params,
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        return [r["chunk_id"] for r in rows], np.vstack([from_blob(r["vector"]) for r in rows])

    def chunk_details(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for i in range(0, len(chunk_ids), 500):
            batch = chunk_ids[i : i + 500]
            rows = self._conn.execute(
                f"""SELECT c.chunk_id, c.item_id, c.text, i.title, i.url, i.source, i.region,
                    i.category, i.published_ts, i.summary, i.engine
                    FROM chunks c JOIN items i ON i.item_id = c.item_id
                    WHERE c.chunk_id IN ({",".join("?" * len(batch))})""",
                batch,
            )
            out.update({r["chunk_id"]: dict(r) for r in rows})
        return out

    def item(self, item_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM items WHERE item_id = ?", (item_id,)).fetchone()
        return None if row is None else dict(row)

    # ---------- maintenance (FR38-FR43) ----------

    def stats(self) -> dict[str, Any]:
        q = self._conn.execute
        by_group = q(
            "SELECT region, category, COUNT(*) AS n FROM items GROUP BY region, category"
        ).fetchall()
        span = q("SELECT MIN(published_ts) AS lo, MAX(published_ts) AS hi FROM items").fetchone()
        engines = q("SELECT engine, COUNT(*) AS n FROM items GROUP BY engine").fetchall()

        def iso(ts: int | None) -> str | None:
            return None if ts is None else datetime.fromtimestamp(ts, tz=UTC).date().isoformat()

        return {
            "items": q("SELECT COUNT(*) FROM items").fetchone()[0],
            "chunks": q("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "seen_urls": q("SELECT COUNT(*) FROM seen_urls").fetchone()[0],
            "runs": q("SELECT COUNT(*) FROM runs").fetchone()[0],
            "by_region_category": {f"{r['region']}/{r['category']}": r["n"] for r in by_group},
            "by_engine": {r["engine"]: r["n"] for r in engines},
            "oldest": iso(span["lo"]),
            "newest": iso(span["hi"]),
            "db_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }

    def cleanup(self, retention_days: int, *, now: datetime, apply: bool) -> CleanupResult:
        """Remove items published before now - retention_days, with their chunks, keyword
        rows and vectors. Seen URLs are kept so old articles are not fetched again."""
        cutoff = now - timedelta(days=retention_days)
        ts = int(cutoff.timestamp())
        ids = [
            r["item_id"]
            for r in self._conn.execute("SELECT item_id FROM items WHERE published_ts < ?", (ts,))
        ]
        chunk_count = 0
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            chunk_count += self._conn.execute(
                f"SELECT COUNT(*) FROM chunks WHERE item_id IN ({','.join('?' * len(batch))})",
                batch,
            ).fetchone()[0]
        if apply and ids:
            with self._conn:
                self._delete_items(ids)
        return CleanupResult(cutoff=cutoff, items=len(ids), chunks=chunk_count, applied=apply)

    def verify(self, *, repair: bool) -> IntegrityReport:
        """Find (and optionally fix) rows that disagree between the tables."""
        q = self._conn.execute
        rep = IntegrityReport()
        rep.orphan_fts = q(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunk_id NOT IN (SELECT chunk_id FROM chunks)"
        ).fetchone()[0]
        rep.orphan_chunks = q(
            "SELECT COUNT(*) FROM chunks WHERE item_id NOT IN (SELECT item_id FROM items)"
        ).fetchone()[0]
        rep.missing_vectors = q(
            "SELECT COUNT(*) FROM chunks WHERE chunk_id NOT IN (SELECT chunk_id FROM embeddings)"
        ).fetchone()[0]
        rep.missing_fts = q(
            "SELECT COUNT(*) FROM chunks WHERE chunk_id NOT IN (SELECT chunk_id FROM chunks_fts)"
        ).fetchone()[0]
        if repair and not rep.clean:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id NOT IN (SELECT chunk_id FROM chunks)"
                )
                self._conn.execute(
                    "DELETE FROM chunks WHERE item_id NOT IN (SELECT item_id FROM items)"
                )
                broken = [
                    r[0]
                    for r in q(
                        """SELECT DISTINCT item_id FROM chunks WHERE chunk_id NOT IN
                           (SELECT chunk_id FROM embeddings) OR chunk_id NOT IN
                           (SELECT chunk_id FROM chunks_fts)"""
                    )
                ]
                if broken:
                    self._delete_items(broken)
                    self._conn.executemany(
                        "DELETE FROM seen_urls WHERE item_id = ?", [(b,) for b in broken]
                    )
                    rep.details.append(
                        f"removed {len(broken)} incomplete items; they will be fetched again"
                    )
            rep.repaired = True
        return rep

    def delete_where(
        self,
        *,
        sources: list[str] | None = None,
        regions: list[str] | None = None,
        categories: list[str] | None = None,
        from_ts: int | None = None,
        to_ts: int | None = None,
        apply: bool,
    ) -> int:
        """Preview (apply=False) or delete items matching all given conditions (FR41).
        At least one condition is required. Seen URLs are kept."""
        if not (sources or regions or categories or from_ts is not None or to_ts is not None):
            raise StoreError("give at least one condition")
        where, params = Filters(regions, categories, from_ts, to_ts).sql("i")
        if sources:
            where += f" AND i.source IN ({','.join('?' * len(sources))})"
            params = [*params, *sources]
        ids = [
            r[0] for r in self._conn.execute(f"SELECT item_id FROM items i WHERE {where}", params)
        ]
        if apply and ids:
            with self._conn:
                self._delete_items(ids)
        return len(ids)

    def sources_in_store(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT DISTINCT source FROM items ORDER BY 1")]

    def runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT run_id, started_ts, finished_ts, summary FROM runs "
            "ORDER BY finished_ts DESC, run_id DESC LIMIT ?",
            (limit,),
        )
        return [
            {
                "run_id": r["run_id"],
                "started": datetime.fromtimestamp(r["started_ts"], tz=UTC),
                "finished": datetime.fromtimestamp(r["finished_ts"], tz=UTC),
                **json.loads(r["summary"]),
            }
            for r in rows
        ]

    def compact(self) -> None:
        self._conn.execute("VACUUM")


def backup_workspace(root: Path, backups_dir: Path, store: Store | None = None) -> Path:
    """Zip the workspace (excluding the backups folder) into backups/newsrag-<time>.zip."""
    backups_dir.mkdir(parents=True, exist_ok=True)
    if store is not None:
        store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    target = backups_dir / f"newsrag-{stamp}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if path.is_file() and backups_dir not in path.parents and path != target:
                zf.write(path, path.relative_to(root))
    return target


def restore_workspace(archive: Path, root: Path) -> None:
    """Replace the workspace contents (except backups) with the archive's contents."""
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        if "workspace.json" not in names:
            raise StoreError("not a newsrag workspace backup")
        for child in root.iterdir():
            if child.name == "backups":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        zf.extractall(root)
