"""SQLite schema. One file per workspace holds rows, keyword index and vectors, so every
write for an item is one transaction (REQUIREMENTS DD4, FR12, FR15)."""

SCHEMA_VERSION = 1

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
    item_id            TEXT PRIMARY KEY,
    url                TEXT NOT NULL UNIQUE,
    kind               TEXT NOT NULL,
    title              TEXT NOT NULL,
    body               TEXT NOT NULL,
    source             TEXT NOT NULL,
    region             TEXT NOT NULL,
    category           TEXT NOT NULL,
    published_ts       INTEGER NOT NULL,
    fetched_via        TEXT NOT NULL,
    also_reported_by   TEXT NOT NULL DEFAULT '[]',
    also_reported_urls TEXT NOT NULL DEFAULT '[]',
    tags               TEXT NOT NULL DEFAULT '{}',
    content_hash       TEXT NOT NULL,
    summary            TEXT NOT NULL,
    key_facts          TEXT NOT NULL DEFAULT '[]',
    entities           TEXT NOT NULL DEFAULT '{}',
    engine             TEXT NOT NULL,
    fallback_reason    TEXT,
    ingested_ts        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS items_filter ON items (category, region, published_ts);

CREATE TABLE IF NOT EXISTS seen_urls (
    url           TEXT PRIMARY KEY,
    item_id       TEXT,
    first_seen_ts INTEGER NOT NULL,
    outcome       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    item_id      TEXT NOT NULL REFERENCES items (item_id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    region       TEXT NOT NULL,
    category     TEXT NOT NULL,
    source       TEXT NOT NULL,
    published_ts INTEGER NOT NULL,
    UNIQUE (item_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_filter ON chunks (region, category, published_ts);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (
    chunk_id UNINDEXED,
    text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks (chunk_id) ON DELETE CASCADE,
    dim      INTEGER NOT NULL,
    vector   BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    started_ts  INTEGER NOT NULL,
    finished_ts INTEGER,
    summary     TEXT NOT NULL
);

-- Numeric daily series (gold rates, FX, ...). Not used by news in v1; natural key enforced.
CREATE TABLE IF NOT EXISTS series (
    series TEXT NOT NULL,
    date   TEXT NOT NULL,
    source TEXT NOT NULL,
    value  REAL NOT NULL,
    unit   TEXT NOT NULL,
    PRIMARY KEY (series, date, source)
);
"""
