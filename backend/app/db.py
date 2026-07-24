"""SQLite access layer.

Plain sqlite3 with a thin helper layer: the schema is small, read-mostly, and
FTS5 requires raw SQL anyway. WAL mode keeps reads fast while ingestion writes.
"""
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY,
    dataset TEXT NOT NULL,
    filename TEXT NOT NULL UNIQUE,
    split TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    filesize INTEGER,
    umap_x REAL,
    umap_y REAL,
    cluster INTEGER
);
CREATE INDEX IF NOT EXISTS idx_samples_split ON samples(split);
CREATE INDEX IF NOT EXISTS idx_samples_dataset ON samples(dataset);

CREATE TABLE IF NOT EXISTS captions (
    id INTEGER PRIMARY KEY,
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_captions_sample ON captions(sample_id);

-- Full-text index over captions (external content table).
CREATE VIRTUAL TABLE IF NOT EXISTS captions_fts USING fts5(
    text, content='captions', content_rowid='id'
);

-- User-curated tags.
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS sample_tags (
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (sample_id, tag_id)
);

-- Tags produced by the optional local VLM enrichment pass.
CREATE TABLE IF NOT EXISTS vlm_tags (
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (sample_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_vlm_tags_tag ON vlm_tags(tag);
"""


def connect() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency-friendly context manager."""
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def fts_escape(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression (implicit AND of terms)."""
    terms = [t.replace('"', '""') for t in query.split() if t.strip()]
    return " ".join(f'"{t}"' for t in terms)
