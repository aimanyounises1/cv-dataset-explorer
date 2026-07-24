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
    text TEXT NOT NULL,
    agreement REAL  -- SigLIP image-caption similarity (CLIPScore-style QA signal)
);
CREATE INDEX IF NOT EXISTS idx_captions_sample ON captions(sample_id);

-- Full-text index over captions (external content table, Porter stemming so
-- "run" matches "running").
CREATE VIRTUAL TABLE IF NOT EXISTS captions_fts USING fts5(
    text, content='captions', content_rowid='id', tokenize='porter unicode61'
);

-- Zero-shot attributes (SigLIP label-bank classification), filterable facets.
CREATE TABLE IF NOT EXISTS attributes (
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    grp TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    PRIMARY KEY (sample_id, grp)
);
CREATE INDEX IF NOT EXISTS idx_attributes_grp_label ON attributes(grp, label);

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
    # check_same_thread=False: FastAPI may run a sync dependency and its
    # endpoint on different threadpool threads; access is still sequential.
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring pre-existing databases up to the current schema (idempotent)."""
    def columns(table: str) -> set[str]:
        return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}

    if "agreement" not in columns("captions"):
        conn.execute("ALTER TABLE captions ADD COLUMN agreement REAL")
    if "caption_consistency" not in columns("samples"):
        conn.execute("ALTER TABLE samples ADD COLUMN caption_consistency REAL")

    # Rebuild the FTS index if it predates Porter stemming.
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'captions_fts'").fetchone()
    if row and "porter" not in (row["sql"] or ""):
        conn.executescript(
            "DROP TABLE captions_fts;"
            "CREATE VIRTUAL TABLE captions_fts USING fts5("
            "  text, content='captions', content_rowid='id', tokenize='porter unicode61');"
        )
        conn.execute("INSERT INTO captions_fts(rowid, text) SELECT id, text FROM captions")


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


# Function words carry no retrieval signal; they are indexed (the tokenizer does
# not strip them) but are never worth reporting to the user as "too common".
STOPWORDS = frozenset(
    "a an and are as at be by for from has he in is it its of on that the to was "
    "were will with his her their this there two three while over under near up "
    "down out off no not they them then than into onto".split()
)
