"""SQLite access layer.

Plain sqlite3 with a thin helper layer: the schema is small, read-mostly, and
FTS5 requires raw SQL anyway. WAL mode keeps reads fast while ingestion writes.
"""
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from . import config

# Difficulty axes, after Mobileye's long-tail filter panel: none of these
# describe what is *in* a scene, they describe how hard it is to work with.
# Each is an integer 0-10 percentile bucket over this dataset — see
# analyze.compute_axes for what that costs.
AXES = ("legibility", "rarity", "difficulty", "clutter")

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

-- Named URL query strings. Stored opaquely — the server never parses one, so a
-- view keeps working when the UI grows a filter the backend knows nothing about.
CREATE TABLE IF NOT EXISTS saved_views (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    query_string TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Albums: first-class ordered collections. A tag is a flat label; an album is
-- a destination — ordered membership, a cover, provenance. `origin` records
-- who proposed it ('manual' | 'tag' | 'agent') so agent-proposed albums stay
-- reviewable rather than indistinguishable from the user's own curation.
-- No FK cascade on album_items: membership is deleted explicitly with its
-- album, so the invariant holds even on connections without foreign_keys=ON.
CREATE TABLE IF NOT EXISTS albums (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    summary TEXT,
    category TEXT,
    notes TEXT,
    cover_sample_id INTEGER,              -- NULL: cover falls back to first item
    origin TEXT NOT NULL DEFAULT 'manual',
    position INTEGER NOT NULL DEFAULT 0,  -- ordering among albums
    created_at TEXT NOT NULL,             -- ISO-8601 UTC, like saved_views
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS album_items (
    album_id INTEGER NOT NULL,
    sample_id INTEGER NOT NULL,
    position INTEGER NOT NULL,            -- ordering within the album
    added_at TEXT NOT NULL,
    PRIMARY KEY (album_id, sample_id)
);
CREATE INDEX IF NOT EXISTS idx_album_items_album ON album_items(album_id, position);

-- Activity: an append-only trail of what happened in the workspace. Server
-- endpoints write their own events (album_*) inside the mutation that caused
-- them; clients may append snapshot kinds through the API. Payload is opaque
-- JSON so the trail survives the UI growing event shapes the server has no
-- column for — the saved_views argument, applied to history.
CREATE TABLE IF NOT EXISTS activity_events (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,                -- JSON object
    created_at TEXT NOT NULL              -- ISO-8601 UTC
);

-- Annotations: regions drawn over a sample. Rows, never pixels — the source
-- images stay immutable. Geometry is normalized 0..1 coordinates so an
-- annotation survives any resize of the rendered image.
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY,
    sample_id INTEGER NOT NULL,
    kind TEXT NOT NULL,                   -- 'rect' | 'polygon'
    geometry TEXT NOT NULL,               -- JSON: rect {x,y,w,h} | polygon {points}
    label TEXT,
    created_at TEXT NOT NULL              -- ISO-8601 UTC
);
CREATE INDEX IF NOT EXISTS idx_annotations_sample ON annotations(sample_id);
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

    # Difficulty axes: integer 0-10 percentile buckets, plus the raw component
    # values behind each so the UI can explain a score without a second query.
    sample_cols = columns("samples")
    for axis in AXES:
        if axis not in sample_cols:
            conn.execute(f"ALTER TABLE samples ADD COLUMN {axis} INTEGER")
    if "axis_detail" not in sample_cols:
        conn.execute("ALTER TABLE samples ADD COLUMN axis_detail TEXT")
    for axis in AXES:
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_samples_{axis} ON samples({axis})")

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


# Function words carry no retrieval signal. They are indexed (the tokenizer does
# not strip them), but they are dropped from the query conjunction, never
# reported to the user as "too common", and excluded from term statistics.
STOPWORDS = frozenset(
    "a an and are as at be by for from has he in is it its of on that the to was "
    "were will with his her their this there two three while over under near up "
    "down out off no not they them then than into onto".split()
)


def _bare(token: str) -> str:
    """The token with punctuation and case stripped, for stopword comparison
    only. The term that reaches FTS5 is always the original."""
    return "".join(ch for ch in token.lower() if ch.isalnum())


def match_terms(query: str) -> list[str]:
    """The query terms that actually take part in the conjunction.

    `fts_escape` ANDs every term, so a function word is a hard requirement: a
    caption that says "man on a bench" cannot match "man on the bench", because
    of "the". Dropping stopwords from the conjunction is worth, on the 1,000
    caption benchmark, keyword R@10 3.6% -> 6.4% with the empty-result rate
    falling 90.6% -> 84.0%; on realistic 3-content-word phrases (which keep
    their function words) R@10 goes 12.6% -> 16.4% and empty 41.5% -> 26.7%.
    Fused ranking is unchanged within noise, measured on both the benchmark
    sample and a disjoint dev split, in both directions (hybrid MRR 0.6313 ->
    0.6308 on whole captions, 0.3310 -> 0.3363 on four-word phrases).

    This narrows the conjunction; it does not widen it to OR. Widening was
    measured twice and refuted twice — keyword R@10 rises to 53.8% in isolation
    while hybrid MRR falls 0.6313 -> 0.5383, because a broad lexical list
    displaces stronger semantic candidates.

    A query made only of stopwords keeps all of its terms: "the two" should
    still search for something rather than silently match everything.
    """
    terms = [t for t in query.split() if t.strip()]
    content = [t for t in terms if _bare(t) and _bare(t) not in STOPWORDS]
    return content or terms


def fts_escape(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression (implicit AND of the
    query's content terms — see `match_terms`)."""
    return " ".join(f'"{t.replace(chr(34), chr(34) * 2)}"' for t in match_terms(query))
