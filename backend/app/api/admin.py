"""Operational endpoints: pick up new ingestion/analysis output without a
server restart, and check that what is on disk still agrees with the database."""
import sqlite3

from fastapi import APIRouter, Depends

from ..ml.index import get_caption_index, get_index, invalidate_index
from .deps import get_conn

router = APIRouter()


def index_consistency(conn) -> dict:
    """Compare each embedding index against the table it indexes.

    The two can silently disagree. `EmbeddingIndex.save` overwrites wholesale and
    nothing re-checks it afterwards, so deleting rows from the database without
    re-running ingest leaves vectors behind that belong to nothing. This is not
    hypothetical: three smoke-test fixture captions were once removed from this
    database and their vectors were not, leaving `caption_embeddings.npy` with
    40,003 rows against 40,000 captions.

    It is harmless while it stays small — lookups are keyed by id, so an orphan
    is simply never reached — but it means the index is not a reliable statement
    about the corpus, and nothing was telling anyone. Reported rather than
    repaired: the fix is re-running ingest, which is not something an endpoint
    should do behind someone's back.
    """
    out: dict[str, dict] = {}
    for kind, index, table, id_col in (
        ("image", get_index(), "samples", "id"),
        ("caption", get_caption_index(), "captions", "id"),
    ):
        if index is None:
            out[kind] = {"available": False}
            continue
        rows = conn.execute(f"SELECT {id_col} FROM {table}").fetchall()
        db_ids = {r[0] for r in rows}
        index_ids = {int(i) for i in index.ids}
        orphans = index_ids - db_ids          # vectors for rows that are gone
        missing = db_ids - index_ids          # rows with no vector
        out[kind] = {
            "available": True,
            "index_vectors": len(index_ids),
            "db_rows": len(db_ids),
            "orphan_vectors": len(orphans),
            "rows_without_vectors": len(missing),
            "consistent": not orphans and not missing,
            "sample_orphan_ids": sorted(orphans)[:5],
        }
    return out


@router.get("/admin/integrity")
def integrity(conn: sqlite3.Connection = Depends(get_conn)):
    """Does the data on disk still describe the data in the database?"""
    checks = index_consistency(conn)
    ok = all(c.get("consistent", True) for c in checks.values())
    return {
        "ok": ok,
        "indexes": checks,
        "hint": None if ok else
                "Re-run `python -m app.ingest` (and `python -m app.analyze`) to "
                "rebuild the indexes from the current database, then POST "
                "/api/admin/reload.",
    }


@router.post("/admin/reload")
def reload_indexes(conn: sqlite3.Connection = Depends(get_conn)):
    from ..ml import providers
    from . import leakage, stats

    # Providers first: index loading below resolves through the active
    # provider, so a freshly built Qwen index (or a repaired SigLIP one) must
    # be re-probed before the caches refill.
    providers.invalidate_providers()
    invalidate_index()
    stats.clear_caches()
    leakage.clear_cache()
    checks = index_consistency(conn)
    return {
        "ok": True,
        "image_index": get_index() is not None,
        "caption_index": get_caption_index() is not None,
        # Surfaced on the operation that exists to pick up new index files, which
        # is exactly when a mismatch is most likely and most cheaply fixed.
        "integrity": checks,
    }
