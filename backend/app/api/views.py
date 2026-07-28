"""Saved views: name a filter set now, come back to it later.

Mounted under /api by main.py. The stored value is the raw URL query string —
see SavedView for why it is kept opaque — plus an appended environment suffix
(`_embed_model`, `_corpus`) recording what the view depended on when saved.
"""
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..schemas import SavedView, SavedViewCreate
from .deps import embeddings_fingerprint, get_conn

router = APIRouter()

# Environment provenance rides inside the stored query string so the table
# keeps its name+string+timestamp shape. It is server-side bookkeeping: written
# on save, stripped before a view is returned, compared to flag staleness. The
# strings are only ever split on "&" and "=" — never URL-decoded — so the
# user's own parameters survive byte-identical, percent-escapes and all.
_ENV_KEYS = ("_embed_model", "_corpus")


def _current_env(conn: sqlite3.Connection) -> dict[str, str]:
    n = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    from ..ml import providers
    return {"_embed_model": providers.active_model_id(),
            "_corpus": f"{n}:{embeddings_fingerprint()}"}


def _split_env(qs: str) -> tuple[str, dict[str, str]]:
    """The user's query string and the env suffix, separated."""
    keep: list[str] = []
    env: dict[str, str] = {}
    for seg in qs.split("&"):
        key, _, val = seg.partition("=")
        if key in _ENV_KEYS:
            env[key] = val
        elif seg:
            keep.append(seg)
    return "&".join(keep), env


def _row_to_view(row: sqlite3.Row, current: dict[str, str]) -> SavedView:
    bare, saved = _split_env(row["query_string"])
    stale = None
    # A view with no env suffix predates fingerprinting: nothing to compare,
    # so no warning — inventing one would punish every pre-existing view.
    if any(saved.get(k) not in (None, current[k]) for k in _ENV_KEYS):
        model = saved.get("_embed_model") or "an unknown model"
        count = saved.get("_corpus", "").partition(":")[0] or "?"
        stale = (f"Saved under {model} with {count} samples; the current "
                 "environment differs — results may not reproduce.")
    return SavedView(name=row["name"], query_string=bare,
                     created_at=row["created_at"], stale_env=stale)


@router.get("/views", response_model=list[SavedView])
def list_views(conn: sqlite3.Connection = Depends(get_conn)):
    # id breaks ties: two views saved inside the same clock tick would otherwise
    # come back in an arbitrary order.
    rows = conn.execute(
        "SELECT name, query_string, created_at FROM saved_views "
        "ORDER BY created_at DESC, id DESC")
    env = _current_env(conn)
    return [_row_to_view(r, env) for r in rows]


@router.post("/views", response_model=SavedView, status_code=201)
def create_view(body: SavedViewCreate, conn: sqlite3.Connection = Depends(get_conn)):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Empty view name")
    # Strip any client-supplied env params before appending fresh ones: the
    # server is the authority on what environment a view was saved under. The
    # appended values contain no "&" or "=", so the suffix parses back cleanly.
    bare, _ = _split_env(body.query_string)
    env = _current_env(conn)
    suffix = f"_embed_model={env['_embed_model']}&_corpus={env['_corpus']}"
    stored = f"{bare}&{suffix}" if bare else suffix
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute(
            "INSERT INTO saved_views(name, query_string, created_at) VALUES (?,?,?)",
            (name, stored, created_at))
    except sqlite3.IntegrityError:
        # Refused rather than overwritten: the name is the user's own label for
        # work they did, and silently replacing it loses that work with no undo.
        raise HTTPException(409, f"A view named '{name}' already exists") from None
    conn.commit()
    return SavedView(name=name, query_string=bare, created_at=created_at)


# `:path` because starlette decodes %2F before routing: a view named
# "night / indoor" saves fine but a single-segment route could never match
# its delete URL again — created once, deletable never.
@router.delete("/views/{name:path}")
def delete_view(name: str, conn: sqlite3.Connection = Depends(get_conn)):
    cur = conn.execute("DELETE FROM saved_views WHERE name = ?", (name.strip(),))
    conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "View not found")
    return {"ok": True}
