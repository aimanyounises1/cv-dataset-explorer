"""Saved views: name a filter set now, come back to it later.

Mounted under /api by main.py. The stored value is the raw URL query string —
see SavedView for why it is kept opaque.
"""
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..schemas import SavedView, SavedViewCreate
from .deps import get_conn

router = APIRouter()


def _row_to_view(row: sqlite3.Row) -> SavedView:
    return SavedView(name=row["name"], query_string=row["query_string"],
                     created_at=row["created_at"])


@router.get("/views", response_model=list[SavedView])
def list_views(conn: sqlite3.Connection = Depends(get_conn)):
    # id breaks ties: two views saved inside the same clock tick would otherwise
    # come back in an arbitrary order.
    rows = conn.execute(
        "SELECT name, query_string, created_at FROM saved_views "
        "ORDER BY created_at DESC, id DESC")
    return [_row_to_view(r) for r in rows]


@router.post("/views", response_model=SavedView, status_code=201)
def create_view(body: SavedViewCreate, conn: sqlite3.Connection = Depends(get_conn)):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Empty view name")
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute(
            "INSERT INTO saved_views(name, query_string, created_at) VALUES (?,?,?)",
            (name, body.query_string, created_at))
    except sqlite3.IntegrityError:
        # Refused rather than overwritten: the name is the user's own label for
        # work they did, and silently replacing it loses that work with no undo.
        raise HTTPException(409, f"A view named '{name}' already exists") from None
    conn.commit()
    return SavedView(name=name, query_string=body.query_string, created_at=created_at)


@router.delete("/views/{name}")
def delete_view(name: str, conn: sqlite3.Connection = Depends(get_conn)):
    cur = conn.execute("DELETE FROM saved_views WHERE name = ?", (name.strip(),))
    conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "View not found")
    return {"ok": True}
