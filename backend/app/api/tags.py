"""User tags (curation) + VLM tag facets."""
import sqlite3

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from ..schemas import TagInfo
from .deps import get_conn

router = APIRouter()


@router.get("/tags", response_model=list[TagInfo])
def list_tags(conn: sqlite3.Connection = Depends(get_conn)):
    rows = conn.execute(
        "SELECT t.name, COUNT(st.sample_id) AS n FROM tags t "
        "LEFT JOIN sample_tags st ON st.tag_id = t.id GROUP BY t.id ORDER BY n DESC, t.name")
    return [TagInfo(name=r["name"], count=r["n"]) for r in rows]


@router.get("/vlm-tags", response_model=list[TagInfo])
def list_vlm_tags(limit: int = Query(40, ge=1, le=1000),
                  conn: sqlite3.Connection = Depends(get_conn)):
    rows = conn.execute(
        "SELECT tag AS name, COUNT(*) AS n FROM vlm_tags GROUP BY tag "
        "ORDER BY n DESC LIMIT ?", (limit,))
    return [TagInfo(name=r["name"], count=r["n"]) for r in rows]


@router.post("/samples/{sample_id}/tags")
def add_tag(sample_id: int, name: str = Body(..., embed=True),
            conn: sqlite3.Connection = Depends(get_conn)):
    name = name.strip().lower()
    if not name:
        raise HTTPException(400, "Empty tag")
    if conn.execute("SELECT 1 FROM samples WHERE id = ?", (sample_id,)).fetchone() is None:
        raise HTTPException(404, "Sample not found")
    conn.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (name,))
    tag_id = conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()["id"]
    conn.execute("INSERT OR IGNORE INTO sample_tags(sample_id, tag_id) VALUES (?,?)",
                 (sample_id, tag_id))
    conn.commit()
    return {"ok": True}


@router.post("/tags/bulk")
def bulk_tag(sample_ids: list[int] = Body(...), name: str = Body(...),
             conn: sqlite3.Connection = Depends(get_conn)):
    """Tag many samples at once (used by map box-selection)."""
    name = name.strip().lower()
    if not name:
        raise HTTPException(400, "Empty tag")
    if not sample_ids:
        raise HTTPException(400, "No samples selected")
    # More ids than the corpus could ever hold is a runaway client, not a
    # selection; a 1M-id body used to be accepted and fed to executemany.
    if len(sample_ids) > 100_000:
        raise HTTPException(400, f"Too many samples: {len(sample_ids):,}. "
                                 "The limit is 100,000.")
    conn.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (name,))
    tag_id = conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()["id"]
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO sample_tags(sample_id, tag_id) "
        "SELECT id, ? FROM samples WHERE id = ?",
        [(tag_id, sid) for sid in sample_ids])
    conn.commit()
    # What THIS call tagged — not the tag's corpus-wide total, which is what
    # the field used to report while sitting next to `ok` in the response.
    return {"ok": True, "tag": name, "tagged": conn.total_changes - before}


@router.delete("/samples/{sample_id}/tags/{name}")
def remove_tag(sample_id: int, name: str, conn: sqlite3.Connection = Depends(get_conn)):
    conn.execute(
        "DELETE FROM sample_tags WHERE sample_id = ? AND "
        "tag_id = (SELECT id FROM tags WHERE name = ?)", (sample_id, name.strip().lower()))
    # Garbage-collect orphan tags.
    conn.execute("DELETE FROM tags WHERE id NOT IN (SELECT DISTINCT tag_id FROM sample_tags)")
    conn.commit()
    return {"ok": True}
