"""Albums: first-class ordered collections of samples.

A tag is a flat label; an album is a destination — ordered membership, a
cover, a summary, provenance. Mounted under /api by main.py. Conversion from a
tag copies the membership and leaves the tag in place: tags stay labels, and
deleting one would turn a conversion into a move.
"""
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from ..schemas import AlbumCreate, AlbumDetail, AlbumSummary, AlbumUpdate
from .deps import (
    ID_PARAM_LIMIT,
    BoundedId,
    PathId,
    first_captions,
    get_conn,
    record_activity,
    row_to_card,
    thumb_url,
)

router = APIRouter()

# Same ceiling as bulk tagging: more ids than the corpus could ever hold is a
# runaway client, not a selection.
MAX_ITEM_IDS = 100_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _album_row(conn: sqlite3.Connection, album_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM albums WHERE id = ?", (album_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Album not found")
    return row


def _cover_thumb(conn: sqlite3.Connection, row: sqlite3.Row) -> Optional[str]:
    """Thumb of the chosen cover, else the first item, else None (empty album)."""
    cover_id = row["cover_sample_id"]
    if cover_id is None:
        first = conn.execute(
            "SELECT sample_id FROM album_items WHERE album_id = ? "
            "ORDER BY position, sample_id LIMIT 1", (row["id"],)).fetchone()
        if first is None:
            return None
        cover_id = first["sample_id"]
    hit = conn.execute("SELECT filename FROM samples WHERE id = ?", (cover_id,)).fetchone()
    return thumb_url(hit["filename"]) if hit else None


def _summary(conn: sqlite3.Connection, row: sqlite3.Row) -> AlbumSummary:
    n = conn.execute("SELECT COUNT(*) FROM album_items WHERE album_id = ?",
                     (row["id"],)).fetchone()[0]
    return AlbumSummary(
        id=row["id"], name=row["name"], summary=row["summary"],
        category=row["category"], origin=row["origin"], item_count=n,
        cover=_cover_thumb(conn, row), created_at=row["created_at"])


def _detail(conn: sqlite3.Connection, row: sqlite3.Row) -> AlbumDetail:
    sample_rows = conn.execute(
        "SELECT s.* FROM album_items ai JOIN samples s ON s.id = ai.sample_id "
        "WHERE ai.album_id = ? ORDER BY ai.position, ai.sample_id",
        (row["id"],)).fetchall()
    ids = [r["id"] for r in sample_rows]
    # Chunked: an album approaching corpus size would otherwise blow SQLite's
    # host-parameter ceiling inside first_captions' IN (...).
    caps: dict[int, str] = {}
    for i in range(0, len(ids), ID_PARAM_LIMIT):
        caps.update(first_captions(conn, ids[i:i + ID_PARAM_LIMIT]))
    return AlbumDetail(
        id=row["id"], name=row["name"], summary=row["summary"],
        category=row["category"], notes=row["notes"], origin=row["origin"],
        item_count=len(ids), cover=_cover_thumb(conn, row),
        cover_sample_id=row["cover_sample_id"],
        created_at=row["created_at"], updated_at=row["updated_at"],
        items=[row_to_card(r, caption=caps.get(r["id"])) for r in sample_rows])


def _next_album_position(conn: sqlite3.Connection) -> int:
    # New albums land at the end of the shelf; reorder rewrites all positions.
    return conn.execute("SELECT COALESCE(MAX(position) + 1, 0) FROM albums").fetchone()[0]


def _compact(conn: sqlite3.Connection, album_id: int) -> None:
    """Renumber positions 0..n-1 preserving order, so positions stay dense."""
    rows = conn.execute(
        "SELECT sample_id FROM album_items WHERE album_id = ? "
        "ORDER BY position, sample_id", (album_id,)).fetchall()
    conn.executemany(
        "UPDATE album_items SET position = ? WHERE album_id = ? AND sample_id = ?",
        [(i, album_id, r["sample_id"]) for i, r in enumerate(rows)])


def _exactness_error(current: set[int], given: list[int], what: str) -> None:
    """400 naming the difference when `given` is not exactly `current`."""
    if len(given) != len(set(given)):
        raise HTTPException(400, f"Duplicate {what} ids in the new order")
    missing = sorted(current - set(given))
    unknown = sorted(set(given) - current)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} current {what}(s): {missing[:10]}")
        if unknown:
            parts.append(f"{len(unknown)} id(s) not present: {unknown[:10]}")
        raise HTTPException(
            400, "New order must be exactly the current membership — " + "; ".join(parts))


@router.get("/albums", response_model=list[AlbumSummary])
def list_albums(conn: sqlite3.Connection = Depends(get_conn)):
    # One count+cover lookup per album: the shelf is user-curated and small.
    rows = conn.execute("SELECT * FROM albums ORDER BY position, id").fetchall()
    return [_summary(conn, r) for r in rows]


@router.post("/albums", response_model=AlbumDetail, status_code=201)
def create_album(body: AlbumCreate, conn: sqlite3.Connection = Depends(get_conn)):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Empty album name")
    now = _now()
    try:
        cur = conn.execute(
            "INSERT INTO albums(name, summary, category, notes, origin, position, "
            "created_at, updated_at) VALUES (?,?,?,?,'manual',?,?,?)",
            (name, body.summary, body.category, body.notes,
             _next_album_position(conn), now, now))
    except sqlite3.IntegrityError:
        # Refused rather than overwritten — same reasoning as saved views: the
        # name is the user's own label for work they did.
        raise HTTPException(409, f"An album named '{name}' already exists") from None
    record_activity(conn, "album_create", {"album_id": cur.lastrowid, "name": name, "n": 0})
    conn.commit()
    return _detail(conn, _album_row(conn, cur.lastrowid))


# Declared before the {album_id} routes: static segments must never be
# shadowed by a path parameter.
@router.put("/albums/order")
def reorder_albums(album_ids: list[BoundedId] = Body(..., embed=True),
                   conn: sqlite3.Connection = Depends(get_conn)):
    if len(album_ids) > MAX_ITEM_IDS:
        raise HTTPException(400, f"Too many ids: {len(album_ids):,}. "
                                 f"The limit is {MAX_ITEM_IDS:,}.")
    current = {r["id"] for r in conn.execute("SELECT id FROM albums")}
    _exactness_error(current, album_ids, "album")
    conn.executemany("UPDATE albums SET position = ? WHERE id = ?",
                     [(i, aid) for i, aid in enumerate(album_ids)])
    conn.commit()
    return {"ok": True}


@router.post("/albums/from-tag", response_model=AlbumDetail, status_code=201)
def album_from_tag(tag: str = Body(..., embed=True, max_length=200),
                   conn: sqlite3.Connection = Depends(get_conn)):
    """Explicit tag → album conversion. The tag survives: tags stay labels,
    the album is a copy with its own life from here on."""
    name = tag.strip().lower()          # tags are stored lowercased
    if not name:
        raise HTTPException(400, "Empty tag")
    tag_row = conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
    if tag_row is None:
        raise HTTPException(404, f"No tag named '{name}'")
    member_ids = [r["sample_id"] for r in conn.execute(
        "SELECT sample_id FROM sample_tags WHERE tag_id = ? ORDER BY sample_id",
        (tag_row["id"],))]
    now = _now()
    try:
        cur = conn.execute(
            "INSERT INTO albums(name, origin, position, created_at, updated_at) "
            "VALUES (?, 'tag', ?, ?, ?)", (name, _next_album_position(conn), now, now))
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"An album named '{name}' already exists") from None
    conn.executemany(
        "INSERT INTO album_items(album_id, sample_id, position, added_at) "
        "VALUES (?,?,?,?)",
        [(cur.lastrowid, sid, i, now) for i, sid in enumerate(member_ids)])
    record_activity(conn, "album_from_tag",
                    {"album_id": cur.lastrowid, "name": name, "n": len(member_ids)})
    conn.commit()
    return _detail(conn, _album_row(conn, cur.lastrowid))


@router.get("/albums/{album_id}", response_model=AlbumDetail)
def get_album(album_id: PathId, conn: sqlite3.Connection = Depends(get_conn)):
    return _detail(conn, _album_row(conn, album_id))


@router.patch("/albums/{album_id}", response_model=AlbumDetail)
def update_album(album_id: PathId, body: AlbumUpdate,
                 conn: sqlite3.Connection = Depends(get_conn)):
    _album_row(conn, album_id)
    changes = body.model_dump(exclude_unset=True)
    if "name" in changes:
        name = (changes["name"] or "").strip()
        if not name:
            raise HTTPException(400, "Empty album name")
        changes["name"] = name
    if changes.get("cover_sample_id") is not None:
        member = conn.execute(
            "SELECT 1 FROM album_items WHERE album_id = ? AND sample_id = ?",
            (album_id, changes["cover_sample_id"])).fetchone()
        if member is None:
            raise HTTPException(400, "Cover must be a member of the album, or null")
    if changes:
        # Keys are AlbumUpdate field names, never user input — safe to inline.
        sets = ", ".join(f"{k} = ?" for k in changes)
        try:
            conn.execute(f"UPDATE albums SET {sets}, updated_at = ? WHERE id = ?",
                         [*changes.values(), _now(), album_id])
        except sqlite3.IntegrityError:
            raise HTTPException(
                409, f"An album named '{changes['name']}' already exists") from None
        conn.commit()
    return _detail(conn, _album_row(conn, album_id))


@router.delete("/albums/{album_id}")
def delete_album(album_id: PathId, conn: sqlite3.Connection = Depends(get_conn)):
    row = _album_row(conn, album_id)
    # Membership goes with the album, explicitly — the schema carries no FK
    # cascade, so this is where the invariant is enforced.
    cur = conn.execute("DELETE FROM album_items WHERE album_id = ?", (album_id,))
    conn.execute("DELETE FROM albums WHERE id = ?", (album_id,))
    record_activity(conn, "album_delete",
                    {"album_id": album_id, "name": row["name"], "n": cur.rowcount})
    conn.commit()
    return {"ok": True}


@router.post("/albums/{album_id}/items")
def add_items(album_id: PathId,
              sample_ids: list[BoundedId] = Body(..., embed=True),
              conn: sqlite3.Connection = Depends(get_conn)):
    album = _album_row(conn, album_id)
    if not sample_ids:
        raise HTTPException(400, "No samples given")
    if len(sample_ids) > MAX_ITEM_IDS:
        raise HTTPException(400, f"Too many samples: {len(sample_ids):,}. "
                                 f"The limit is {MAX_ITEM_IDS:,}.")
    base = conn.execute(
        "SELECT COALESCE(MAX(position) + 1, 0) FROM album_items WHERE album_id = ?",
        (album_id,)).fetchone()[0]
    now = _now()
    before = conn.total_changes
    # INSERT ... SELECT keeps unknown ids out (the bulk-tag shape); OR IGNORE
    # keeps an existing member at its position rather than moving it. Skipped
    # rows leave position gaps, which is fine — order is what matters, and
    # removal compacts.
    conn.executemany(
        "INSERT OR IGNORE INTO album_items(album_id, sample_id, position, added_at) "
        "SELECT ?, id, ?, ? FROM samples WHERE id = ?",
        [(album_id, base + i, now, sid) for i, sid in enumerate(sample_ids)])
    # What THIS call added — not the album's total. Measured before the
    # updated_at bump so that write cannot inflate the count.
    added = conn.total_changes - before
    conn.execute("UPDATE albums SET updated_at = ? WHERE id = ?", (now, album_id))
    record_activity(conn, "album_items_add",
                    {"album_id": album_id, "name": album["name"], "n": added})
    conn.commit()
    return {"ok": True, "added": added}


@router.delete("/albums/{album_id}/items/{sample_id}")
def remove_item(album_id: PathId, sample_id: PathId,
                conn: sqlite3.Connection = Depends(get_conn)):
    _album_row(conn, album_id)
    cur = conn.execute("DELETE FROM album_items WHERE album_id = ? AND sample_id = ?",
                       (album_id, sample_id))
    if cur.rowcount == 0:
        raise HTTPException(404, "Sample is not in this album")
    # A removed cover would silently keep fronting an album it no longer
    # belongs to; clearing it hands the job back to the first-item fallback.
    conn.execute("UPDATE albums SET cover_sample_id = NULL "
                 "WHERE id = ? AND cover_sample_id = ?", (album_id, sample_id))
    _compact(conn, album_id)
    conn.execute("UPDATE albums SET updated_at = ? WHERE id = ?", (_now(), album_id))
    conn.commit()
    return {"ok": True}


@router.put("/albums/{album_id}/items/order")
def reorder_items(album_id: PathId,
                  sample_ids: list[BoundedId] = Body(..., embed=True),
                  conn: sqlite3.Connection = Depends(get_conn)):
    album = _album_row(conn, album_id)
    if len(sample_ids) > MAX_ITEM_IDS:
        raise HTTPException(400, f"Too many ids: {len(sample_ids):,}. "
                                 f"The limit is {MAX_ITEM_IDS:,}.")
    current = {r["sample_id"] for r in conn.execute(
        "SELECT sample_id FROM album_items WHERE album_id = ?", (album_id,))}
    _exactness_error(current, sample_ids, "member")
    conn.executemany(
        "UPDATE album_items SET position = ? WHERE album_id = ? AND sample_id = ?",
        [(i, album_id, sid) for i, sid in enumerate(sample_ids)])
    conn.execute("UPDATE albums SET updated_at = ? WHERE id = ?", (_now(), album_id))
    record_activity(conn, "album_reorder",
                    {"album_id": album_id, "name": album["name"], "n": len(sample_ids)})
    conn.commit()
    return {"ok": True}
