"""Activity: an append-only trail of what happened in the workspace.

Two writers, one table. Server endpoints record the mutations they perform
(`album_*` kinds, written via `deps.record_activity` inside the transaction
that did the work); clients may append snapshot kinds through the POST here.
The POST's kind allowlist is what keeps the two apart — a client cannot write
an `album_*` event, so the trail never claims a server action that did not
happen. Mounted under /api by main.py.
"""
import json
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from ..schemas import ActivityCreate, ActivityEvent
from .deps import PathId, get_conn

router = APIRouter()

# Payload is opaque JSON, but opaque is not unbounded: the saved-view name
# measured this lesson already (a 200,000-character name was stored).
MAX_PAYLOAD_CHARS = 4000


@router.get("/activity", response_model=list[ActivityEvent])
def list_activity(limit: int = Query(50, ge=1, le=200),
                  conn: sqlite3.Connection = Depends(get_conn)):
    # Newest first by id, not created_at: ids are monotonic, timestamps can
    # collide within one clock tick (the saved-views ordering lesson).
    rows = conn.execute(
        "SELECT id, kind, payload, created_at FROM activity_events "
        "ORDER BY id DESC LIMIT ?", (limit,))
    out = []
    for r in rows:
        try:
            payload = json.loads(r["payload"])
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {"value": payload}
        out.append(ActivityEvent(id=r["id"], kind=r["kind"], payload=payload,
                                 created_at=r["created_at"]))
    return out


@router.post("/activity", response_model=ActivityEvent, status_code=201)
def add_activity(body: ActivityCreate, conn: sqlite3.Connection = Depends(get_conn)):
    serialized = json.dumps(body.payload, separators=(",", ":"))
    if len(serialized) > MAX_PAYLOAD_CHARS:
        raise HTTPException(400, f"Payload too large: {len(serialized):,} chars "
                                 f"serialized. The limit is {MAX_PAYLOAD_CHARS:,}.")
    created_at = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO activity_events(kind, payload, created_at) VALUES (?,?,?)",
        (body.kind, serialized, created_at))
    conn.commit()
    return ActivityEvent(id=cur.lastrowid, kind=body.kind, payload=body.payload,
                         created_at=created_at)


@router.delete("/activity/{event_id}")
def delete_activity(event_id: PathId, conn: sqlite3.Connection = Depends(get_conn)):
    cur = conn.execute("DELETE FROM activity_events WHERE id = ?", (event_id,))
    conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "Event not found")
    return {"ok": True}


@router.delete("/activity")
def clear_activity(conn: sqlite3.Connection = Depends(get_conn)):
    cur = conn.execute("DELETE FROM activity_events")
    conn.commit()
    return {"ok": True, "cleared": cur.rowcount}
