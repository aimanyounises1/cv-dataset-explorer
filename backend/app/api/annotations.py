"""Annotations: regions drawn over a sample.

Rows, never pixels — the source images stay immutable, and an annotation is a
record about an image, not a change to it. Geometry is normalized 0..1
coordinates (validated in `AnnotationCreate`) so a region survives any
rendered size. Mounted under /api by main.py.
"""
import json
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..schemas import AnnotationCreate, AnnotationOut
from .deps import PathId, get_conn

router = APIRouter()

# More regions than a human draws on one image is a runaway client, and an
# unbounded list would eventually be a payload problem for the sample page.
MAX_PER_SAMPLE = 200


def _require_sample(conn: sqlite3.Connection, sample_id: int) -> None:
    if conn.execute("SELECT 1 FROM samples WHERE id = ?", (sample_id,)).fetchone() is None:
        raise HTTPException(404, "Sample not found")


def _row_out(r: sqlite3.Row) -> AnnotationOut:
    try:
        geometry = json.loads(r["geometry"])
    except ValueError:
        geometry = {}
    return AnnotationOut(id=r["id"], sample_id=r["sample_id"], kind=r["kind"],
                         geometry=geometry, label=r["label"],
                         created_at=r["created_at"])


@router.get("/samples/{sample_id}/annotations", response_model=list[AnnotationOut])
def list_annotations(sample_id: PathId, conn: sqlite3.Connection = Depends(get_conn)):
    _require_sample(conn, sample_id)
    rows = conn.execute(
        "SELECT * FROM annotations WHERE sample_id = ? ORDER BY id", (sample_id,))
    return [_row_out(r) for r in rows]


@router.post("/samples/{sample_id}/annotations", response_model=AnnotationOut,
             status_code=201)
def add_annotation(sample_id: PathId, body: AnnotationCreate,
                   conn: sqlite3.Connection = Depends(get_conn)):
    _require_sample(conn, sample_id)
    n = conn.execute("SELECT COUNT(*) FROM annotations WHERE sample_id = ?",
                     (sample_id,)).fetchone()[0]
    if n >= MAX_PER_SAMPLE:
        raise HTTPException(400, f"This sample already has {n} annotations. "
                                 f"The limit is {MAX_PER_SAMPLE}.")
    created_at = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO annotations(sample_id, kind, geometry, label, created_at) "
        "VALUES (?,?,?,?,?)",
        (sample_id, body.kind, json.dumps(body.geometry, separators=(",", ":")),
         body.label, created_at))
    conn.commit()
    return AnnotationOut(id=cur.lastrowid, sample_id=sample_id, kind=body.kind,
                         geometry=body.geometry, label=body.label,
                         created_at=created_at)


@router.delete("/annotations/{annotation_id}")
def delete_annotation(annotation_id: PathId,
                      conn: sqlite3.Connection = Depends(get_conn)):
    cur = conn.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
    conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "Annotation not found")
    return {"ok": True}
