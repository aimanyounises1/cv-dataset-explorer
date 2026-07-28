"""Annotations: regions drawn over a sample.

Rows, never pixels — the source images stay immutable, and an annotation is a
record about an image, not a change to it. Geometry is normalized 0..1
coordinates (validated in `AnnotationCreate`) so a region survives any
rendered size. Mounted under /api by main.py.
"""
import json
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response

from ..schemas import AnnotationCreate, AnnotationOut, ObjectLabelOut
from .deps import PathId, get_conn

router = APIRouter()

# More regions than a human draws on one image is a runaway client, and an
# unbounded list would eventually be a payload problem for the sample page.
MAX_PER_SAMPLE = 200


def _require_sample(conn: sqlite3.Connection, sample_id: int) -> None:
    if conn.execute("SELECT 1 FROM samples WHERE id = ?", (sample_id,)).fetchone() is None:
        raise HTTPException(404, "Sample not found")


def canonical_object_name(name: str) -> str:
    """Normalize a detector phrase to a taxonomy key.

    Grounding DINO is prompted with natural-language phrases (``a dog``), while
    COCO category names are nouns (``dog``). Only a leading English article is
    removed; the remaining label is preserved as explicit taxonomy data.
    """
    words = " ".join(name.strip().lower().split()).split()
    if len(words) > 1 and words[0] in {"a", "an"}:
        words = words[1:]
    return " ".join(words)


def _label_path(conn: sqlite3.Connection, label_id: int) -> list[str]:
    rows = conn.execute(
        "WITH RECURSIVE ancestors(id, name, parent_id, depth) AS ("
        " SELECT id, name, parent_id, 0 FROM object_labels WHERE id = ?"
        " UNION ALL"
        " SELECT p.id, p.name, p.parent_id, ancestors.depth + 1"
        " FROM object_labels p JOIN ancestors ON p.id = ancestors.parent_id"
        ") SELECT name FROM ancestors ORDER BY depth DESC",
        (label_id,)).fetchall()
    return [r["name"] for r in rows]


def lookup_object_label(conn: sqlite3.Connection, name: str | None) -> ObjectLabelOut | None:
    if not name:
        return None
    canonical = canonical_object_name(name)
    row = conn.execute(
        "SELECT id, name, parent_id FROM object_labels WHERE lower(name) = lower(?)",
        (canonical,)).fetchone()
    if row is None:
        return None
    return ObjectLabelOut(id=row["id"], name=row["name"], parent_id=row["parent_id"],
                          path=_label_path(conn, row["id"]))


def ensure_object_label(
    conn: sqlite3.Connection,
    label_name: str,
    parent_name: str | None,
) -> ObjectLabelOut:
    """Resolve or create one explicit label edge, rejecting conflicts."""
    label_name = canonical_object_name(label_name)
    parent_name = canonical_object_name(parent_name) if parent_name else None
    if not label_name:
        raise HTTPException(422, "label_name cannot be blank")
    if label_name == parent_name:
        raise HTTPException(422, "an object label cannot be its own parent")

    parent = None
    if parent_name:
        parent = lookup_object_label(conn, parent_name)
        if parent is None:
            cur = conn.execute(
                "INSERT INTO object_labels(name, parent_id) VALUES (?, NULL)",
                (parent_name,))
            parent = ObjectLabelOut(id=cur.lastrowid, name=parent_name,
                                    parent_id=None, path=[parent_name])

    existing = lookup_object_label(conn, label_name)
    wanted_parent = parent.id if parent else None
    if existing is not None:
        if existing.parent_id != wanted_parent and parent_name is not None:
            raise HTTPException(
                409, f"Object label '{label_name}' already has a different parent")
        return existing
    cur = conn.execute(
        "INSERT INTO object_labels(name, parent_id) VALUES (?, ?)",
        (label_name, wanted_parent))
    return ObjectLabelOut(
        id=cur.lastrowid, name=label_name, parent_id=wanted_parent,
        path=([*parent.path, label_name] if parent else [label_name]))


def _row_out(conn: sqlite3.Connection, r: sqlite3.Row) -> AnnotationOut:
    try:
        geometry = json.loads(r["geometry"])
    except ValueError:
        geometry = {}
    mask = conn.execute(
        "SELECT width, height, model_id, prompt_json, predicted_iou "
        "FROM annotation_masks WHERE annotation_id = ?", (r["id"],)).fetchone()
    label = conn.execute(
        "SELECT l.id, l.name, l.parent_id FROM annotation_object_labels al "
        "JOIN object_labels l ON l.id = al.label_id WHERE al.annotation_id = ?",
        (r["id"],)).fetchone()
    label_name = label["name"] if label else None
    label_path = _label_path(conn, label["id"]) if label else []
    prompt = None
    if mask is not None:
        try:
            prompt = json.loads(mask["prompt_json"])
        except ValueError:
            prompt = None
    points = prompt.get("points", []) if isinstance(prompt, dict) else []
    box = prompt.get("box") if isinstance(prompt, dict) else None
    return AnnotationOut(id=r["id"], sample_id=r["sample_id"], kind=r["kind"],
                         geometry=geometry, label=r["label"],
                         created_at=r["created_at"], label_name=label_name,
                         parent_name=(label_path[-2] if len(label_path) > 1 else None),
                         label_path=label_path, points=points, box=box,
                         bbox=geometry if r["kind"] == "mask" else None,
                         mask_url=(f"/api/annotations/{r['id']}/mask"
                                   if mask is not None else None),
                         mask_width=mask["width"] if mask else None,
                         mask_height=mask["height"] if mask else None,
                         model_id=mask["model_id"] if mask else None,
                         prompt=prompt,
                         predicted_iou=mask["predicted_iou"] if mask else None)


@router.get("/samples/{sample_id}/annotations", response_model=list[AnnotationOut])
def list_annotations(sample_id: PathId, conn: sqlite3.Connection = Depends(get_conn)):
    _require_sample(conn, sample_id)
    rows = conn.execute(
        "SELECT * FROM annotations WHERE sample_id = ? ORDER BY id", (sample_id,))
    return [_row_out(conn, r) for r in rows]


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


@router.get("/object-labels", response_model=list[ObjectLabelOut])
def list_object_labels(conn: sqlite3.Connection = Depends(get_conn)):
    rows = conn.execute(
        "SELECT id, name, parent_id FROM object_labels ORDER BY name COLLATE NOCASE")
    return [ObjectLabelOut(id=r["id"], name=r["name"], parent_id=r["parent_id"],
                           path=_label_path(conn, r["id"])) for r in rows]


@router.get("/annotations/{annotation_id}/mask")
def get_annotation_mask(annotation_id: PathId,
                        conn: sqlite3.Connection = Depends(get_conn)):
    row = conn.execute(
        "SELECT png FROM annotation_masks WHERE annotation_id = ?",
        (annotation_id,)).fetchone()
    if row is None:
        exists = conn.execute(
            "SELECT 1 FROM annotations WHERE id = ?", (annotation_id,)).fetchone()
        if exists is None:
            raise HTTPException(404, "Annotation not found")
        raise HTTPException(404, "Annotation has no mask")
    return Response(content=row["png"], media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@router.delete("/annotations/{annotation_id}")
def delete_annotation(annotation_id: PathId,
                      conn: sqlite3.Connection = Depends(get_conn)):
    cur = conn.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
    conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "Annotation not found")
    return {"ok": True}
