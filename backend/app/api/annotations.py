"""Annotations: regions drawn over a sample.

Rows, never pixels — the source images stay immutable, and an annotation is a
record about an image, not a change to it. Geometry is normalized 0..1
coordinates (validated in `AnnotationCreate`) so a region survives any
rendered size. Mounted under /api by main.py.
"""
import hashlib
import io
import json
import sqlite3
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone

import PIL
from fastapi import APIRouter, Depends, HTTPException, Response
from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from pydantic import ValidationError

from .. import config
from ..schemas import (
    AnnotationCreate,
    AnnotationOut,
    DetectionProposalSource,
    ObjectLabelOut,
)
from .deps import PathId, get_conn

router = APIRouter()

# More regions than a human draws on one image is a runaway client, and an
# unbounded list would eventually be a payload problem for the sample page.
MAX_PER_SAMPLE = 200


@dataclass(frozen=True)
class _AcceptedMaskArtifacts:
    row: sqlite3.Row
    annotation: AnnotationOut
    source_bytes: bytes
    source_width: int
    source_height: int
    source_mode: str
    source_format: str | None
    mask_png: bytes
    cutout_png: bytes
    bbox: tuple[int, int, int, int]


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
        "SELECT width, height, model_id, model_revision, prompt_json, "
        "proposal_json, predicted_iou "
        "FROM annotation_masks WHERE annotation_id = ?", (r["id"],)).fetchone()
    label = conn.execute(
        "SELECT l.id, l.name, l.parent_id FROM annotation_object_labels al "
        "JOIN object_labels l ON l.id = al.label_id WHERE al.annotation_id = ?",
        (r["id"],)).fetchone()
    label_name = label["name"] if label else None
    label_path = _label_path(conn, label["id"]) if label else []
    prompt = None
    proposal_source = None
    if mask is not None:
        try:
            prompt = json.loads(mask["prompt_json"])
        except ValueError:
            prompt = None
        if mask["proposal_json"]:
            try:
                proposal_source = DetectionProposalSource.model_validate_json(
                    mask["proposal_json"],
                ).model_dump()
            except ValidationError as exc:
                raise HTTPException(
                    503,
                    "Persisted detector proposal provenance is invalid",
                ) from exc
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
                         cutout_url=(f"/api/annotations/{r['id']}/cutout"
                                      if mask is not None else None),
                         artifact_package_url=(
                             f"/api/annotations/{r['id']}/export"
                             if mask is not None else None
                         ),
                         mask_width=mask["width"] if mask else None,
                         mask_height=mask["height"] if mask else None,
                         model_id=mask["model_id"] if mask else None,
                         model_revision=(
                             mask["model_revision"] if mask else None),
                         prompt=prompt,
                         proposal_source=proposal_source,
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


def _accepted_mask_artifacts(
    conn: sqlite3.Connection,
    annotation_id: int,
) -> _AcceptedMaskArtifacts:
    """Decode one accepted mask and derive its transparent object cutout.

    The source stays immutable. The returned cutout is a new RGBA PNG whose
    alpha channel is the accepted mask, cropped to the mask's non-zero bounds.
    """
    owns_snapshot = not conn.in_transaction
    if owns_snapshot:
        # A plain SELECT does not retain a SQLite snapshot. Keep every mask,
        # taxonomy and model-provenance read in one explicit read transaction so
        # a concurrent delete cannot mix old pixels with a newer manifest.
        conn.execute("BEGIN")
    try:
        row = conn.execute(
            "SELECT a.*, s.filename, s.split, "
            "m.png AS mask_png, m.width AS stored_mask_width, "
            "m.height AS stored_mask_height "
            "FROM annotations a "
            "JOIN samples s ON s.id = a.sample_id "
            "LEFT JOIN annotation_masks m ON m.annotation_id = a.id "
            "WHERE a.id = ?",
            (annotation_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "Annotation not found")
        if row["mask_png"] is None:
            raise HTTPException(404, "Annotation has no accepted mask")
        if row["kind"] != "mask":
            raise HTTPException(503, "Accepted mask has an invalid annotation kind")

        root = config.IMAGES_DIR.resolve()
        source_path = (root / row["filename"]).resolve()
        if not source_path.is_relative_to(root):
            raise HTTPException(
                500,
                "Sample image path escapes the configured image directory",
            )
        try:
            source_bytes = source_path.read_bytes()
        except OSError as exc:
            raise HTTPException(
                503,
                f"Image file unreadable: {row['filename']}",
            ) from exc
        if not source_bytes:
            raise HTTPException(503, f"Image file is empty: {row['filename']}")

        mask_bytes = bytes(row["mask_png"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", PILImage.DecompressionBombWarning)
                with PILImage.open(io.BytesIO(source_bytes)) as source_check:
                    source_check.verify()
                with PILImage.open(io.BytesIO(source_bytes)) as source_image:
                    source_image.load()
                    source_width, source_height = source_image.size
                    source_mode = source_image.mode
                    source_format = source_image.format
                    source = source_image.convert("RGBA")
                with PILImage.open(io.BytesIO(mask_bytes)) as mask_check:
                    if mask_check.format != "PNG" or mask_check.mode not in {"1", "L"}:
                        raise HTTPException(
                            503,
                            "Accepted mask is not a one-channel PNG artifact",
                        )
                    mask_check.verify()
                with PILImage.open(io.BytesIO(mask_bytes)) as mask_image:
                    mask_image.load()
                    mask = mask_image.convert("L")
        except (
            PILImage.DecompressionBombError,
            PILImage.DecompressionBombWarning,
        ) as exc:
            raise HTTPException(
                503,
                "Source image or accepted mask exceeds Pillow's safe decode limit",
            ) from exc
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise HTTPException(
                503,
                "Source image or accepted mask failed integrity/decode checks",
            ) from exc

        stored_size = (row["stored_mask_width"], row["stored_mask_height"])
        if mask.size != stored_size or source.size != mask.size:
            raise HTTPException(
                503,
                "Accepted mask dimensions do not match its source image",
            )
        if any(mask.histogram()[1:255]):
            raise HTTPException(503, "Accepted mask is not binary")
        bbox = mask.getbbox()
        if bbox is None:
            raise HTTPException(503, "Accepted mask contains no foreground pixels")

        cutout = source.crop(bbox)
        cutout.putalpha(mask.crop(bbox))
        cutout.info.clear()
        cutout_file = io.BytesIO()
        cutout.save(cutout_file, "PNG")
        return _AcceptedMaskArtifacts(
            row=row,
            annotation=_row_out(conn, row),
            source_bytes=source_bytes,
            source_width=source_width,
            source_height=source_height,
            source_mode=source_mode,
            source_format=source_format,
            mask_png=mask_bytes,
            cutout_png=cutout_file.getvalue(),
            bbox=bbox,
        )
    finally:
        if owns_snapshot and conn.in_transaction:
            conn.rollback()


@router.get("/annotations/{annotation_id}/cutout")
def get_annotation_cutout(
    annotation_id: PathId,
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Return the accepted object as a tightly cropped transparent PNG."""
    artifacts = _accepted_mask_artifacts(conn, annotation_id)
    filename = (
        f"cvde-sample-{artifacts.row['sample_id']}-"
        f"annotation-{annotation_id}-cutout.png"
    )
    return Response(
        content=artifacts.cutout_png,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/annotations/{annotation_id}/export")
def export_annotation_package(
    annotation_id: PathId,
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Export one atomic, integrity-linked mask/cutout review package."""
    artifacts = _accepted_mask_artifacts(
        conn,
        annotation_id,
    )
    row = artifacts.row
    source_bytes = artifacts.source_bytes
    mask_png = artifacts.mask_png
    cutout_png = artifacts.cutout_png
    base_name = f"cvde-sample-{row['sample_id']}-annotation-{annotation_id}"
    mask_filename = f"{base_name}-mask.png"
    cutout_filename = f"{base_name}-cutout.png"
    manifest_filename = f"{base_name}-manifest.json"
    left, upper, right, lower = artifacts.bbox
    annotation_record = artifacts.annotation.model_dump(exclude={"mask_data_url"})
    manifest = {
        "format": "cvde.segment-annotation-export",
        "version": 3,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_image": {
            "sample_id": row["sample_id"],
            "filename": row["filename"],
            "split": row["split"],
            "byte_length": len(source_bytes),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "width": artifacts.source_width,
            "height": artifacts.source_height,
            "mode": artifacts.source_mode,
            "format": artifacts.source_format,
        },
        "annotation": annotation_record,
        "derivation": {
            "library": "Pillow",
            "library_version": PIL.__version__,
            "operations": ["Image.getbbox", "Image.crop", "Image.putalpha"],
            "cutout_bbox_pixels": {
                "left": left,
                "upper": upper,
                "right": right,
                "lower": lower,
            },
        },
        "artifacts": {
            "mask": {
                "filename": mask_filename,
                "media_type": "image/png",
                "byte_length": len(mask_png),
                "sha256": hashlib.sha256(mask_png).hexdigest(),
                "width": row["stored_mask_width"],
                "height": row["stored_mask_height"],
            },
            "cutout": {
                "filename": cutout_filename,
                "media_type": "image/png",
                "byte_length": len(cutout_png),
                "sha256": hashlib.sha256(cutout_png).hexdigest(),
                "width": right - left,
                "height": lower - upper,
            },
        },
    }

    package = io.BytesIO()
    with zipfile.ZipFile(
        package,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(mask_filename, mask_png)
        archive.writestr(cutout_filename, cutout_png)
        archive.writestr(
            manifest_filename,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    return Response(
        content=package.getvalue(),
        media_type="application/zip",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'attachment; filename="{base_name}.zip"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.delete("/annotations/{annotation_id}")
def delete_annotation(annotation_id: PathId,
                      conn: sqlite3.Connection = Depends(get_conn)):
    cur = conn.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
    conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "Annotation not found")
    return {"ok": True}
