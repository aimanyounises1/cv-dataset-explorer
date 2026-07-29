"""Promptable SAM2 previews and human-approved mask annotations."""
import base64
import hashlib
import io
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from PIL import Image as PILImage

from .. import config
from ..ml import segment as segment_ml
from ..proposal_tokens import (
    ProposalTokenError,
    SegmentPreviewTokenError,
    issue_segment_preview,
    resolve_detection_proposal,
    resolve_segment_preview,
)
from ..schemas import (
    AnnotationOut,
    ModelCapabilityStatus,
    SegmentAcceptRequest,
    SegmentBox,
    SegmentPreview,
    SegmentPrompt,
    SegmentRequest,
)
from .annotations import (
    MAX_PER_SAMPLE,
    _require_sample,
    _row_out,
    canonical_object_name,
    ensure_object_label,
    lookup_object_label,
)
from .deps import PathId, get_conn

router = APIRouter()

_BENCHMARKED_MODEL = "facebook/sam2.1-hiera-tiny"
_BENCHMARKED_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
_BENCHMARK_MEASUREMENT = "~72 ms/mask warm on the reference M4 Max (MPS)"


def _load_original(conn: sqlite3.Connection, sample_id: int):
    row = conn.execute(
        "SELECT filename FROM samples WHERE id = ?", (sample_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Sample not found")
    try:
        source_bytes = (config.IMAGES_DIR / row["filename"]).read_bytes()
        image = PILImage.open(io.BytesIO(source_bytes)).convert("RGB")
        return image, hashlib.sha256(source_bytes).hexdigest()
    except OSError as exc:
        raise HTTPException(
            503, f"Image file unreadable: {row['filename']}") from exc


def _prompt_dict(body: SegmentRequest | SegmentAcceptRequest) -> dict:
    return {
        "points": [point.model_dump() for point in body.points],
        "box": body.box.model_dump() if body.box is not None else None,
    }


def _run_segment(image, body: SegmentRequest | SegmentAcceptRequest):
    segmenter = segment_ml.get_segmenter()
    if segmenter is None:
        _, reason = segment_ml.segment_ready()
        raise HTTPException(
            503, reason or "segmenter failed to load — see server log")
    width, height = image.size
    points = [(point.x * width, point.y * height) for point in body.points]
    labels = [point.label for point in body.points]
    box = None
    if body.box is not None:
        box = (
            body.box.x * width,
            body.box.y * height,
            (body.box.x + body.box.w) * width,
            (body.box.y + body.box.h) * height,
        )
    mask, predicted_iou = segmenter.segment(
        image, points=points, labels=labels, box=box)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (height, width):
        raise HTTPException(503, "Segmenter returned a mask with the wrong dimensions")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise HTTPException(422, "Segmenter returned an empty mask; refine the prompt")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bbox = {
        "x": round(x0 / width, 6),
        "y": round(y0 / height, 6),
        "w": round((x1 - x0) / width, 6),
        "h": round((y1 - y0) / height, 6),
    }
    out = io.BytesIO()
    PILImage.fromarray(mask.astype("uint8") * 255, mode="L").save(out, "PNG")
    png = out.getvalue()
    data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    return {
        "mask": mask,
        "png": png,
        "model_id": segmenter.model_id,
        "model_revision": segmenter.revision,
        "predicted_iou": predicted_iou,
        "bbox": bbox,
        "area_fraction": float(mask.mean()),
        "mask_width": width,
        "mask_height": height,
        "mask_data_url": data_url,
    }


def _mask_png_from_data_url(value: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not value.startswith(prefix):
        raise HTTPException(422, "Segment preview mask must be an inline PNG")
    encoded = value[len(prefix):]
    try:
        png = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "Segment preview mask is not valid base64") from exc
    if not png:
        raise HTTPException(422, "Segment preview mask is empty")
    return png


def _preview_label(conn: sqlite3.Connection, body: SegmentRequest):
    if not body.label_name:
        return None, body.parent_name, []
    resolved = lookup_object_label(conn, body.label_name)
    if resolved is not None:
        if body.parent_name:
            requested_parent = lookup_object_label(conn, body.parent_name)
            if (
                requested_parent is None
                or resolved.parent_id != requested_parent.id
            ):
                raise HTTPException(
                    409,
                    f"Object label '{resolved.name}' already has a different parent",
                )
        parent = resolved.path[-2] if len(resolved.path) > 1 else None
        return resolved.name, parent, resolved.path
    name = canonical_object_name(body.label_name)
    if not body.parent_name:
        return name, None, [name]
    parent = lookup_object_label(conn, body.parent_name)
    if parent is not None:
        return name, parent.name, [*parent.path, name]
    parent_name = canonical_object_name(body.parent_name)
    return name, parent_name, [parent_name, name]


@router.get("/segment/status", response_model=ModelCapabilityStatus)
def segment_status():
    state = segment_ml.segment_availability()
    measured = (
        _BENCHMARK_MEASUREMENT
        if (
            state.model == _BENCHMARKED_MODEL
            and state.revision == _BENCHMARKED_REVISION
        )
        else "not measured for the configured segmenter artifact"
    )
    return {
        "ready": state.ready,
        "reason": state.reason,
        "model": state.model,
        "revision": state.revision,
        "measured": measured,
    }


@router.post("/segment", response_model=SegmentPreview)
def preview_segment(body: SegmentRequest,
                    conn: sqlite3.Connection = Depends(get_conn)):
    image, source_sha256 = _load_original(conn, body.sample_id)
    result = _run_segment(image, body)
    prompt = SegmentPrompt.model_validate(_prompt_dict(body))
    mask_sha256 = hashlib.sha256(result["png"]).hexdigest()
    preview_token = issue_segment_preview(
        sample_id=body.sample_id,
        source_sha256=source_sha256,
        prompt=prompt,
        mask_png=result["png"],
        mask_width=result["mask_width"],
        mask_height=result["mask_height"],
        model_id=result["model_id"],
        model_revision=result["model_revision"],
        predicted_iou=result["predicted_iou"],
        bbox=SegmentBox.model_validate(result["bbox"]),
        area_fraction=result["area_fraction"],
    )
    label_name, parent_name, label_path = _preview_label(conn, body)
    return SegmentPreview(
        sample_id=body.sample_id,
        model=result["model_id"],
        model_revision=result["model_revision"],
        preview_token=preview_token,
        source_sha256=source_sha256,
        mask_sha256=mask_sha256,
        prompt=prompt,
        predicted_iou=round(result["predicted_iou"], 4),
        bbox=result["bbox"],
        area_fraction=round(result["area_fraction"], 6),
        mask_width=result["mask_width"],
        mask_height=result["mask_height"],
        mask_data_url=result["mask_data_url"],
        label_name=label_name,
        parent_name=parent_name,
        label_path=label_path,
    )


@router.get("/samples/{sample_id}/segment-annotations",
            response_model=list[AnnotationOut])
def list_segment_annotations(
    sample_id: PathId,
    conn: sqlite3.Connection = Depends(get_conn),
):
    _require_sample(conn, sample_id)
    rows = conn.execute(
        "SELECT * FROM annotations WHERE sample_id = ? AND kind = 'mask' ORDER BY id",
        (sample_id,))
    return [_row_out(conn, row) for row in rows]


@router.post("/samples/{sample_id}/segment-annotations",
             response_model=AnnotationOut, status_code=201)
def accept_segment_annotation(
    sample_id: PathId,
    body: SegmentAcceptRequest,
    conn: sqlite3.Connection = Depends(get_conn),
):
    _, source_sha256 = _load_original(conn, sample_id)
    count = conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE sample_id = ?",
        (sample_id,)).fetchone()[0]
    if count >= MAX_PER_SAMPLE:
        raise HTTPException(
            400, f"This sample already has {count} annotations. "
                 f"The limit is {MAX_PER_SAMPLE}.")
    proposal_source = None
    if body.proposal_token is not None:
        try:
            proposal_source = resolve_detection_proposal(
                body.proposal_token,
                sample_id=sample_id,
                prompt_box=body.box,
            )
        except ProposalTokenError as exc:
            raise HTTPException(
                422,
                f"{exc}. Run the detector again before accepting this mask.",
            ) from exc
    prompt_model = SegmentPrompt.model_validate(_prompt_dict(body))
    if body.preview_token is None and body.mask_data_url is None:
        raise HTTPException(
            422,
            "Generate and review a segment preview before accepting this annotation",
        )
    if (body.preview_token is None) != (body.mask_data_url is None):
        raise HTTPException(
            422,
            "Segment acceptance requires both preview_token and mask_data_url",
        )
    assert body.preview_token is not None and body.mask_data_url is not None
    mask_png = _mask_png_from_data_url(body.mask_data_url)
    try:
        evidence = resolve_segment_preview(
            body.preview_token,
            sample_id=sample_id,
            prompt=prompt_model,
            source_sha256=source_sha256,
            mask_png=mask_png,
        )
    except SegmentPreviewTokenError as exc:
        raise HTTPException(
            422,
            f"{exc}. Generate the mask again before accepting it.",
        ) from exc
    result = {
        "png": mask_png,
        "mask_width": evidence.mask_width,
        "mask_height": evidence.mask_height,
        "model_id": evidence.model_id,
        "model_revision": evidence.model_revision,
        "predicted_iou": evidence.predicted_iou,
        "bbox": evidence.bbox.model_dump(mode="json"),
    }
    prompt = prompt_model.model_dump(mode="json")
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        label = ensure_object_label(
            conn, body.label_name, body.parent_name)
        cur = conn.execute(
            "INSERT INTO annotations(sample_id, kind, geometry, label, created_at) "
            "VALUES (?, 'mask', ?, ?, ?)",
            (sample_id, json.dumps(result["bbox"], separators=(",", ":")),
             label.name, created_at))
        annotation_id = cur.lastrowid
        conn.execute(
            "INSERT INTO annotation_masks("
            "annotation_id, png, width, height, model_id, model_revision, "
            "prompt_json, proposal_json, predicted_iou"
            ") VALUES (?,?,?,?,?,?,?,?,?)",
            (annotation_id, result["png"], result["mask_width"],
             result["mask_height"], result["model_id"],
             result["model_revision"],
             json.dumps(prompt, separators=(",", ":")),
             (
                 proposal_source.model_dump_json()
                 if proposal_source is not None
                 else None
             ),
             result["predicted_iou"]))
        conn.execute(
            "INSERT INTO annotation_object_labels(annotation_id, label_id) "
            "VALUES (?, ?)", (annotation_id, label.id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    row = conn.execute(
        "SELECT * FROM annotations WHERE id = ?", (annotation_id,)).fetchone()
    return _row_out(conn, row)


@router.delete("/segment-annotations/{annotation_id}")
def delete_segment_annotation(
    annotation_id: PathId,
    conn: sqlite3.Connection = Depends(get_conn),
):
    cur = conn.execute(
        "DELETE FROM annotations WHERE id = ? AND kind = 'mask'",
        (annotation_id,))
    conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "Segment annotation not found")
    return {"ok": True}
