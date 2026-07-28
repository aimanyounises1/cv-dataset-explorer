"""Region suggestions: zero-shot detection over one sample's original image.

Boxes are proposals for the region-search flow — the user clicks one and it
becomes positive or negative evidence through POST /api/search/by-region.
Detection itself writes nothing and follows the optional-layer contract:
GET /api/detect/status names the enabling command; the POST refuses with the
same reason rather than downloading a model mid-request.
"""
import sqlite3

from fastapi import APIRouter, Depends, HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel, Field

from .. import config
from ..ml import detect as detect_ml
from .annotations import canonical_object_name, lookup_object_label
from .deps import get_conn

router = APIRouter()


@router.get("/detect/status")
def detect_status():
    """Availability probe, so the UI can offer the control only when it works."""
    ok, reason = detect_ml.detect_ready()
    return {"ready": ok, "reason": reason, "model": detect_ml.DETECT_MODEL,
            "measured": "~330 ms/image warm on the reference M4 Max (MPS)"}


class DetectRequest(BaseModel):
    sample_id: int = Field(..., ge=1, le=2**63 - 1)
    # Period-separated phrases, per the model's own query format.
    queries: str = Field("a person. an animal. a vehicle. an object.",
                         min_length=3, max_length=300)


@router.post("/detect")
def detect_regions(body: DetectRequest,
                   conn: sqlite3.Connection = Depends(get_conn)):
    detector = detect_ml.get_detector()
    if detector is None:
        _, reason = detect_ml.detect_ready()
        raise HTTPException(503, reason or "detector failed to load — see server log")
    row = conn.execute("SELECT filename FROM samples WHERE id = ?",
                       (body.sample_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Sample not found")
    try:
        img = PILImage.open(config.IMAGES_DIR / row["filename"]).convert("RGB")
    except OSError as exc:
        raise HTTPException(503, f"Image file unreadable: {row['filename']}") from exc
    boxes = detector.detect(img, body.queries)
    for box in boxes:
        resolved = lookup_object_label(conn, box["label"])
        box["label_name"] = (resolved.name if resolved is not None
                             else canonical_object_name(box["label"]))
        box["parent_name"] = (
            resolved.path[-2] if resolved is not None and len(resolved.path) > 1
            else None)
        box["label_path"] = resolved.path if resolved is not None else []
    return {"sample_id": body.sample_id, "model": detect_ml.DETECT_MODEL,
            "queries": body.queries,
            "boxes": boxes,
            "note": "zero-shot proposals — click one to use it as region "
                    "evidence; scores are detector confidences, not "
                    "retrieval scores"}
