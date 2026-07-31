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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .. import config
from ..ml import detect as detect_ml
from ..proposal_tokens import issue_detection_proposal
from ..schemas import (
    DetectionProposalSource,
    DetectResponse,
    ModelCapabilityStatus,
    SegmentBox,
)
from .annotations import canonical_object_name, lookup_object_label
from .deps import get_conn

router = APIRouter()

_BENCHMARKED_MODEL = "IDEA-Research/grounding-dino-tiny"
_BENCHMARKED_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
_BENCHMARK_MEASUREMENT = "~330 ms/image warm on the reference M4 Max (MPS)"
AUTO_DETECT_QUERIES = "a person. an animal. a vehicle. an object."


@router.get("/detect/status", response_model=ModelCapabilityStatus)
def detect_status():
    """Availability probe, so the UI can offer the control only when it works."""
    state = detect_ml.detector_availability()
    measured = (
        _BENCHMARK_MEASUREMENT
        if (
            state.model == _BENCHMARKED_MODEL
            and state.revision == _BENCHMARKED_REVISION
        )
        else "not measured for the configured detector artifact"
    )
    return {"ready": state.ready, "reason": state.reason, "model": state.model,
            "revision": state.revision,
            "measured": measured}


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: int = Field(..., strict=True, ge=1, le=2**63 - 1)
    # Period-separated phrases, per the model's own query format.
    queries: str = Field(AUTO_DETECT_QUERIES, max_length=300)

    @field_validator("queries")
    @classmethod
    def _normalize_queries(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            return AUTO_DETECT_QUERIES
        if len(normalized) < 3:
            raise ValueError("queries must contain a detector phrase")
        # Periods only separate phrases; input like "..." carries none.
        if not detect_ml.phrases_from(normalized):
            raise ValueError(
                "queries must contain at least one phrase between periods")
        return normalized


@router.post("/detect", response_model=DetectResponse)
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
        source = DetectionProposalSource(
            model_id=detector.model_id,
            model_revision=detector.revision,
            queries=body.queries,
            original_label=box["label"],
            proposed_label=box["label_name"],
            score=box["score"],
            box=SegmentBox(
                x=box["x"],
                y=box["y"],
                w=box["w"],
                h=box["h"],
            ),
        )
        box["proposal_token"] = issue_detection_proposal(
            body.sample_id,
            source,
        )
    return {"sample_id": body.sample_id, "model": detector.model_id,
            "revision": detector.revision,
            "queries": body.queries,
            "boxes": boxes,
            "note": "zero-shot proposals — click one to use it as region "
                    "evidence; scores rank phrase alignment within this "
                    "detector run and are neither calibrated probabilities "
                    "nor retrieval scores"}
