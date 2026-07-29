"""Server-issued evidence tokens for short-lived vision proposals.

Detection boxes and segmentation masks are drafts, so persisting every preview
would add database state that has no value after the editor closes. Instead the
server returns authenticated, self-contained evidence. Acceptance verifies the
token and persists only the exact proposal the reviewer saw; model identity,
geometry, scores, and mask bytes are therefore never trusted from
client-authored JSON.

The signing key is intentionally process-local. A proposal visible across a
backend restart must be regenerated, while an already accepted annotation keeps
its resolved provenance and pixels in SQLite.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .schemas import DetectionProposalSource, SegmentBox, SegmentPrompt

TOKEN_VERSION = 1
TOKEN_TTL_SECONDS = 30 * 60
_CLOCK_SKEW_SECONDS = 30
_SIGNING_KEY = secrets.token_bytes(32)


class ProposalTokenError(ValueError):
    """The token is invalid, expired, or does not describe this mask prompt."""


class SegmentPreviewTokenError(ValueError):
    """The token is invalid, expired, or does not bind these preview bytes."""


class SegmentPreviewEvidence(BaseModel):
    """Authenticated evidence recovered from one reviewed SAM preview."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["segment_preview"] = "segment_preview"
    version: int
    issued_at: int
    sample_id: int = Field(..., ge=1, le=2**63 - 1)
    source_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    prompt: SegmentPrompt
    mask_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    mask_width: int = Field(..., ge=1, le=50_000)
    mask_height: int = Field(..., ge=1, le=50_000)
    model_id: str = Field(..., min_length=1, max_length=200)
    model_revision: str = Field(..., min_length=1, max_length=200)
    predicted_iou: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    bbox: SegmentBox
    area_fraction: float = Field(..., gt=0.0, le=1.0, allow_inf_nan=False)


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(
    value: str,
    *,
    error_type: type[ValueError] = ProposalTokenError,
    label: str = "detector proposal",
) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError) as exc:
        raise error_type(f"invalid {label} token") from exc


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _issue(payload: dict[str, Any]) -> str:
    encoded_payload = _canonical(payload)
    signature = hmac.new(
        _SIGNING_KEY,
        encoded_payload,
        hashlib.sha256,
    ).digest()
    return f"{_encode(encoded_payload)}.{_encode(signature)}"


def _authenticate(
    token: str,
    *,
    error_type: type[ValueError],
    label: str,
) -> dict[str, Any]:
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
    except ValueError as exc:
        raise error_type(f"invalid {label} token") from exc

    payload_bytes = _decode(
        encoded_payload,
        error_type=error_type,
        label=label,
    )
    signature = _decode(
        encoded_signature,
        error_type=error_type,
        label=label,
    )
    expected = hmac.new(
        _SIGNING_KEY,
        payload_bytes,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(signature, expected):
        raise error_type(f"invalid {label} token")
    try:
        payload = json.loads(payload_bytes)
    except (TypeError, ValueError) as exc:
        raise error_type(f"invalid {label} token") from exc
    if not isinstance(payload, dict):
        raise error_type(f"invalid {label} token")
    return payload


def issue_detection_proposal(
    sample_id: int,
    source: DetectionProposalSource,
    *,
    issued_at: int | None = None,
) -> str:
    """Return one opaque token binding a proposal to its sample and evidence."""
    payload = {
        "version": TOKEN_VERSION,
        "issued_at": int(time.time()) if issued_at is None else issued_at,
        "sample_id": sample_id,
        "source": source.model_dump(mode="json"),
    }
    return _issue(payload)


def resolve_detection_proposal(
    token: str,
    *,
    sample_id: int,
    prompt_box: SegmentBox | None,
    now: int | None = None,
) -> DetectionProposalSource:
    """Authenticate and resolve evidence for one accepted detector-origin mask."""
    payload = _authenticate(
        token,
        error_type=ProposalTokenError,
        label="detector proposal",
    )
    try:
        version = payload["version"]
        issued_at = payload["issued_at"]
        token_sample_id = payload["sample_id"]
        source = DetectionProposalSource.model_validate(payload["source"])
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise ProposalTokenError("invalid detector proposal token") from exc

    current = int(time.time()) if now is None else now
    if version != TOKEN_VERSION:
        raise ProposalTokenError("unsupported detector proposal token")
    if not isinstance(issued_at, int):
        raise ProposalTokenError("invalid detector proposal timestamp")
    if issued_at > current + _CLOCK_SKEW_SECONDS:
        raise ProposalTokenError("invalid detector proposal timestamp")
    if current - issued_at > TOKEN_TTL_SECONDS:
        raise ProposalTokenError("detector proposal expired")
    if token_sample_id != sample_id:
        raise ProposalTokenError("detector proposal belongs to another sample")
    if prompt_box is None or source.box != prompt_box:
        raise ProposalTokenError(
            "detector proposal box does not match the mask prompt"
        )
    return source


def issue_segment_preview(
    *,
    sample_id: int,
    source_sha256: str,
    prompt: SegmentPrompt,
    mask_png: bytes,
    mask_width: int,
    mask_height: int,
    model_id: str,
    model_revision: str,
    predicted_iou: float,
    bbox: SegmentBox,
    area_fraction: float,
    issued_at: int | None = None,
) -> str:
    """Bind one displayed SAM mask to its source, prompt, and model evidence."""
    evidence = SegmentPreviewEvidence(
        version=TOKEN_VERSION,
        issued_at=int(time.time()) if issued_at is None else issued_at,
        sample_id=sample_id,
        source_sha256=source_sha256,
        prompt=prompt,
        mask_sha256=hashlib.sha256(mask_png).hexdigest(),
        mask_width=mask_width,
        mask_height=mask_height,
        model_id=model_id,
        model_revision=model_revision,
        predicted_iou=predicted_iou,
        bbox=bbox,
        area_fraction=area_fraction,
    )
    return _issue(evidence.model_dump(mode="json"))


def resolve_segment_preview(
    token: str,
    *,
    sample_id: int,
    prompt: SegmentPrompt,
    source_sha256: str,
    mask_png: bytes,
    now: int | None = None,
) -> SegmentPreviewEvidence:
    """Authenticate the exact displayed mask before it becomes an annotation."""
    payload = _authenticate(
        token,
        error_type=SegmentPreviewTokenError,
        label="segment preview",
    )
    try:
        evidence = SegmentPreviewEvidence.model_validate(payload)
    except ValidationError as exc:
        raise SegmentPreviewTokenError("invalid segment preview token") from exc

    current = int(time.time()) if now is None else now
    if evidence.version != TOKEN_VERSION:
        raise SegmentPreviewTokenError("unsupported segment preview token")
    if evidence.issued_at > current + _CLOCK_SKEW_SECONDS:
        raise SegmentPreviewTokenError("invalid segment preview timestamp")
    if current - evidence.issued_at > TOKEN_TTL_SECONDS:
        raise SegmentPreviewTokenError("segment preview expired")
    if evidence.sample_id != sample_id:
        raise SegmentPreviewTokenError("segment preview belongs to another sample")
    if evidence.prompt != prompt:
        raise SegmentPreviewTokenError("segment preview prompt does not match")
    if evidence.source_sha256 != source_sha256:
        raise SegmentPreviewTokenError("source image changed after the segment preview")
    if evidence.mask_sha256 != hashlib.sha256(mask_png).hexdigest():
        raise SegmentPreviewTokenError("mask bytes do not match the segment preview")
    return evidence
