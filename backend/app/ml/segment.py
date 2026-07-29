"""Optional promptable object segmentation through Transformers SAM2.

The implementation is the production form of ``scripts/bench_sam2.py``:
official ``Sam2Processor`` point/label and box inputs, official
``post_process_masks``, one lazy model, and one inference lock. Model loading is
local-only so an HTTP request can never start a checkpoint download.
"""
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .. import config
from . import providers
from .providers import ModelSnapshot

logger = logging.getLogger(__name__)

SEGMENT_MODEL = config.SEGMENT_MODEL
SEGMENT_REVISION = config.SEGMENT_REVISION
FETCH_HINT = ("python -c \"from huggingface_hub import snapshot_download; "
              f"snapshot_download(repo_id={SEGMENT_MODEL!r}, "
              f"revision={SEGMENT_REVISION!r})\"")

_lock = threading.Lock()
_segmenter = None
_failed_at: Optional[float] = None
_failed_reason: Optional[str] = None
_RETRY_AFTER_S = 120.0


@dataclass(frozen=True)
class SegmentAvailability:
    ready: bool
    reason: str | None
    model: str
    revision: str | None
    snapshot: ModelSnapshot | None = None


def _resolve_snapshot(requested_revision: str | None = None) -> ModelSnapshot:
    """Resolve the configured immutable snapshot without network access."""
    requested = requested_revision or configured_revision()
    snapshot = providers.resolve_model_snapshot(
        SEGMENT_MODEL,
        revision=requested,
        local_files_only=True,
    )
    if snapshot.revision != requested:
        raise RuntimeError(
            f"resolved segmenter commit {snapshot.revision} does not match "
            f"CVDE_SEGMENT_REVISION={requested}")
    return snapshot


def configured_revision() -> str:
    """Return the configured SAM2 commit, rejecting a moving selector."""
    return providers.require_full_commit_revision(
        SEGMENT_REVISION,
        "CVDE_SEGMENT_REVISION",
    )


def segment_availability() -> SegmentAvailability:
    """One truthful, atomic readiness probe with resolved provenance."""
    if _segmenter is not None:
        return SegmentAvailability(
            True,
            None,
            _segmenter.model_id,
            _segmenter.revision,
        )
    try:
        revision = configured_revision()
    except ValueError as exc:
        return SegmentAvailability(
            False,
            str(exc),
            SEGMENT_MODEL,
            None,
        )
    if _failed_at is not None:
        remaining = _RETRY_AFTER_S - (time.monotonic() - _failed_at)
        if remaining > 0:
            return SegmentAvailability(
                False,
                _failed_reason
                or f"segmenter load failed; retry in {remaining:.0f} seconds",
                SEGMENT_MODEL,
                revision,
            )
    try:
        import torch  # noqa: F401
        import transformers
    except ImportError:
        return SegmentAvailability(
            False,
            "torch/transformers not installed — the base requirements provide them",
            SEGMENT_MODEL,
            revision,
        )
    if not hasattr(transformers, "Sam2Model") or not hasattr(transformers, "Sam2Processor"):
        return SegmentAvailability(
            False,
            f"transformers {transformers.__version__} has no SAM2 image "
            "segmentation API",
            SEGMENT_MODEL,
            revision,
        )
    try:
        snapshot = _resolve_snapshot(revision)
    except Exception as exc:  # noqa: BLE001 - the reason is returned, not hidden
        return SegmentAvailability(
            False,
            f"SAM2 snapshot {SEGMENT_MODEL}@{SEGMENT_REVISION} is not "
            f"available locally ({exc}) — run: {FETCH_HINT}",
            SEGMENT_MODEL,
            revision,
        )
    return SegmentAvailability(
        True,
        None,
        snapshot.model_id,
        snapshot.revision,
        snapshot,
    )


def segment_ready() -> tuple[bool, Optional[str]]:
    """Compatibility wrapper for callers that only need readiness and reason."""
    state = segment_availability()
    return state.ready, state.reason


class Segmenter:
    """One local SAM2 image model with serialized forwards."""

    def __init__(self, snapshot=None):
        import torch
        from transformers import Sam2Model, Sam2Processor

        from .embedder import _pick_device

        snapshot = snapshot or _resolve_snapshot()
        self.device = _pick_device()
        self.model_id = snapshot.model_id
        self.revision = snapshot.revision
        model_path = str(snapshot.snapshot_path)
        logger.info(
            "Loading %s@%s on %s",
            self.model_id,
            self.revision,
            self.device,
        )
        self.processor = Sam2Processor.from_pretrained(
            model_path, local_files_only=True)
        self.model, loading = Sam2Model.from_pretrained(
            model_path, local_files_only=True, output_loading_info=True)
        problems = {name: loading.get(name) or ()
                    for name in ("missing_keys", "unexpected_keys", "mismatched_keys")}
        if any(problems.values()):
            raise RuntimeError(f"SAM2 checkpoint does not match Sam2Model: {problems}")
        self.model = self.model.to(self.device).eval()
        self._torch = torch
        self._infer = threading.Lock()

    def segment(
        self,
        image,
        *,
        points: list[tuple[float, float]] | None = None,
        labels: list[int] | None = None,
        box: tuple[float, float, float, float] | None = None,
    ):
        """Return ``(full_resolution_bool_mask, predicted_iou)`` for one object.

        Coordinates are pixel-space because that is the processor's documented
        contract. REST schemas keep normalized coordinates and convert them
        after loading the immutable original image.
        """
        prompt = {}
        if box is not None:
            prompt["input_boxes"] = [[list(box)]]
        if points:
            if labels is None or len(points) != len(labels):
                raise ValueError("each SAM2 point needs one point label")
            # batch -> object -> points -> (x, y)
            prompt["input_points"] = [[[list(point) for point in points]]]
            prompt["input_labels"] = [[list(labels)]]
        with self._infer, self._torch.no_grad():
            inputs = self.processor(
                images=image, return_tensors="pt", **prompt).to(self.device)
            result = self.model(**inputs, multimask_output=False)
            masks = self.processor.post_process_masks(
                result.pred_masks.cpu(), inputs["original_sizes"])[0]
        mask = masks[0, 0].numpy().astype(bool)
        score = float(result.iou_scores.detach().float().cpu().reshape(-1)[0])
        return mask, score


def get_segmenter() -> Optional[Segmenter]:
    """Thread-safe lazy singleton, with a cooldown after a failed load."""
    global _segmenter, _failed_at, _failed_reason
    if _segmenter is not None:
        return _segmenter
    availability = segment_availability()
    if not availability.ready or availability.snapshot is None:
        return None
    with _lock:
        if _segmenter is not None:
            return _segmenter
        if _failed_at is not None and time.monotonic() - _failed_at < _RETRY_AFTER_S:
            return None
        try:
            _segmenter = Segmenter(availability.snapshot)
            _failed_at = None
            _failed_reason = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Segmenter unavailable: %s", exc)
            _failed_at = time.monotonic()
            _failed_reason = f"segmenter load failed: {exc}"
            return None
    return _segmenter
