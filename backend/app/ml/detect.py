"""Optional zero-shot object detection: Grounding DINO tiny.

Measured on this machine (M4 Max, MPS) before integration, per the rule that
a model ships with numbers or not at all: ~330 ms per image warm (p50 over a
30-image sample; 267 ms fastest, 448 ms p95), ~1 GB resident, correct boxes
("a person", "a dog") on a real corpus image; cold load ~6 s once the weights
are cached, plus a one-time ~700 MB download. `scripts/bench_detector.py`
re-measures every number in this paragraph. Grounding DINO proposals can feed
the separately measured SAM 2.1 editor; masks are exposed for refinement,
annotation and explicit object search, never presented as an automatic ranking
improvement. ``scripts/bench_sam2.py`` records that product boundary.

Same optional-layer contract as every ML capability: lazy singleton, its own
inference lock (Metal cannot run one module concurrently), a cheap readiness
probe that names the enabling command, and NO downloads on the request path —
weights are fetched explicitly or not at all.
"""
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .. import config
from . import providers
from .providers import ModelSnapshot

logger = logging.getLogger(__name__)

DETECT_MODEL = config.DETECT_MODEL
DETECT_REVISION = config.DETECT_REVISION
FETCH_HINT = ("python -c \"from huggingface_hub import snapshot_download; "
              f"snapshot_download(repo_id={DETECT_MODEL!r}, "
              f"revision={DETECT_REVISION!r})\"")

_lock = threading.Lock()
_detector = None
_failed_at: Optional[float] = None
_failed_reason: Optional[str] = None
_RETRY_AFTER_S = 120.0


@dataclass(frozen=True)
class DetectorAvailability:
    ready: bool
    reason: str | None
    model: str
    revision: str | None
    snapshot: ModelSnapshot | None = None


def _resolve_snapshot(requested_revision: str | None = None) -> ModelSnapshot:
    """Resolve the configured immutable snapshot without network access."""
    requested = requested_revision or configured_revision()
    snapshot = providers.resolve_model_snapshot(
        DETECT_MODEL,
        revision=requested,
        local_files_only=True,
    )
    if snapshot.revision != requested:
        raise RuntimeError(
            f"resolved detector commit {snapshot.revision} does not match "
            f"CVDE_DETECT_REVISION={requested}")
    return snapshot


def configured_revision() -> str:
    """Return the configured detector commit, rejecting a moving selector."""
    return providers.require_full_commit_revision(
        DETECT_REVISION,
        "CVDE_DETECT_REVISION",
    )


def detector_availability() -> DetectorAvailability:
    """One truthful, atomic readiness probe with resolved provenance."""
    if _detector is not None:
        return DetectorAvailability(
            True,
            None,
            _detector.model_id,
            _detector.revision,
        )
    try:
        revision = configured_revision()
    except ValueError as exc:
        return DetectorAvailability(
            False,
            str(exc),
            DETECT_MODEL,
            None,
        )
    if _failed_at is not None:
        remaining = _RETRY_AFTER_S - (time.monotonic() - _failed_at)
        if remaining > 0:
            return DetectorAvailability(
                False,
                _failed_reason
                or f"detector load failed; retry in {remaining:.0f} seconds",
                DETECT_MODEL,
                revision,
            )
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return DetectorAvailability(
            False,
            "torch/transformers not installed — the base requirements provide them",
            DETECT_MODEL,
            revision,
        )
    try:
        snapshot = _resolve_snapshot(revision)
    except Exception as exc:  # noqa: BLE001 - the reason is returned, not hidden
        return DetectorAvailability(
            False,
            f"detector snapshot {DETECT_MODEL}@{DETECT_REVISION} is not "
            f"available locally ({exc}) — run: {FETCH_HINT}",
            DETECT_MODEL,
            revision,
        )
    return DetectorAvailability(
        True,
        None,
        snapshot.model_id,
        snapshot.revision,
        snapshot,
    )


def detect_ready() -> tuple[bool, Optional[str]]:
    """Compatibility wrapper for callers that only need readiness and reason."""
    state = detector_availability()
    return state.ready, state.reason


def phrases_from(queries: str) -> list[str]:
    """Split a period-separated query string into dot-free phrases.

    The installed processor (transformers 5.14.1) applies its documented
    candidate-label normalization — strip, lowercase, ``". ".join(labels)
    + "."`` — only to a list/tuple of strings with no ``.`` inside any item
    (``_is_list_of_candidate_labels`` / ``_merge_candidate_labels_text``).
    A raw string bypasses that path entirely, so natural input like ``"dog"``
    reached the model unnormalized and failed. Splitting here keeps the wire
    format (one period-separated string) while handing the library the list
    form it normalizes itself.
    """
    return [phrase.strip() for phrase in queries.split(".") if phrase.strip()]


class _Detector:
    def __init__(self, snapshot=None):
        import torch
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

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
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True)
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            model_path, local_files_only=True).to(self.device).eval()
        self._torch = torch
        self._infer = threading.Lock()

    def detect(self, image, queries: str, threshold: float = 0.3,
               max_boxes: int = 12) -> list[dict]:
        """Normalized [0..1] boxes for period-separated phrase queries."""
        torch = self._torch
        with self._infer, torch.no_grad():
            # The processor normalizes candidate labels only when given a
            # list of dot-free phrases; a raw string would bypass that path.
            inputs = self.processor(images=image, text=phrases_from(queries),
                                    return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            res = self.processor.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=threshold,
                text_threshold=threshold, target_sizes=[image.size[::-1]])[0]
        W, H = image.size
        boxes = []
        # transformers renamed this key; falling back to unlabelled boxes keeps
        # the count aligned, so a rename can never silently drop detections.
        labels = (res.get("text_labels") or res.get("labels")
                  or [""] * len(res["boxes"]))
        for (x0, y0, x1, y1), label, score in zip(
                res["boxes"].tolist(), labels, res["scores"].tolist(), strict=True):
            coordinates = tuple(float(value) for value in (x0, y0, x1, y1))
            confidence = float(score)
            if (
                not all(math.isfinite(value) for value in coordinates)
                or not math.isfinite(confidence)
            ):
                continue
            clipped_x0 = min(max(coordinates[0], 0.0), float(W))
            clipped_y0 = min(max(coordinates[1], 0.0), float(H))
            clipped_x1 = min(max(coordinates[2], 0.0), float(W))
            clipped_y1 = min(max(coordinates[3], 0.0), float(H))
            if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
                continue
            boxes.append({
                "x": clipped_x0 / W,
                "y": clipped_y0 / H,
                "w": (clipped_x1 - clipped_x0) / W,
                "h": (clipped_y1 - clipped_y0) / H,
                "label": str(label),
                "score": round(confidence, 3),
            })
        boxes.sort(key=lambda b: -b["score"])
        return boxes[:max_boxes]


def get_detector() -> Optional[_Detector]:
    """Thread-safe singleton, None when unavailable; failed loads cool down."""
    global _detector, _failed_at, _failed_reason
    if _detector is not None:
        return _detector
    availability = detector_availability()
    if not availability.ready or availability.snapshot is None:
        return None
    with _lock:
        if _detector is not None:
            return _detector
        if _failed_at is not None and time.monotonic() - _failed_at < _RETRY_AFTER_S:
            return None
        try:
            _detector = _Detector(availability.snapshot)
            _failed_at = None
            _failed_reason = None
        except Exception as exc:                          # noqa: BLE001
            logger.warning("Detector unavailable: %s", exc)
            _failed_at = time.monotonic()
            _failed_reason = f"detector load failed: {exc}"
            return None
    return _detector
