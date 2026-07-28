"""Optional promptable object segmentation through Transformers SAM2.

The implementation is the production form of ``scripts/bench_sam2.py``:
official ``Sam2Processor`` point/label and box inputs, official
``post_process_masks``, one lazy model, and one inference lock. Model loading is
local-only so an HTTP request can never start a checkpoint download.
"""
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from .. import config

logger = logging.getLogger(__name__)

SEGMENT_MODEL = config.SEGMENT_MODEL
FETCH_HINT = ("python -c \"from huggingface_hub import snapshot_download; "
              f"snapshot_download('{SEGMENT_MODEL}')\"")

_lock = threading.Lock()
_segmenter = None
_failed_at: Optional[float] = None
_RETRY_AFTER_S = 120.0


def _weights_cached() -> bool:
    from huggingface_hub.constants import HF_HUB_CACHE

    return (Path(HF_HUB_CACHE) /
            f"models--{SEGMENT_MODEL.replace('/', '--')}").exists()


def segment_ready() -> tuple[bool, Optional[str]]:
    """Cheap capability probe: imports and cache only, never a model load."""
    try:
        import torch  # noqa: F401
        import transformers
    except ImportError:
        return False, "torch/transformers not installed — the base requirements provide them"
    if not hasattr(transformers, "Sam2Model") or not hasattr(transformers, "Sam2Processor"):
        return False, (f"transformers {transformers.__version__} has no SAM2 image "
                       "segmentation API")
    if not _weights_cached():
        return False, f"SAM2 weights not downloaded — run: {FETCH_HINT}"
    return True, None


class Segmenter:
    """One local SAM2 image model with serialized forwards."""

    def __init__(self):
        import torch
        from transformers import Sam2Model, Sam2Processor

        from .embedder import _pick_device

        self.device = _pick_device()
        logger.info("Loading %s on %s", SEGMENT_MODEL, self.device)
        self.processor = Sam2Processor.from_pretrained(
            SEGMENT_MODEL, local_files_only=True)
        self.model, loading = Sam2Model.from_pretrained(
            SEGMENT_MODEL, local_files_only=True, output_loading_info=True)
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
    global _segmenter, _failed_at
    if _segmenter is not None:
        return _segmenter
    ok, _ = segment_ready()
    if not ok:
        return None
    with _lock:
        if _segmenter is not None:
            return _segmenter
        if _failed_at is not None and time.monotonic() - _failed_at < _RETRY_AFTER_S:
            return None
        try:
            _segmenter = Segmenter()
            _failed_at = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Segmenter unavailable: %s", exc)
            _failed_at = time.monotonic()
            return None
    return _segmenter
