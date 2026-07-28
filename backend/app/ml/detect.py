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
import threading
import time
from pathlib import Path
from typing import Optional

from .. import config

logger = logging.getLogger(__name__)

DETECT_MODEL = config.DETECT_MODEL
FETCH_HINT = ("python -c \"from huggingface_hub import snapshot_download; "
              f"snapshot_download('{DETECT_MODEL}')\"")

_lock = threading.Lock()
_detector = None
_failed_at: Optional[float] = None
_RETRY_AFTER_S = 120.0


def _weights_cached() -> bool:
    from huggingface_hub.constants import HF_HUB_CACHE

    return (Path(HF_HUB_CACHE) /
            f"models--{DETECT_MODEL.replace('/', '--')}").exists()


def detect_ready() -> tuple[bool, Optional[str]]:
    """Cheap probe — no model load, safe on any request path."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False, "torch/transformers not installed — the base requirements provide them"
    if not _weights_cached():
        return False, f"detector weights not downloaded — run: {FETCH_HINT}"
    return True, None


class _Detector:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        from .embedder import _pick_device

        self.device = _pick_device()
        logger.info("Loading %s on %s", DETECT_MODEL, self.device)
        self.processor = AutoProcessor.from_pretrained(
            DETECT_MODEL, local_files_only=True)
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            DETECT_MODEL, local_files_only=True).to(self.device).eval()
        self._torch = torch
        self._infer = threading.Lock()

    def detect(self, image, queries: str, threshold: float = 0.3,
               max_boxes: int = 12) -> list[dict]:
        """Normalized [0..1] boxes for period-separated phrase queries."""
        torch = self._torch
        with self._infer, torch.no_grad():
            inputs = self.processor(images=image, text=queries,
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
            boxes.append({
                "x": round(max(x0, 0) / W, 4), "y": round(max(y0, 0) / H, 4),
                "w": round(min(x1 - x0, W) / W, 4),
                "h": round(min(y1 - y0, H) / H, 4),
                "label": str(label), "score": round(float(score), 3)})
        boxes.sort(key=lambda b: -b["score"])
        return boxes[:max_boxes]


def get_detector() -> Optional[_Detector]:
    """Thread-safe singleton, None when unavailable; failed loads cool down."""
    global _detector, _failed_at
    if _detector is not None:
        return _detector
    ok, _ = detect_ready()
    if not ok:
        return None
    with _lock:
        if _detector is not None:
            return _detector
        if _failed_at is not None and time.monotonic() - _failed_at < _RETRY_AFTER_S:
            return None
        try:
            _detector = _Detector()
            _failed_at = None
        except Exception as exc:                          # noqa: BLE001
            logger.warning("Detector unavailable: %s", exc)
            _failed_at = time.monotonic()
            return None
    return _detector
