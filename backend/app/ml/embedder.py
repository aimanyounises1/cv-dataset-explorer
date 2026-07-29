"""SigLIP 2 embedding wrapper.

Torch/transformers are imported lazily so the API server can start (and serve
keyword search, browsing, stats) on machines where the model stack isn't
installed or embeddings haven't been computed yet — graceful degradation.

Inference is serialized. One shared module cannot be run concurrently on the
Metal backend: two threads calling `encode_texts` on the same model either crash
the process with a SIGSEGV inside `copy_cast_kernel_mps` or deadlock inside
Metal, both of which were reproduced on this machine (two threads, 25 encodes
each — one run segfaulted, another hung past ten minutes). FastAPI serves sync
endpoints from a thread pool, so two simultaneous semantic searches were already
enough to hit it; parallel agent lanes made it routine. A lock is the right fix
rather than a per-thread model: encoding one query is milliseconds, and a second
copy of the weights costs ~1.5 GB to avoid a wait nobody can perceive.
"""
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config

logger = logging.getLogger(__name__)


def _pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    def __init__(
        self,
        model_path: str | Path,
        *,
        model_id: str,
        revision: str,
        local_files_only: bool = True,
        device: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.model_id = model_id
        self.revision = revision
        self.model_path = Path(model_path)
        self.device = device or _pick_device()
        logger.info(
            "Loading %s@%s from %s on %s",
            model_id,
            revision,
            self.model_path,
            self.device,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=local_files_only,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=local_files_only,
        )
        self._torch = torch
        # Guards every forward pass through this module — see the module
        # docstring for the crash this prevents. Held for one batch at a time so
        # a long image run cannot block a single-query text encode for its whole
        # duration.
        self._infer = threading.Lock()

    def encode_images(self, images, batch_size: int = config.EMBED_BATCH_SIZE) -> np.ndarray:
        """images: list of PIL images. Returns L2-normalized (N, D) float32."""
        torch = self._torch
        chunks = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            with self._infer, torch.no_grad():
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                feats = _features_tensor(self.model.get_image_features(**inputs))
                chunks.append(feats.float().cpu().numpy())
        embs = np.concatenate(chunks, axis=0)
        return _normalize(embs)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        with self._infer, torch.no_grad():
            inputs = self.processor(
                text=texts, padding="max_length", max_length=64,
                truncation=True, return_tensors="pt",
            ).to(self.device)
            feats = _features_tensor(self.model.get_text_features(**inputs))
        return _normalize(feats.float().cpu().numpy())


def _features_tensor(feats):
    """transformers <5 returns a tensor; >=5 wraps it in a ModelOutput."""
    return feats.pooler_output if hasattr(feats, "pooler_output") else feats


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)
