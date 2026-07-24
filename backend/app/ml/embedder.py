"""SigLIP 2 embedding wrapper.

Torch/transformers are imported lazily so the API server can start (and serve
keyword search, browsing, stats) on machines where the model stack isn't
installed or embeddings haven't been computed yet — graceful degradation.
"""
import logging
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
    def __init__(self, model_name: str = config.EMBED_MODEL, device: Optional[str] = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = device or _pick_device()
        logger.info("Loading %s on %s", model_name, self.device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self._torch = torch

    def encode_images(self, images, batch_size: int = config.EMBED_BATCH_SIZE) -> np.ndarray:
        """images: list of PIL images. Returns L2-normalized (N, D) float32."""
        torch = self._torch
        chunks = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                feats = _features_tensor(self.model.get_image_features(**inputs))
                chunks.append(feats.float().cpu().numpy())
        embs = np.concatenate(chunks, axis=0)
        return _normalize(embs)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
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


_embedder: Optional[Embedder] = None


def get_embedder() -> Optional[Embedder]:
    """Singleton, or None if the model stack is unavailable."""
    global _embedder
    if _embedder is None:
        try:
            _embedder = Embedder()
        except Exception as exc:  # torch missing, no network for weights, etc.
            logger.warning("Embedding model unavailable: %s", exc)
            return None
    return _embedder
