"""In-memory embedding indexes (images and captions).

8k x 768 float32 ≈ 24 MB — exact brute-force cosine search via a single
matrix multiply answers in ~1 ms. An ANN index (FAISS/HNSW) would add build
time and recall loss for zero benefit at this scale; this class is the seam
where one would slot in for much larger datasets.
"""
import logging
import threading
from typing import Optional

import numpy as np

from .. import config

logger = logging.getLogger(__name__)

FILES = {
    "image": ("sample_ids.npy", "image_embeddings.npy"),
    "caption": ("caption_ids.npy", "caption_embeddings.npy"),
}


class EmbeddingIndex:
    def __init__(self, ids: np.ndarray, embeddings: np.ndarray):
        assert len(ids) == len(embeddings)
        self.ids = ids.astype(np.int64)
        self.embeddings = embeddings.astype(np.float32)  # already L2-normalized
        self._id_to_row = {int(i): r for r, i in enumerate(self.ids)}

    # -- persistence ---------------------------------------------------------
    @classmethod
    def load(cls, kind: str = "image") -> Optional["EmbeddingIndex"]:
        ids_file, embs_file = FILES[kind]
        ids_path, embs_path = config.EMB_DIR / ids_file, config.EMB_DIR / embs_file
        if not ids_path.exists() or not embs_path.exists():
            return None
        return cls(np.load(ids_path), np.load(embs_path))

    @classmethod
    def save(cls, ids: np.ndarray, embeddings: np.ndarray, kind: str = "image") -> None:
        config.ensure_dirs()
        ids_file, embs_file = FILES[kind]
        np.save(config.EMB_DIR / ids_file, ids)
        np.save(config.EMB_DIR / embs_file, embeddings)

    # -- queries -------------------------------------------------------------
    def row_of(self, item_id: int) -> Optional[int]:
        return self._id_to_row.get(item_id)

    def vector_of(self, item_id: int) -> Optional[np.ndarray]:
        row = self.row_of(item_id)
        return None if row is None else self.embeddings[row]

    def search(
        self, query_vec: np.ndarray, top_k: int = 50,
        allowed_ids: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """Cosine similarity of a normalized query against all items.
        `allowed_ids` restricts the candidate set *before* top-k (so filters
        can never empty out an oversampled result list)."""
        scores = self.embeddings @ query_vec.reshape(-1)
        if allowed_ids is not None:
            mask = np.fromiter((int(i) in allowed_ids for i in self.ids),
                               dtype=bool, count=len(self.ids))
            scores = np.where(mask, scores, -np.inf)
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self.ids[i]), float(scores[i])) for i in top
                if np.isfinite(scores[i])]

    def similar_to(self, item_id: int, top_k: int = 12) -> list[tuple[int, float]]:
        vec = self.vector_of(item_id)
        if vec is None:
            return []
        results = self.search(vec, top_k + 1)
        return [(i, s) for i, s in results if i != item_id][:top_k]

    def duplicate_pairs(
        self, threshold: float = config.DUPLICATE_THRESHOLD, max_pairs: int = 200,
        chunk: int = 1024,
    ) -> list[tuple[int, int, float]]:
        """Near-duplicate pairs (cosine > threshold), chunked to bound memory."""
        pairs: list[tuple[int, int, float]] = []
        n = len(self.embeddings)
        for start in range(0, n, chunk):
            block = self.embeddings[start : start + chunk] @ self.embeddings.T
            for bi in range(block.shape[0]):
                i = start + bi
                for j in np.nonzero(block[bi] > threshold)[0]:
                    if j > i:  # dedupe (i, j) / (j, i) and self-match
                        pairs.append((int(self.ids[i]), int(self.ids[j]), float(block[bi, j])))
        pairs.sort(key=lambda p: -p[2])
        return pairs[:max_pairs]


# -- thread-safe cached singletons -------------------------------------------
_lock = threading.Lock()
_cache: dict[str, Optional[EmbeddingIndex]] = {}


def _get(kind: str) -> Optional[EmbeddingIndex]:
    if kind not in _cache:
        with _lock:
            if kind not in _cache:  # double-checked: load once
                _cache[kind] = EmbeddingIndex.load(kind)
                if _cache[kind] is None:
                    logger.warning("No %s embedding index found.", kind)
    return _cache[kind]


def get_index() -> Optional[EmbeddingIndex]:
    """Image index, or None when embeddings haven't been computed."""
    return _get("image")


def get_caption_index() -> Optional[EmbeddingIndex]:
    """Caption index (built by `python -m app.analyze`), or None."""
    return _get("caption")


def invalidate_index() -> None:
    """Force reload on next access (used by /api/admin/reload after ingestion)."""
    with _lock:
        _cache.clear()
