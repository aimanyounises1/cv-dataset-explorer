"""In-memory embedding index.

8k x 768 float32 ≈ 24 MB — exact brute-force cosine search via a single
matrix multiply answers in ~1 ms. An ANN index (FAISS/HNSW) would add build
time and recall loss for zero benefit at this scale; the interface below is
the seam where one would slot in for much larger datasets.
"""
import logging
from typing import Optional

import numpy as np

from .. import config

logger = logging.getLogger(__name__)

IDS_FILE = "sample_ids.npy"
EMBS_FILE = "image_embeddings.npy"


class EmbeddingIndex:
    def __init__(self, sample_ids: np.ndarray, embeddings: np.ndarray):
        assert len(sample_ids) == len(embeddings)
        self.sample_ids = sample_ids.astype(np.int64)
        self.embeddings = embeddings.astype(np.float32)  # already L2-normalized
        self._id_to_row = {int(sid): i for i, sid in enumerate(self.sample_ids)}

    # -- persistence ---------------------------------------------------------
    @classmethod
    def load(cls) -> Optional["EmbeddingIndex"]:
        ids_path = config.EMB_DIR / IDS_FILE
        embs_path = config.EMB_DIR / EMBS_FILE
        if not ids_path.exists() or not embs_path.exists():
            return None
        return cls(np.load(ids_path), np.load(embs_path))

    @classmethod
    def save(cls, sample_ids: np.ndarray, embeddings: np.ndarray) -> None:
        config.ensure_dirs()
        np.save(config.EMB_DIR / IDS_FILE, sample_ids)
        np.save(config.EMB_DIR / EMBS_FILE, embeddings)

    # -- queries -------------------------------------------------------------
    def search(self, query_vec: np.ndarray, top_k: int = 50) -> list[tuple[int, float]]:
        """Cosine similarity of a normalized query against all images."""
        scores = self.embeddings @ query_vec.reshape(-1)
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self.sample_ids[i]), float(scores[i])) for i in top]

    def similar_to(self, sample_id: int, top_k: int = 12) -> list[tuple[int, float]]:
        row = self._id_to_row.get(sample_id)
        if row is None:
            return []
        results = self.search(self.embeddings[row], top_k + 1)
        return [(sid, s) for sid, s in results if sid != sample_id][:top_k]

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
                js = np.nonzero(block[bi] > threshold)[0]
                for j in js:
                    if j > i:  # dedupe (i, j) / (j, i) and self-match
                        pairs.append((int(self.sample_ids[i]), int(self.sample_ids[j]), float(block[bi, j])))
        pairs.sort(key=lambda p: -p[2])
        return pairs[:max_pairs]


_index: Optional[EmbeddingIndex] = None
_index_loaded = False


def get_index() -> Optional[EmbeddingIndex]:
    """Cached index, or None when embeddings haven't been computed."""
    global _index, _index_loaded
    if not _index_loaded:
        _index = EmbeddingIndex.load()
        _index_loaded = True
        if _index is None:
            logger.warning("No embedding index found — semantic search disabled until ingestion runs.")
    return _index


def invalidate_index() -> None:
    global _index_loaded
    _index_loaded = False
