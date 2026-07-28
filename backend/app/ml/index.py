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
    def load(cls, kind: str = "image", emb_dir=None) -> Optional["EmbeddingIndex"]:
        d = emb_dir if emb_dir is not None else config.EMB_DIR
        ids_file, embs_file = FILES[kind]
        ids_path, embs_path = d / ids_file, d / embs_file
        if not ids_path.exists() or not embs_path.exists():
            return None
        return cls(np.load(ids_path), np.load(embs_path))

    @classmethod
    def save(cls, ids: np.ndarray, embeddings: np.ndarray, kind: str = "image",
             emb_dir=None) -> None:
        config.ensure_dirs()
        d = emb_dir if emb_dir is not None else config.EMB_DIR
        d.mkdir(parents=True, exist_ok=True)
        ids_file, embs_file = FILES[kind]
        np.save(d / ids_file, ids)
        np.save(d / embs_file, embeddings)

    # -- queries -------------------------------------------------------------
    def row_of(self, item_id: int) -> Optional[int]:
        return self._id_to_row.get(item_id)

    def vector_of(self, item_id: int) -> Optional[np.ndarray]:
        row = self.row_of(item_id)
        return None if row is None else self.embeddings[row]

    def search(
        self, query_vec: np.ndarray, top_k: int = 50,
        allowed_ids: Optional[set[int]] = None,
        penalty: Optional[np.ndarray] = None,
    ) -> list[tuple[int, float]]:
        """Cosine similarity of a normalized query against all items.
        `allowed_ids` restricts the candidate set *before* top-k (so filters
        can never empty out an oversampled result list).

        `penalty` is an optional per-item vector subtracted before ranking (the
        hubness correction — see `app.ml.hubness`). The score returned is the
        one that did the ranking, penalty included: an earlier version returned
        the raw cosine instead, on the theory that a cosine is the more
        interpretable number, and it made the results list read as broken —
        **106 of 228 adjacent pairs on the first page showed a score going UP**,
        because the displayed value was no longer the sort key. A list whose
        numbers do not descend looks like a bug regardless of how principled the
        number is. Callers that surface this score must therefore say which
        basis it is on; `api.search` reports `cosine_adj` when a penalty applied.

        Opt-in per call rather than folded into the index, because the penalty is
        derived from a bank of TEXT queries and is meaningless for the
        image-to-image paths (`similar_to`, duplicate detection) that share this
        method.
        """
        scores = self.embeddings @ query_vec.reshape(-1)
        if penalty is not None:
            if penalty.shape != scores.shape:
                raise ValueError(
                    f"penalty shape {penalty.shape} does not match index "
                    f"{scores.shape}; a mis-aligned penalty would silently "
                    "re-rank against the wrong images")
            scores = scores - penalty
        if allowed_ids is not None:
            # `np.isin`, not a Python generator over every id. The generator
            # version cost 0.334 ms on this 8,000-row index — nearly twice the
            # 0.181 ms of the entire unfiltered search — so applying a filter
            # made a query 2.7x more expensive, essentially all of it
            # interpreter overhead. Measured identical output, 16x faster.
            #
            # It also scales far worse than the maths it guards: the mask is
            # O(n) in Python where the scan is O(n) in BLAS, so at 250k rows the
            # generator would cost ~10.8 ms against a 4.1 ms scan. This one line
            # is why "filters are applied inside ranking" is affordable.
            allowed = np.fromiter(allowed_ids, dtype=self.ids.dtype,
                                  count=len(allowed_ids))
            scores = np.where(np.isin(self.ids, allowed), scores, -np.inf)
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

    def all_pairs_above(
        self, threshold: float, chunk: int = 1024,
    ) -> list[tuple[int, int, float]]:
        """EVERY near-duplicate pair above `threshold`, sorted by similarity.

        Uncapped, deliberately. `duplicate_pairs` truncates to a display limit,
        which is right for a gallery and wrong for a count: "200 duplicate pairs"
        when there are 2,458 is not a smaller answer, it is a false one. Counting
        and listing are different jobs and now have different methods.

        Cost is one pass of the full similarity matrix — measured at 0.2 s for
        8,000 x 768 on an M-series Mac, chunked so the matrix is never
        materialised whole. Lower the threshold rather than raising the cap.
        """
        pairs: list[tuple[int, int, float]] = []
        n = len(self.embeddings)
        for start in range(0, n, chunk):
            block = self.embeddings[start : start + chunk] @ self.embeddings.T
            for bi in range(block.shape[0]):
                i = start + bi
                js = np.nonzero(block[bi] > threshold)[0]
                for j in js[js > i]:   # dedupe (i, j) / (j, i) and self-match
                    pairs.append((int(self.ids[i]), int(self.ids[j]),
                                  float(block[bi, j])))
        pairs.sort(key=lambda p: -p[2])
        return pairs

    def duplicate_pairs(
        self, threshold: float = config.DUPLICATE_THRESHOLD, max_pairs: int = 200,
        chunk: int = 1024,
    ) -> list[tuple[int, int, float]]:
        """The strongest `max_pairs` near-duplicate pairs, for display."""
        return self.all_pairs_above(threshold, chunk=chunk)[:max_pairs]


# -- thread-safe cached singletons -------------------------------------------
_lock = threading.Lock()
_cache: dict[str, Optional[EmbeddingIndex]] = {}


def _get(kind: str) -> Optional[EmbeddingIndex]:
    # The active provider owns the index directory: query vectors and index
    # rows always come from the same embedding space. Cache is keyed by
    # provider so a runtime fallback flip reloads rather than mixing spaces.
    from . import providers

    key = f"{providers.active_provider()}:{kind}"
    if key not in _cache:
        with _lock:
            if key not in _cache:  # double-checked: load once
                _cache[key] = EmbeddingIndex.load(kind, emb_dir=providers.active_emb_dir())
                if _cache[key] is None:
                    logger.warning("No %s embedding index found.", kind)
    return _cache[key]


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
    # The hubness penalty is positionally aligned to this index, so it cannot
    # outlive it. Imported here to keep `hubness` -> `index` a one-way dependency.
    from . import hubness
    hubness.invalidate()
