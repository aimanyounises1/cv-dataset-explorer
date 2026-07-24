"""2-D projection + clustering of image embeddings for the dataset map view.

UMAP is preferred (better local structure); PCA is the dependency-light
fallback. KMeans gives stable, fast cluster colors for visual grouping.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def project_2d(embeddings: np.ndarray) -> np.ndarray:
    try:
        import umap

        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine",
                            random_state=42)
        xy = reducer.fit_transform(embeddings)
    except Exception as exc:
        logger.warning("UMAP unavailable (%s); falling back to PCA", exc)
        from sklearn.decomposition import PCA

        xy = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    # Normalize to [0, 1] for the frontend canvas.
    xy = xy - xy.min(axis=0)
    span = xy.max(axis=0)
    span[span == 0] = 1.0
    return (xy / span).astype(np.float32)


def cluster(embeddings: np.ndarray, n_clusters: int = 12) -> np.ndarray:
    from sklearn.cluster import KMeans

    n_clusters = max(2, min(n_clusters, len(embeddings) // 10 or 2))
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    return km.fit_predict(embeddings).astype(int)
