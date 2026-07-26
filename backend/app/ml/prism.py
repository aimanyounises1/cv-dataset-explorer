"""PRISM scoring core — numpy only. See docs/PRISM.md.

Each gallery image i carries a *speaker model*: a diagonal Gaussian
N(mu_i, diag(sigma_i^2)) over the text-embedding space, predicting how humans
describe that image. Retrieval ranks by log-likelihood of the query under each
speaker model, minus a Bayes prior that demotes hub images.

The log-likelihood over all N images decomposes into two matmuls:

    ll(q, i) = -0.5 * [ (q*q)·r_i  -  2 q·(mu_i*r_i)  +  A_i + C_i ]
    r_i = 1/sigma_i^2,  A_i = sum(mu_i^2 * r_i),  C_i = sum(2*log sigma_i)

(The d*log(2*pi) constant is dropped: it shifts every score equally.)
With sigma constant, ll is a monotone transform of q·mu — i.e., zero-init
PRISM *is* cosine ranking. Training can only earn its way away from that.
"""
import json
import logging
import threading
from typing import Optional

import numpy as np

from .. import config

logger = logging.getLogger(__name__)

IDS_FILE = "prism_ids.npy"
MU_FILE = "prism_mu.npy"
LOG_SIGMA_FILE = "prism_log_sigma.npy"
META_FILE = "prism_meta.json"


def _logmeanexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    out = m.squeeze(axis) + np.log(np.mean(np.exp(x - m), axis=axis))
    return out


class PrismIndex:
    def __init__(self, ids: np.ndarray, mu: np.ndarray, log_sigma: np.ndarray):
        assert mu.shape == log_sigma.shape and len(ids) == len(mu)
        self.ids = ids.astype(np.int64)
        self.mu = mu.astype(np.float32)
        self.log_sigma = log_sigma.astype(np.float32)
        r = np.exp(-2.0 * self.log_sigma)                      # 1/sigma^2
        self._r = r
        self._p = self.mu * r                                  # mu/sigma^2
        self._a = np.sum(self.mu * self.mu * r, axis=1)        # A_i
        self._c = np.sum(2.0 * self.log_sigma, axis=1)         # C_i
        self._prior: Optional[np.ndarray] = None               # lambda*log p_hat
        self._id_to_row = {int(i): k for k, i in enumerate(self.ids)}

    # -- construction --------------------------------------------------------
    @classmethod
    def identity_from_embeddings(cls, ids: np.ndarray, embeddings: np.ndarray,
                                 bandwidth: float = 1.0) -> "PrismIndex":
        """The zero-init/A0 model: mu = embedding, constant sigma. Ranks
        identically to cosine for L2-normalized queries — the sanity anchor."""
        log_sigma = np.full_like(embeddings, np.log(bandwidth), dtype=np.float32)
        return cls(ids, embeddings.astype(np.float32), log_sigma)

    # -- scoring -------------------------------------------------------------
    def log_likelihood(self, queries: np.ndarray) -> np.ndarray:
        """(m, d) queries -> (m, N) log-likelihood matrix (constant dropped)."""
        q = np.atleast_2d(queries).astype(np.float32)
        quad = (q * q) @ self._r.T - 2.0 * (q @ self._p.T) + self._a + self._c
        return -0.5 * quad

    def set_prior(self, bank: np.ndarray, weight: float, chunk: int = 2048) -> None:
        """Bayes prior from a caption bank (train split only — inductive).
        log p_hat(i) = logmeanexp_b ll(b, i); hubs have high p_hat and pay."""
        parts = []
        for start in range(0, len(bank), chunk):
            parts.append(self.log_likelihood(bank[start:start + chunk]))
        ll = np.concatenate(parts, axis=0)                     # (m_bank, N)
        self._prior = (weight * _logmeanexp(ll, axis=0)).astype(np.float32)

    def clear_prior(self) -> None:
        self._prior = None

    def scores(self, queries: np.ndarray) -> np.ndarray:
        s = self.log_likelihood(queries)
        if self._prior is not None:
            s = s - self._prior
        return s

    def search(self, query_vec: np.ndarray, top_k: int = 50,
               allowed_ids: Optional[set[int]] = None) -> list[tuple[int, float]]:
        """`allowed_ids` restricts candidates BEFORE top-k — the same filter
        contract the plain embedding index honors."""
        s = self.scores(query_vec.reshape(1, -1))[0]
        if allowed_ids is not None:
            mask = np.fromiter((int(i) in allowed_ids for i in self.ids),
                               dtype=bool, count=len(self.ids))
            s = np.where(mask, s, -np.inf)
        k = min(top_k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [(int(self.ids[i]), float(s[i])) for i in top if np.isfinite(s[i])]

    def rank_of(self, queries: np.ndarray, target_ids: np.ndarray,
                chunk: int = 512) -> np.ndarray:
        """0-based rank of each query's target image (exact, full pool)."""
        targets = np.array([self._id_to_row[int(t)] for t in target_ids])
        ranks = np.empty(len(queries), dtype=np.int64)
        for start in range(0, len(queries), chunk):
            s = self.scores(queries[start:start + chunk])
            own = s[np.arange(s.shape[0]), targets[start:start + chunk]]
            ranks[start:start + chunk] = (s > own[:, None]).sum(axis=1)
        return ranks

    # -- persistence ---------------------------------------------------------
    def save(self, meta: Optional[dict] = None) -> None:
        config.ensure_dirs()
        np.save(config.EMB_DIR / IDS_FILE, self.ids)
        np.save(config.EMB_DIR / MU_FILE, self.mu)
        np.save(config.EMB_DIR / LOG_SIGMA_FILE, self.log_sigma)
        if meta is not None:
            (config.EMB_DIR / META_FILE).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls) -> Optional["PrismIndex"]:
        paths = [config.EMB_DIR / f for f in (IDS_FILE, MU_FILE, LOG_SIGMA_FILE)]
        if not all(p.exists() for p in paths):
            return None
        return cls(np.load(paths[0]), np.load(paths[1]), np.load(paths[2]))


# -- serving cache ------------------------------------------------------------
_runtime_lock = threading.Lock()
_runtime: dict[str, Optional[PrismIndex]] = {}


def get_prism_index(image_index=None) -> Optional[PrismIndex]:
    """Cached speaker index for serving, or None when artifacts are absent.

    Validated against the live image index when one is passed: an artifact
    trained on a different corpus (different id set) must never rank — a
    silently wrong ranking is worse than no boost at all.
    """
    if "prism" not in _runtime:
        with _runtime_lock:
            if "prism" not in _runtime:
                idx = PrismIndex.load()
                # Dimension first: after a backbone swap the sample ids are
                # unchanged, so the id check below would happily accept
                # artifacts from a different embedding space and the first
                # boosted query would crash on a shape mismatch.
                if idx is not None and image_index is not None and \
                        idx.mu.shape[1] != image_index.embeddings.shape[1]:
                    logger.warning(
                        "PRISM artifacts are %d-dimensional but the index is "
                        "%d-dimensional (backbone changed?) — ignoring them. "
                        "Re-run `python -m app.train_prism --no-sigma` after "
                        "re-ingestion.",
                        idx.mu.shape[1], image_index.embeddings.shape[1])
                    idx = None
                if idx is not None and image_index is not None and not np.array_equal(
                        np.sort(idx.ids), np.sort(image_index.ids)):
                    logger.warning(
                        "PRISM artifacts do not match the current image index "
                        "(stale corpus?) — ignoring them. Re-run `python -m "
                        "app.train_prism --no-sigma` after re-ingestion.")
                    idx = None
                _runtime["prism"] = idx
    return _runtime["prism"]


def invalidate_prism() -> None:
    """Force reload on next access (wired into /api/admin/reload)."""
    with _runtime_lock:
        _runtime.clear()


def recall_table(ranks: np.ndarray, ks=(1, 5, 10)) -> dict:
    out = {f"R@{k}": round(float((ranks < k).mean()), 4) for k in ks}
    rr = np.where(ranks < 10, 1.0 / (ranks + 1), 0.0)
    out["MRR@10"] = round(float(rr.mean()), 4)
    out["median_rank"] = float(np.median(ranks) + 1)
    return out


def paired_bootstrap_delta(ranks_a: np.ndarray, ranks_b: np.ndarray, k: int = 1,
                           n_boot: int = 2000, seed: int = 0) -> dict:
    """95% CI for R@k(B) - R@k(A) over the same queries (paired)."""
    rng = np.random.default_rng(seed)
    hit_a, hit_b = (ranks_a < k).astype(float), (ranks_b < k).astype(float)
    n = len(hit_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = (hit_b[idx].mean(axis=1) - hit_a[idx].mean(axis=1))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": round(float(hit_b.mean() - hit_a.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "significant": bool(lo > 0 or hi < 0)}
