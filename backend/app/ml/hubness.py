"""Inverted-softmax hubness correction for text->image retrieval.

The problem, measured on this corpus: image vectors are not uniformly
reachable. Mean pairwise cosine is 0.556, and a handful of images sit close to
almost *every* caption, so they surface for queries they do not depict while the
right image ranks second or third. This is hubness, and it is a property of the
gallery, not of any one query — which is exactly what makes it correctable
offline.

The correction. Take a bank of held-out captions B, encoded the way a real query
is encoded, and give every image a scalar saying how close it is to queries in
general:

    h_i = T * logsumexp_{b in B, image(b) != i} ( s(b, i) / T )

then rank on `s(q, i) - beta * h_i`. That is the log-domain form of the inverted
softmax: dividing each image's score by its total affinity to a query
distribution, instead of comparing raw similarities. `T -> inf` degenerates to
plain mean-subtraction, which measured worse (dev MRR 0.6469 vs 0.6519); `T -> 0`
degenerates to subtracting each image's single best bank match.

Three details that are the whole difference between this working and not:

1. **The bank is re-encoded, not read from `caption_embeddings.npy`.** Those
   stored vectors are encodings of the *raw* caption text; a real query goes
   through `normalize_query_text` first, and the two land in measurably
   different places (that difference is worth 7.2 points of R@1 on its own). A
   bank built from the stored vectors estimates hubness in the wrong region of
   the space and measured **exactly zero** gain — eval MRR 0.6280 against a
   0.6280 baseline. The bank must be encoded through the same seam as a query,
   which is why `build` takes the encoder as an argument.

2. **Self-exclusion.** Every gallery image may have one of its own captions in
   the bank, and that caption is a near-perfect match, so without masking it the
   image is penalised simply for being described. Left in, the correction gave
   MRR 0.6339; masked, 0.6356, on an 0.6280 baseline.

3. **The bank is held out of the benchmark.** `api.eval` excludes these caption
   ids from its sample, so the numbers are never measured against captions that
   helped build the thing being measured.

Cost: one (bank x corpus) matmul offline, 8,000 float32 on disk (32 kB), and one
vector subtract per query. `beta = 0` restores the previous ranking exactly.

Lineage: QB-Norm, Bogolin et al., CVPR 2022 — cited as provenance for the idea.
The temperature, beta, bank size and every number above are ours, measured on
this corpus, tuned on a dev split disjoint from both the bank and the benchmark.
"""
import logging
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .. import config
from .index import EmbeddingIndex, get_index

logger = logging.getLogger(__name__)

# Fixed so the bank is the same set on every machine and every rebuild. It has
# to be reproducible: `api.eval` excludes exactly this set from its sample, and
# a bank that drifted between processes would silently re-contaminate it.
BANK_SEED = 20260726

# A bank is an estimate of "how close is this image to queries in general", and
# on a small corpus that estimate is noise built from the same handful of
# captions the user is searching. Below this many usable captions there is no
# bank and no correction — which is also what keeps small fixtures and
# part-ingested databases behaving exactly as they did before.
MIN_CORPUS_CAPTIONS = 1000
# And the bank never eats more than this share of the corpus, so the benchmark
# (which holds the bank out) always has a real sample left to draw from.
MAX_BANK_FRACTION = 0.25

Encoder = Callable[[list[str]], np.ndarray]


def bank_caption_ids(
    conn,
    size: int = None,
    index: EmbeddingIndex | None = None,
) -> list[int]:
    """The held-out caption ids that estimate hubness.

    Deterministic given the corpus: a fixed seed over the sorted list of caption
    ids that are actually usable (their image must be in the index). Sorted
    first so the draw cannot depend on SQLite's row order — `api.eval` subtracts
    exactly this set from its sample, so it has to be the same set every time,
    in every process.
    """
    size = config.HUBNESS_BANK_SIZE if size is None else size
    index = index if index is not None else get_index()
    if index is None or size <= 0:
        return []
    rows = conn.execute("SELECT id, sample_id FROM captions ORDER BY id").fetchall()
    usable = sorted(r["id"] for r in rows if index.row_of(r["sample_id"]) is not None)
    if len(usable) < MIN_CORPUS_CAPTIONS:
        return []
    rng = np.random.default_rng(BANK_SEED)
    n = min(size, int(len(usable) * MAX_BANK_FRACTION))
    if n <= 0:
        return []
    return sorted(int(usable[i]) for i in rng.choice(len(usable), n, replace=False))


def compute(
    bank_vectors: np.ndarray, bank_sample_ids: np.ndarray, index: EmbeddingIndex,
    temperature: float = None,
) -> np.ndarray:
    """The per-image penalty vector, aligned to `index.ids`.

    `bank_vectors` must be L2-normalized query-side encodings (see the module
    docstring on why stored caption vectors are not a substitute).
    """
    T = config.HUBNESS_TEMPERATURE if temperature is None else temperature
    scores = (bank_vectors @ index.embeddings.T) / T          # (B, N)
    # Mask each bank caption against its own image, so an image is not penalised
    # for being described by the bank. Done in the exponent's input, before the
    # logsumexp, so the masked term contributes nothing rather than a small
    # amount.
    rows, cols = [], []
    for b, sid in enumerate(bank_sample_ids):
        col = index.row_of(int(sid))
        if col is not None:
            rows.append(b)
            cols.append(col)
    if rows:
        scores[np.array(rows), np.array(cols)] = -np.inf
    # logsumexp with the max pulled out — the raw exp overflows for small T.
    peak = scores.max(axis=0)
    # An image can have every bank entry masked away (it owns them all), which
    # leaves peak = -inf and turns `scores - peak` into nan. A nan penalty is the
    # worst possible outcome: nan compares False against everything, so the image
    # would sort somewhere arbitrary instead of being left alone. Score those
    # images as "no evidence, no correction" — penalty 0.
    unseen = ~np.isfinite(peak)
    safe_peak = np.where(unseen, 0.0, peak)
    total = np.exp(scores - safe_peak).sum(axis=0)
    penalty = np.where(unseen, 0.0, T * (safe_peak + np.log(np.where(total > 0, total, 1.0))))
    return penalty.astype(np.float32)


# -- persistence --------------------------------------------------------------

def _fingerprint(
    index: EmbeddingIndex,
    *,
    emb_dir: Path | None = None,
    model_id: str | None = None,
    generation: str | None = None,
) -> str:
    """Everything the artifact depends on. A stale penalty vector is worse than
    none: it is silently mis-aligned with the index it is subtracted from.

    Deliberately **not** the database's mtime. `explorer.db` is rewritten by every
    tag edit, every saved view and every WAL checkpoint, so including it meant a
    single tag invalidated a perfectly good penalty vector: the artifact's ids,
    shape, model, temperature and bank size all still matched the index and it was
    thrown away anyway. The cost was not a warning — it was the correction
    silently ceasing to apply until something rebuilt it (~5 s of encoding plus a
    bank x corpus matmul), which made the benchmark's semantic baseline depend on
    whether anyone had tagged anything since the last build.

    What the penalty actually depends on is the embedding space and the corpus it
    was estimated over. Both are covered here by the embedding artifacts' mtimes
    and the id count, and `load` separately requires the stored id vector to equal
    the index's element for element — which is the check that really guards
    alignment. Captions can only change via re-ingestion, which rewrites
    `image_embeddings.npy` and moves that mtime.
    """
    from . import providers

    target = emb_dir if emb_dir is not None else providers.active_emb_dir()
    bound_model = model_id or providers.active_model_id()
    stamp = 0.0
    for name in ("image_embeddings.npy", "sample_ids.npy"):
        p = target / name
        if p.exists():
            stamp = max(stamp, p.stat().st_mtime)
    return (f"{bound_model}|{generation or ''}|{config.HUBNESS_TEMPERATURE}|"
            f"{config.HUBNESS_BANK_SIZE}|{len(index.ids)}|{int(stamp)}")


def _path(emb_dir: Path | None = None):
    # Provider-scoped: a penalty estimated in one embedding space is
    # meaningless in another, so each provider's index dir carries its own.
    from . import providers

    target = emb_dir if emb_dir is not None else providers.active_emb_dir()
    return target / "hubness.npz"


def _bound_path(emb_dir: Path | None):
    """Keep the legacy no-argument path seam for direct unit callers."""
    return _path() if emb_dir is None else _path(emb_dir)


def build(
    conn,
    encode: Encoder,
    index: Optional[EmbeddingIndex] = None,
    *,
    emb_dir: Path | None = None,
    model_id: str | None = None,
    generation: str | None = None,
) -> Optional[np.ndarray]:
    """Encode the bank, compute the penalty, and cache it. Returns None if there
    is nothing to build from.

    `encode` is passed in rather than imported so the bank goes through the
    exact seam a query does — see the module docstring, point 1.
    """
    index = index if index is not None else get_index()
    if index is None:
        return None
    ids = bank_caption_ids(conn, index=index)
    if not ids:
        return None
    qmarks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, sample_id, text FROM captions WHERE id IN ({qmarks}) ORDER BY id",
        ids).fetchall()
    if not rows:
        return None
    vectors = encode([r["text"] for r in rows])
    sample_ids = np.array([r["sample_id"] for r in rows], dtype=np.int64)
    penalty = compute(vectors, sample_ids, index)
    config.ensure_dirs()
    artifact_path = _bound_path(emb_dir)
    np.savez(
        artifact_path,
        penalty=penalty,
        ids=index.ids,
        fingerprint=np.array(_fingerprint(
            index,
            emb_dir=emb_dir,
            model_id=model_id,
            generation=generation,
        )),
    )
    logger.info("Built hubness penalty from %d bank captions", len(rows))
    return penalty


def load(
    index: Optional[EmbeddingIndex] = None,
    *,
    emb_dir: Path | None = None,
    model_id: str | None = None,
    generation: str | None = None,
) -> Optional[np.ndarray]:
    """The cached penalty, or None when it is absent or stale."""
    index = index if index is not None else get_index()
    artifact_path = _bound_path(emb_dir)
    if index is None or not artifact_path.exists():
        return None
    try:
        blob = np.load(artifact_path, allow_pickle=False)
        fingerprint = _fingerprint(
            index,
            emb_dir=emb_dir,
            model_id=model_id,
            generation=generation,
        )
        if str(blob["fingerprint"]) != fingerprint:
            logger.info("Hubness artifact is stale; ignoring it.")
            return None
        penalty = blob["penalty"]
        # Alignment is positional, so a mismatched id vector must never be used.
        if penalty.shape != (len(index.ids),) or not np.array_equal(blob["ids"], index.ids):
            logger.warning("Hubness artifact does not match the index; ignoring it.")
            return None
        return penalty.astype(np.float32)
    except Exception as exc:
        logger.warning("Could not read the hubness artifact: %s", exc)
        return None


# -- cached accessor ----------------------------------------------------------

_lock = threading.Lock()
_cache: dict[str, Optional[np.ndarray]] = {}
_build_failed: set[str] = set()


def get_penalty(
    conn,
    encode: Optional[Encoder] = None,
    *,
    runtime=None,
) -> Optional[np.ndarray]:
    """The vector to subtract from a semantic score vector, or None.

    Already scaled by `HUBNESS_BETA`, so a caller subtracts it directly and
    cannot forget the weight. That is not a stylistic choice: an earlier version
    returned the unscaled `h` and left the multiply to each of the two call
    sites, both of which omitted it — so the app shipped beta = 1.0 and the
    benchmark's semantic MRR came out at 0.6463 against a 0.6471 baseline, a
    correction that made retrieval very slightly worse. The artifact on disk
    stays unscaled so beta remains a runtime knob needing no rebuild.

    None whenever the correction cannot or should not apply — beta is 0, there
    are no embeddings, the artifact is missing and cannot be built. Callers
    treat None as "rank exactly as before", which is what keeps the app working
    with no model and no artifact on disk.
    """
    if config.HUBNESS_BETA == 0:
        return None
    if runtime is None:
        index = get_index()
        emb_dir = model_id = generation = None
    else:
        index = runtime.image_index
        emb_dir = runtime.emb_dir
        model_id = runtime.model_id
        generation = runtime.generation
    if index is None:
        return None
    fingerprint = _fingerprint(
        index,
        emb_dir=emb_dir,
        model_id=model_id,
        generation=generation,
    )
    cache_key = f"{fingerprint}|beta={config.HUBNESS_BETA}"
    if cache_key in _cache:
        return _cache[cache_key]
    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]
        if runtime is None:
            penalty = load(index)
        else:
            penalty = load(
                index,
                emb_dir=emb_dir,
                model_id=model_id,
                generation=generation,
            )
        if penalty is None and encode is not None and config.HUBNESS_AUTOBUILD \
                and cache_key not in _build_failed:
            try:
                if runtime is None:
                    penalty = build(conn, encode, index)
                else:
                    penalty = build(
                        conn,
                        encode,
                        index,
                        emb_dir=emb_dir,
                        model_id=model_id,
                        generation=generation,
                    )
            except Exception as exc:
                # Never let a failed build break search; degrade to uncorrected
                # ranking and stop retrying the build on every request.
                logger.warning("Hubness build failed: %s", exc)
                _build_failed.add(cache_key)
        # The outcome is cached even when it is None — a corpus too small for a
        # bank would otherwise re-scan the captions table on every single query.
        # `invalidate()` is what re-opens the question, and /api/admin/reload
        # calls it.
        _cache[cache_key] = (
            None if penalty is None
            else (config.HUBNESS_BETA * penalty).astype(np.float32)
        )
    return _cache[cache_key]


def invalidate() -> None:
    """Drop the cached penalty (called when the indexes are reloaded)."""
    with _lock:
        _cache.clear()
        _build_failed.clear()


def main() -> None:
    """`python -m app.ml.hubness` — build the artifact ahead of first search."""
    import logging as _logging

    from .. import db
    from ..api import search as search_api

    _logging.basicConfig(level=_logging.INFO)
    runtime = search_api.get_retrieval_bundle()
    if runtime is None:
        raise SystemExit("Embedding model unavailable — cannot build the bank.")
    embedder = runtime.encoder
    index = runtime.image_index

    def encode(texts: list[str]) -> np.ndarray:
        prompts = [search_api.normalize_query_text(t) for t in texts]
        return np.concatenate(
            [embedder.encode_texts(prompts[i:i + 128])
             for i in range(0, len(prompts), 128)], axis=0)

    conn = db.connect()
    try:
        penalty = build(
            conn,
            encode,
            index,
            emb_dir=runtime.emb_dir,
            model_id=runtime.model_id,
            generation=runtime.generation,
        )
    finally:
        conn.close()
    if penalty is None:
        raise SystemExit("Nothing to build from.")
    print(f"Wrote {_path(runtime.emb_dir)} — {penalty.nbytes / 1024:.0f} kB, "
          f"penalty range [{penalty.min():.4f}, {penalty.max():.4f}]")


if __name__ == "__main__":
    main()
