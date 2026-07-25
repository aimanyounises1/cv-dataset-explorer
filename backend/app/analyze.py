"""Post-ingestion analysis (idempotent, re-runnable):

    python -m app.analyze                  # everything
    python -m app.analyze --only captions  # caption embeddings + QA scores
    python -m app.analyze --only attributes
    python -m app.analyze --only axes      # difficulty axes (0-10 buckets)

Produces:
- caption embeddings (enables retrieval benchmark + "why it matched")
- caption-image agreement (CLIPScore-style QA: low = suspect annotation)
- per-sample caption consistency (low = ambiguous image or outlier caption)
- zero-shot attributes from the label bank (coverage facets)
- difficulty axes: legibility, rarity, difficulty, clutter (see compute_axes)

Run `POST /api/admin/reload` (or restart the API) afterwards to pick up results.
"""
import argparse
import json
import logging
import math
import re
import sys
from collections import Counter

import numpy as np
from tqdm import tqdm

from . import config, db
from .ml.labels import LABEL_BANK, SOFTMAX_SCALE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("analyze")


def embed_captions(conn) -> None:
    from tqdm import tqdm

    from .ml.embedder import Embedder
    from .ml.index import EmbeddingIndex

    rows = conn.execute("SELECT id, text FROM captions ORDER BY id").fetchall()
    if not rows:
        logger.warning("No captions to embed.")
        return
    embedder = Embedder()
    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    texts = [r["text"] for r in rows]
    chunks = []
    batch = 256
    for i in tqdm(range(0, len(texts), batch), desc="Embedding captions"):
        chunks.append(embedder.encode_texts(texts[i : i + batch]))
    EmbeddingIndex.save(ids, np.concatenate(chunks, axis=0), kind="caption")
    logger.info("Saved %d caption embeddings.", len(ids))


def compute_caption_scores(conn) -> None:
    """Agreement: cosine(image, its own caption). Consistency: mean pairwise
    cosine among a sample's captions (unit vectors: (|Σv|² − n) / (n(n−1)))."""
    from .ml.index import EmbeddingIndex

    img = EmbeddingIndex.load("image")
    cap = EmbeddingIndex.load("caption")
    if img is None or cap is None:
        logger.error("Need both image and caption embeddings first.")
        return

    rows = conn.execute("SELECT id, sample_id FROM captions ORDER BY id").fetchall()
    updates, by_sample = [], {}
    for r in rows:
        cvec = cap.vector_of(r["id"])
        ivec = img.vector_of(r["sample_id"])
        if cvec is None or ivec is None:
            continue
        updates.append((float(np.dot(cvec, ivec)), r["id"]))
        by_sample.setdefault(r["sample_id"], []).append(cvec)
    conn.executemany("UPDATE captions SET agreement = ? WHERE id = ?", updates)

    cons = []
    for sid, vecs in by_sample.items():
        n = len(vecs)
        if n < 2:
            continue
        total = np.linalg.norm(np.sum(vecs, axis=0)) ** 2
        cons.append((float((total - n) / (n * (n - 1))), sid))
    conn.executemany("UPDATE samples SET caption_consistency = ? WHERE id = ?", cons)
    conn.commit()
    logger.info("Scored %d captions, %d samples.", len(updates), len(cons))


def compute_attributes(conn) -> None:
    from .ml.embedder import Embedder
    from .ml.index import EmbeddingIndex

    img = EmbeddingIndex.load("image")
    if img is None:
        logger.error("Need image embeddings first (run `python -m app.ingest`).")
        return
    embedder = Embedder()

    conn.execute("DELETE FROM attributes")
    for grp, labels in LABEL_BANK.items():
        names = list(labels.keys())
        prompts = [labels[n] for n in names]
        text_embs = embedder.encode_texts(prompts)          # (L, D)
        logits = (img.embeddings @ text_embs.T) * SOFTMAX_SCALE  # (N, L)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        best = probs.argmax(axis=1)
        conn.executemany(
            "INSERT INTO attributes(sample_id, grp, label, confidence) VALUES (?,?,?,?)",
            [(int(img.ids[i]), grp, names[best[i]], float(probs[i, best[i]]))
             for i in range(len(img.ids))],
        )
        logger.info("Attributed group '%s' (%d labels).", grp, len(names))
    conn.commit()


def _percentile_buckets(values: dict[int, float], invert: bool = False) -> dict[int, int]:
    """Map raw measurements onto integer 0-10 buckets by percentile rank.

    Why percentile and not the raw value: a Laplacian variance, a cosine
    distance and an inverse document frequency live on entirely different
    scales with distributions nobody can predict in advance, so a range filter
    over them behaves erratically — "blur >= 40" means something different on
    every dataset and nothing to a user. Ranking first makes "rarity >= 7" and
    "difficulty >= 7" both mean "roughly the top 30% of this dataset", which is
    what lets several sliders be used together.

    What it costs: the scores become **dataset-relative**. A 7 here is not a 7
    on COCO, the buckets are near-uniformly populated by construction, and
    re-running after ingesting more images can move a sample's bucket without
    its image having changed. These are difficulty *ranks*, not measurements.

    Ties share the lowest bucket, which keeps the pass idempotent.
    """
    if not values:
        return {}
    ids = list(values)
    arr = np.array([values[i] for i in ids], dtype=np.float64)
    if invert:
        arr = -arr
    order = np.sort(arr)
    ranks = np.searchsorted(order, arr, side="left").astype(np.float64)
    pct = ranks / max(1, len(arr) - 1)
    buckets = np.clip(np.round(pct * 10), 0, 10).astype(int)
    return {sid: int(b) for sid, b in zip(ids, buckets, strict=True)}


def _image_quality(conn) -> tuple[dict[int, float], dict[int, float]]:
    """Blur and darkness, measured on the stored thumbnails.

    Variance of the Laplacian is the standard cheap focus measure: a sharp
    image has strong second derivatives, a blurred one does not. Computed with
    plain array slicing so this needs no convolution library.
    """
    from PIL import Image

    blur, luma = {}, {}
    rows = conn.execute("SELECT id, filename FROM samples ORDER BY id").fetchall()
    for r in tqdm(rows, desc="Measuring legibility"):
        path = config.THUMBS_DIR / r["filename"]
        if not path.exists():
            continue
        try:
            with Image.open(path) as im:
                g = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
        except Exception as exc:            # a corrupt thumbnail is not fatal
            logger.warning("Could not read %s: %s", r["filename"], exc)
            continue
        if g.shape[0] < 3 or g.shape[1] < 3:
            continue
        lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
               - 4.0 * g[1:-1, 1:-1])
        blur[r["id"]] = float(lap.var())
        luma[r["id"]] = float(g.mean())
    return blur, luma


def _caption_signals(conn) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Rarity, vocabulary spread and length dispersion, from caption text alone.

    This half of the scoring needs no model at all, which is why it still works
    when embeddings are absent.
    """
    from .db import STOPWORDS

    per_sample: dict[int, list[list[str]]] = {}
    doc_freq: Counter = Counter()
    for r in conn.execute("SELECT sample_id, text FROM captions"):
        toks = [t for t in re.findall(r"[a-z']+", r["text"].lower())
                if len(t) > 2 and t not in STOPWORDS]
        per_sample.setdefault(r["sample_id"], []).append(toks)
        doc_freq.update(set(toks))

    n_docs = max(1, len(per_sample))
    idf = {w: math.log(n_docs / (1 + df)) for w, df in doc_freq.items()}

    word_rarity, vocab_spread, length_spread = {}, {}, {}
    for sid, caps in per_sample.items():
        flat = [t for c in caps for t in c]
        # Mean IDF over the sample's rarest terms: a caption mentioning
        # "unicycle" is a rarer sample than one mentioning "dog", and averaging
        # over everything would drown that in the common words beside it.
        if flat:
            top = sorted((idf.get(t, 0.0) for t in set(flat)), reverse=True)[:3]
            word_rarity[sid] = float(np.mean(top))
        vocab_spread[sid] = float(len(set(flat)))
        lens = [len(c) for c in caps]
        length_spread[sid] = float(np.std(lens)) if len(lens) > 1 else 0.0
    return word_rarity, vocab_spread, length_spread


def _embedding_isolation(img, k: int = 10) -> dict[int, float]:
    """Mean cosine distance to the k nearest neighbours: an image with no close
    neighbours in embedding space is, by construction, unlike the rest of the
    dataset. Chunked so the 8k x 8k similarity matrix is never materialised."""
    isolation: dict[int, float] = {}
    n = len(img.ids)
    if n < 2:
        return isolation
    kk = min(k, n - 1)
    for start in range(0, n, 512):
        block = img.embeddings[start : start + 512] @ img.embeddings.T
        for bi in range(block.shape[0]):
            row = block[bi]
            row[start + bi] = -np.inf                      # exclude self
            top = np.partition(row, -kk)[-kk:]
            isolation[int(img.ids[start + bi])] = float(1.0 - top.mean())
    return isolation


def compute_axes(conn) -> None:
    """Score every sample on four difficulty axes (see db.AXES).

    Deliberately omitted: Meteor's *Dynamic Complexity* axis (Stable Flow ->
    Rule Violation). It is about agent behaviour over time and Flickr8k is
    still photographs with no motion, no agents and no rules — there is no
    honest analogue, and inventing one to match the count would be worse than
    shipping four. Recorded in the README as an omission.
    """
    from .ml.index import EmbeddingIndex

    ids = [r["id"] for r in conn.execute("SELECT id FROM samples ORDER BY id")]
    if not ids:
        logger.warning("No samples; nothing to score.")
        return

    blur, luma = _image_quality(conn)
    word_rarity, vocab_spread, length_spread = _caption_signals(conn)

    img = EmbeddingIndex.load("image")
    if img is None:
        logger.warning("No image embeddings — rarity uses caption vocabulary only "
                       "and difficulty will be left unscored. Run `python -m "
                       "app.ingest` then re-run this pass for the full picture.")
    isolation = _embedding_isolation(img) if img is not None else {}

    agreement = {r["sample_id"]: r["m"] for r in conn.execute(
        "SELECT sample_id, AVG(agreement) AS m FROM captions "
        "WHERE agreement IS NOT NULL GROUP BY sample_id")}
    consistency = {r["id"]: r["caption_consistency"] for r in conn.execute(
        "SELECT id, caption_consistency FROM samples "
        "WHERE caption_consistency IS NOT NULL")}

    # Each component becomes a percentile in [0, 1]; an axis built from several
    # components averages those percentiles, so no component can dominate by
    # having a wider raw range than the others.
    def pct(values: dict[int, float], invert: bool = False) -> dict[int, float]:
        return {sid: b / 10.0 for sid, b in _percentile_buckets(values, invert).items()}

    def combine(name: str, parts: list[dict[int, float]]) -> dict[int, int]:
        merged: dict[int, float] = {}
        for sid in ids:
            vals = [p[sid] for p in parts if sid in p]
            if vals:
                merged[sid] = float(np.mean(vals))
        logger.info("Axis '%s' scored for %d/%d samples.", name, len(merged), len(ids))
        return _percentile_buckets(merged)

    axes = {
        # Low sharpness and low luminance both reduce legibility, so the axis
        # runs Clear(0) -> Blind(10) with both components inverted.
        "legibility": combine("legibility", [pct(blur, invert=True), pct(luma, invert=True)]),
        "rarity": combine("rarity", [pct(word_rarity), pct(isolation)]),
        # Where the model's own understanding is weakest: it agrees least with
        # the caption, and the five human captions agree least with each other.
        "difficulty": combine("difficulty", [pct(agreement, invert=True),
                                             pct(consistency, invert=True)]),
        "clutter": combine("clutter", [pct(vocab_spread), pct(length_spread)]),
    }

    raw = {
        "legibility": {"blur": blur, "luma": luma},
        "rarity": {"word_rarity": word_rarity, "isolation": isolation},
        "difficulty": {"agreement": agreement, "consistency": consistency},
        "clutter": {"vocab": vocab_spread, "length_sd": length_spread},
    }
    updates = []
    for sid in ids:
        detail = {
            axis: {k: round(v[sid], 4) for k, v in comps.items() if sid in v}
            for axis, comps in raw.items()
        }
        updates.append((
            axes["legibility"].get(sid), axes["rarity"].get(sid),
            axes["difficulty"].get(sid), axes["clutter"].get(sid),
            json.dumps({a: d for a, d in detail.items() if d}), sid,
        ))
    conn.executemany(
        "UPDATE samples SET legibility=?, rarity=?, difficulty=?, clutter=?, "
        "axis_detail=? WHERE id=?", updates)
    conn.commit()
    logger.info("Wrote axis scores for %d samples.", len(updates))


def run_all(conn, only: str = "all") -> None:
    if only in ("all", "captions"):
        embed_captions(conn)
        compute_caption_scores(conn)
    if only in ("all", "attributes"):
        compute_attributes(conn)
    if only in ("all", "axes"):
        # Last: difficulty reads the caption scores the earlier passes write.
        compute_axes(conn)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dataset analysis passes.")
    parser.add_argument("--only", choices=["all", "captions", "attributes", "axes"],
                        default="all")
    args = parser.parse_args()

    config.ensure_dirs()
    conn = db.connect()
    db.init_db(conn)
    run_all(conn, only=args.only)
    conn.close()
    logger.info("Done. Hit POST /api/admin/reload (or restart the API) to pick this up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
