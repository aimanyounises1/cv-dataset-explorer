"""Post-ingestion analysis (idempotent, re-runnable):

    python -m app.analyze                  # everything
    python -m app.analyze --only captions  # caption embeddings + QA scores
    python -m app.analyze --only attributes

Produces:
- caption embeddings (enables retrieval benchmark + "why it matched")
- caption-image agreement (CLIPScore-style QA: low = suspect annotation)
- per-sample caption consistency (low = ambiguous image or outlier caption)
- zero-shot attributes from the label bank (coverage facets)

Run `POST /api/admin/reload` (or restart the API) afterwards to pick up results.
"""
import argparse
import logging
import sys

import numpy as np

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


def run_all(conn, only: str = "all") -> None:
    if only in ("all", "captions"):
        embed_captions(conn)
        compute_caption_scores(conn)
    if only in ("all", "attributes"):
        compute_attributes(conn)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dataset analysis passes.")
    parser.add_argument("--only", choices=["all", "captions", "attributes"], default="all")
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
