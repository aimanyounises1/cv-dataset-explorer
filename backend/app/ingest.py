"""Ingestion pipeline (one-time setup, idempotent).

    python -m app.ingest                     # full Flickr8k
    python -m app.ingest --limit 200         # quick dev run
    python -m app.ingest --skip-embeddings   # browse/keyword-search only

Steps: download dataset -> store images + thumbnails -> SQLite (captions,
FTS index) -> SigLIP 2 embeddings -> UMAP projection + clusters.
Each stage skips work that is already done, so re-running is cheap.
"""
import argparse
import logging
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

from . import config, db
from .datasets import get_adapter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest")


def ingest_samples(conn, dataset_name: str, limit=None) -> int:
    adapter = get_adapter(dataset_name)
    existing = {r["filename"] for r in conn.execute("SELECT filename FROM samples")}
    n_new = 0
    for sample in tqdm(adapter.iter_samples(limit=limit), desc="Ingesting samples"):
        if sample.filename in existing:
            continue
        img_path = config.IMAGES_DIR / sample.filename
        sample.image.save(img_path, "JPEG", quality=92)

        thumb = sample.image.copy()
        thumb.thumbnail((config.THUMB_SIZE, config.THUMB_SIZE))
        thumb.save(config.THUMBS_DIR / sample.filename, "JPEG", quality=85)

        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
            "VALUES (?,?,?,?,?,?)",
            (dataset_name, sample.filename, sample.split,
             sample.image.width, sample.image.height, img_path.stat().st_size),
        )
        sample_id = cur.lastrowid
        for idx, text in enumerate(sample.captions):
            ccur = conn.execute(
                "INSERT INTO captions(sample_id, idx, text) VALUES (?,?,?)",
                (sample_id, idx, text),
            )
            conn.execute(
                "INSERT INTO captions_fts(rowid, text) VALUES (?,?)",
                (ccur.lastrowid, text),
            )
        n_new += 1
        if n_new % 500 == 0:
            conn.commit()
    conn.commit()
    return n_new


def compute_embeddings(conn) -> None:
    from .ml.embedder import Embedder
    from .ml.index import EmbeddingIndex

    rows = conn.execute("SELECT id, filename FROM samples ORDER BY id").fetchall()
    if not rows:
        logger.warning("No samples in DB; nothing to embed.")
        return

    embedder = Embedder()
    ids, embs = [], []
    batch_ids, batch_imgs = [], []

    def flush():
        if batch_imgs:
            embs.append(embedder.encode_images(batch_imgs))
            ids.extend(batch_ids)
            batch_ids.clear()
            batch_imgs.clear()

    for row in tqdm(rows, desc="Embedding images"):
        batch_ids.append(row["id"])
        batch_imgs.append(Image.open(config.IMAGES_DIR / row["filename"]).convert("RGB"))
        if len(batch_imgs) >= config.EMBED_BATCH_SIZE:
            flush()
    flush()

    embeddings = np.concatenate(embs, axis=0)
    EmbeddingIndex.save(np.array(ids, dtype=np.int64), embeddings)
    logger.info("Saved %d embeddings (dim=%d)", len(ids), embeddings.shape[1])


def compute_projection(conn) -> None:
    from .ml.index import EmbeddingIndex
    from .ml.projection import cluster, project_2d

    index = EmbeddingIndex.load()
    if index is None:
        logger.warning("No embeddings; skipping projection.")
        return
    logger.info("Computing 2-D projection (UMAP) and clusters...")
    xy = project_2d(index.embeddings)
    labels = cluster(index.embeddings)
    conn.executemany(
        "UPDATE samples SET umap_x=?, umap_y=?, cluster=? WHERE id=?",
        [(float(xy[i, 0]), float(xy[i, 1]), int(labels[i]), int(sid))
         for i, sid in enumerate(index.ids)],
    )
    conn.commit()
    logger.info("Projection stored for %d samples.", len(index.ids))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a dataset into the explorer.")
    parser.add_argument("--dataset", default="flickr8k")
    parser.add_argument("--limit", type=int, default=None, help="Cap sample count (dev runs)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip SigLIP embeddings (disables semantic search/map)")
    args = parser.parse_args()

    config.ensure_dirs()
    conn = db.connect()
    db.init_db(conn)

    n_new = ingest_samples(conn, args.dataset, limit=args.limit)
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    logger.info("Ingested %d new samples (%d total).", n_new, total)

    if not args.skip_embeddings:
        compute_embeddings(conn)
        compute_projection(conn)
        from .analyze import run_all
        run_all(conn)  # caption QA scores + zero-shot attributes
    else:
        logger.info("Skipped embeddings (--skip-embeddings).")

    conn.close()
    logger.info("Done. Start the API with: uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
