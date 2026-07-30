"""Ingestion pipeline (one-time setup, idempotent).

    python -m app.ingest                     # full Flickr8k
    python -m app.ingest --limit 200         # quick dev run
    python -m app.ingest --skip-embeddings   # browse/keyword-search only

Steps: download dataset -> store images + thumbnails -> SQLite (captions,
FTS index) -> SigLIP 2 embeddings -> UMAP projection + clusters.

Idempotent in the sense that matters -- rerunning never duplicates a sample --
but not uniformly cheap: ingestion skips samples that already exist, while
embeddings, the UMAP projection and the analysis pass recompute. Use
--skip-embeddings to rerun only the parts that do skip.
"""
import argparse
import logging
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

from . import config, db
from .datasets import get_adapter
from .ml import providers

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest")


def ingest_samples(conn, dataset_name: str, limit=None) -> int:
    adapter = get_adapter(dataset_name)
    existing = {r["filename"] for r in conn.execute("SELECT filename FROM samples")}
    n_new = 0
    manifests_invalidated = False
    for sample in tqdm(adapter.iter_samples(limit=limit), desc="Ingesting samples"):
        if sample.filename in existing:
            continue
        if not manifests_invalidated:
            # Every provider index is corpus-wide. Remove all commit markers
            # before the first image or database row changes, even when this
            # command later skips embeddings or a model load fails.
            providers.remove_all_provider_manifests()
            manifests_invalidated = True
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


def compute_embeddings(
    conn,
) -> tuple[object, providers.ModelSnapshot] | None:
    """Build SigLIP image vectors under one exact cached model revision."""
    rows = conn.execute("SELECT id, filename FROM samples ORDER BY id").fetchall()
    if not rows:
        logger.warning("No samples in DB; nothing to embed.")
        return None

    snapshot = providers.resolve_model_snapshot(
        config.EMBED_MODEL,
        local_files_only=False,
    )
    embedder = providers.load_encoder_for("siglip2", snapshot=snapshot)
    emb_dir = config.emb_dir_for("siglip2")
    # The marker is removed only after the exact model is available, but before
    # the first array changes. A failed download preserves the previous valid
    # index; a failed rebuild can never leave it falsely committed.
    providers.remove_manifest(emb_dir)
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
    providers.atomic_save_npy(
        emb_dir / "sample_ids.npy",
        np.array(ids, dtype=np.int64),
    )
    providers.atomic_save_npy(emb_dir / "image_embeddings.npy", embeddings)
    logger.info("Saved %d embeddings (dim=%d)", len(ids), embeddings.shape[1])
    return embedder, snapshot


def compute_embeddings_provider(conn, provider: str) -> None:
    """Embed the corpus with a non-default provider into its own index dir.

    Images and captions both: the retrieval benchmark's row selection needs a
    caption index in the active space. Arrays are written atomically and the
    manifest last, so an interrupted run can never masquerade as a complete
    index — a directory without a complete manifest is reported as such and the
    fix is simply to re-run this command. UMAP, clusters, caption agreement and
    the difficulty axes are deliberately NOT recomputed here: those stored
    signals are SigLIP-derived and documented as such.
    """
    import time as _time

    snapshot = providers.resolve_model_snapshot(
        providers.provider_model_id(provider),
        local_files_only=False,
    )
    encoder = providers.load_encoder_for(
        provider,
        snapshot=snapshot,
    )
    emb_dir = config.emb_dir_for(provider)
    emb_dir.mkdir(parents=True, exist_ok=True)
    # A previous manifest must not vouch for arrays about to be replaced.
    providers.remove_manifest(emb_dir)

    rows = conn.execute("SELECT id, filename FROM samples ORDER BY id").fetchall()
    if not rows:
        logger.warning("No samples in DB; nothing to embed.")
        return

    ids, chunks = [], []
    batch_ids, batch_imgs = [], []
    t0 = _time.perf_counter()

    def flush():
        if batch_imgs:
            chunks.append(encoder.encode_images(batch_imgs))
            ids.extend(batch_ids)
            batch_ids.clear()
            batch_imgs.clear()

    for row in tqdm(rows, desc=f"Embedding images [{provider}]"):
        batch_ids.append(row["id"])
        batch_imgs.append(Image.open(config.IMAGES_DIR / row["filename"]).convert("RGB"))
        if len(batch_imgs) >= config.QWEN_EMBED_BATCH:
            flush()
    flush()
    embs = np.concatenate(chunks, axis=0)
    image_seconds = _time.perf_counter() - t0
    providers.atomic_save_npy(
        emb_dir / "sample_ids.npy", np.array(ids, dtype=np.int64))
    providers.atomic_save_npy(emb_dir / "image_embeddings.npy", embs)

    cap_rows = conn.execute("SELECT id, text FROM captions ORDER BY id").fetchall()
    cvecs = []
    t1 = _time.perf_counter()
    for i in tqdm(range(0, len(cap_rows), 256), desc=f"Embedding captions [{provider}]"):
        cvecs.append(encoder.encode_texts(
            [r["text"] for r in cap_rows[i:i + 256]], kind="document"))
    caps = (np.concatenate(cvecs, axis=0) if cvecs
            else np.zeros((0, embs.shape[1]), np.float32))
    caption_seconds = _time.perf_counter() - t1
    providers.atomic_save_npy(
        emb_dir / "caption_ids.npy",
        np.array([r["id"] for r in cap_rows], dtype=np.int64),
    )
    providers.atomic_save_npy(emb_dir / "caption_embeddings.npy", caps)
    manifest = providers.finalize_provider_manifest(
        conn,
        snapshot,
        provider=provider,
        emb_dir=emb_dir,
        prompt_version=providers.PROVIDERS[provider].prompt_version,
        image_encode_seconds=image_seconds,
        caption_encode_seconds=caption_seconds,
    )
    logger.info("Saved %d image + %d caption embeddings (dim=%d) under %s",
                len(ids), len(cap_rows), embs.shape[1], emb_dir)
    logger.info(
        "Provider build committed at %s (images %.1fs, captions %.1fs).",
        manifest["revision"],
        image_seconds,
        caption_seconds,
    )


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
    parser.add_argument("--provider", choices=list(providers.PROVIDERS), default="siglip2",
                        help="Retrieval provider to embed for. qwen3_vl builds an "
                             "additive index under data/embeddings/qwen3_vl/ and "
                             "leaves the SigLIP index untouched.")
    args = parser.parse_args()

    config.ensure_dirs()
    conn = db.connect()
    db.init_db(conn)

    n_new = ingest_samples(conn, args.dataset, limit=args.limit)
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    logger.info("Ingested %d new samples (%d total).", n_new, total)

    if args.provider != "siglip2":
        # Build only the selected provider. If ingestion added corpus rows, all
        # old provider commit markers were already invalidated above.
        compute_embeddings_provider(conn, args.provider)
        logger.info("Activate it with CVDE_EMBED_PROVIDER=%s and "
                    "POST /api/admin/reload.", args.provider)
    elif not args.skip_embeddings:
        build = compute_embeddings(conn)
        if build is None:
            conn.close()
            return 1
        embedder, snapshot = build
        compute_projection(conn)
        from .analyze import run_all
        run_all(
            conn,
            embedder=embedder,
            snapshot=snapshot,
        )  # caption QA scores + zero-shot attributes; manifest committed last
    else:
        logger.info("Skipped embeddings (--skip-embeddings).")

    conn.close()
    logger.info("Done. Start the API with: uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
