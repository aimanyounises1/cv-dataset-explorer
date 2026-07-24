"""Self-benchmark: text-to-image retrieval recall@k on the dataset's own
ground truth (each caption should retrieve its own image — the standard
Flickr8k/30k/COCO retrieval protocol).

This lets the tool *prove* which search mode is best instead of asserting it.
Note one caveat surfaced in the UI copy: keyword search is flattered by this
protocol because the query caption is literally in the FTS index; semantic
recall is the honest generalization signal.
"""
import json
import sqlite3

import numpy as np
from fastapi import APIRouter, Depends, Query

from .. import config, db
from ..ml.index import get_caption_index, get_index
from ..schemas import EvalModeResult, EvalResponse
from .deps import get_conn

router = APIRouter()

KS = (1, 5, 10)
TOP = 10


def _cache_path(sample_size: int):
    stamp = 0.0
    for _ids, embs in (("sample_ids.npy", "image_embeddings.npy"),
                       ("caption_ids.npy", "caption_embeddings.npy")):
        p = config.EMB_DIR / embs
        if p.exists():
            stamp = max(stamp, p.stat().st_mtime)
    return config.CACHE_DIR / f"eval_{sample_size}_{int(stamp)}.json"


def _keyword_ranks(conn, texts: list[str], target_ids: list[int]) -> np.ndarray:
    ranks = np.full(len(texts), TOP + 1, dtype=np.int32)
    for i, (text, target) in enumerate(zip(texts, target_ids, strict=True)):
        match = db.fts_escape(text)
        if not match:
            continue
        rows = conn.execute(
            "SELECT c.sample_id AS sid, MIN(rank) AS best FROM captions_fts f "
            "JOIN captions c ON c.id = f.rowid WHERE captions_fts MATCH ? "
            "GROUP BY c.sample_id ORDER BY best LIMIT ?", (match, TOP)).fetchall()
        for r_i, row in enumerate(rows):
            if row["sid"] == target:
                ranks[i] = r_i
                break
    return ranks


@router.get("/eval/retrieval", response_model=EvalResponse)
def retrieval_benchmark(
    sample_size: int = Query(1000, ge=50, le=5000),
    conn: sqlite3.Connection = Depends(get_conn),
):
    img, cap = get_index(), get_caption_index()
    if img is None or cap is None:
        return EvalResponse(
            available=False,
            message="Requires image + caption embeddings — run `python -m app.ingest` "
                    "and `python -m app.analyze` first.")

    cache = _cache_path(sample_size)
    if cache.exists():
        return EvalResponse(**json.loads(cache.read_text()))

    rows = conn.execute("SELECT id, sample_id, text FROM captions ORDER BY id").fetchall()
    rows = [r for r in rows if cap.row_of(r["id"]) is not None
            and img.row_of(r["sample_id"]) is not None]
    rng = np.random.default_rng(42)
    picked = [rows[i] for i in rng.choice(len(rows), min(sample_size, len(rows)),
                                          replace=False)]

    cap_rows = np.array([cap.row_of(r["id"]) for r in picked])
    target_cols = np.array([img.row_of(r["sample_id"]) for r in picked])

    # Semantic: full score matrix (S x N), exact ranks.
    scores = cap.embeddings[cap_rows] @ img.embeddings.T
    own = scores[np.arange(len(picked)), target_cols]
    sem_ranks = (scores > own[:, None]).sum(axis=1)

    # Semantic top-k id lists (for hybrid fusion); k bounded by corpus size.
    k = min(TOP, scores.shape[1])
    top_cols = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    row_order = np.take_along_axis(scores, top_cols, axis=1).argsort(axis=1)[:, ::-1]
    top_cols = np.take_along_axis(top_cols, row_order, axis=1)

    kw_ranks = _keyword_ranks(conn, [r["text"] for r in picked],
                              [r["sample_id"] for r in picked])

    # Hybrid: RRF of the two top-10 lists, rank of the target in the fusion.
    hy_ranks = np.full(len(picked), TOP + 1, dtype=np.int32)
    for i, r in enumerate(picked):
        fused: dict[int, float] = {}
        for rank, col in enumerate(top_cols[i]):
            fused[int(img.ids[col])] = fused.get(int(img.ids[col]), 0.0) + 1 / (60 + rank + 1)
        match = db.fts_escape(r["text"])
        if match:
            for rank, row in enumerate(conn.execute(
                "SELECT c.sample_id AS sid, MIN(rank) AS best FROM captions_fts f "
                "JOIN captions c ON c.id = f.rowid WHERE captions_fts MATCH ? "
                "GROUP BY c.sample_id ORDER BY best LIMIT ?", (match, TOP))):
                fused[row["sid"]] = fused.get(row["sid"], 0.0) + 1 / (60 + rank + 1)
        ordered = sorted(fused, key=lambda sid: -fused[sid])[:TOP]
        if r["sample_id"] in ordered:
            hy_ranks[i] = ordered.index(r["sample_id"])

    def recalls(ranks) -> dict[str, float]:
        return {str(k): round(float((ranks < k).mean()), 4) for k in KS}

    resp = EvalResponse(
        available=True, sample_size=len(picked),
        results=[
            EvalModeResult(mode="semantic", recall_at=recalls(sem_ranks)),
            EvalModeResult(mode="keyword", recall_at=recalls(kw_ranks)),
            EvalModeResult(mode="hybrid", recall_at=recalls(hy_ranks)),
        ])
    config.ensure_dirs()
    cache.write_text(resp.model_dump_json())
    return resp
