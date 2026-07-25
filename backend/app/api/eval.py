"""Self-benchmark: text-to-image retrieval recall@k on the dataset's own
ground truth (each caption should retrieve its own image — the standard
Flickr8k/30k/COCO retrieval protocol).

This lets the tool *prove* which search mode is best instead of asserting it.

Leakage control: the held-out caption is excluded from every index it could be
retrieved from. The semantic path is clean by construction (captions are never
in the image index), and the lexical path excludes the query caption's own row
from the FTS scan — without that exclusion the query text is literally in the
index and keyword recall measures nothing but self-retrieval. The other four
captions of the same image stay indexed: that is the protocol, not a leak.
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
# Bumped whenever the protocol or reported metrics change, so a cached result
# from an older definition is never served as if it were current.
PROTOCOL_VERSION = 3


def _cache_path(sample_size: int):
    """Key on everything the result actually depends on.

    The embeddings drive the semantic path, but the captions table and FTS
    index drive the lexical and fused paths, and RRF_K changes the fusion — so
    a key over embedding mtimes alone will happily serve a result computed
    against different data or a different constant. (Bumping PROTOCOL_VERSION
    by hand is not invalidation; it only covers changes a human remembered.)
    """
    stamp = 0.0
    for embs in ("image_embeddings.npy", "caption_embeddings.npy"):
        p = config.EMB_DIR / embs
        if p.exists():
            stamp = max(stamp, p.stat().st_mtime)
    if config.DB_PATH.exists():
        stamp = max(stamp, config.DB_PATH.stat().st_mtime)
    return (config.CACHE_DIR /
            f"eval_v{PROTOCOL_VERSION}_{sample_size}_k{config.RRF_K}_{int(stamp)}.json")


def lexical_candidates(conn, text: str, own_caption_id: int, limit: int = TOP):
    """Ranked sample ids for one caption query, with the query caption's own
    row excluded (FR-EV-2).

    Single definition on purpose: both the lexical path and the fused path need
    this exclusion, and the earlier duplicated SQL meant a test could pin one
    copy while the other silently regressed.
    """
    match = db.fts_escape(text)
    if not match:
        return []
    return conn.execute(
        "SELECT c.sample_id AS sid, MIN(rank) AS best FROM captions_fts f "
        "JOIN captions c ON c.id = f.rowid WHERE captions_fts MATCH ? "
        "AND c.id != ? GROUP BY c.sample_id ORDER BY best LIMIT ?",
        (match, own_caption_id, limit)).fetchall()


def _keyword_ranks(conn, texts: list[str], target_ids: list[int],
                   query_caption_ids: list[int]):
    """Rank of each caption's own image under BM25, plus how much the lexical
    path actually had to work with.

    The second return value matters more than it looks. `fts_escape` builds an
    implicit AND, and these queries are whole captions (~12 words), so for most
    of them the conjunction is satisfied by no other caption in the corpus and
    the candidate list comes back empty. A recall number computed over mostly
    empty candidate lists says nothing about BM25's ranking; reporting it
    without that context is how a benchmark misleads.
    """
    ranks = np.full(len(texts), TOP + 1, dtype=np.int32)
    n_candidates = np.zeros(len(texts), dtype=np.int32)
    for i, (text, target, own) in enumerate(
            zip(texts, target_ids, query_caption_ids, strict=True)):
        rows = lexical_candidates(conn, text, own)
        n_candidates[i] = len(rows)
        for r_i, row in enumerate(rows):
            if row["sid"] == target:
                ranks[i] = r_i
                break
    return ranks, n_candidates


def _metrics(ranks: np.ndarray, exact: bool) -> tuple[dict[str, float], float, float | None]:
    """Recall@k, MRR@TOP, and median rank (1-based) from 0-based ranks.

    `exact` says whether ranks past TOP are real or censored: the semantic path
    scores the whole pool so its ranks are exact at any depth, while the lexical
    and fused paths only look TOP deep. A censored median past TOP is reported
    as None (the UI renders "> 10") rather than as a number it cannot support.
    """
    recall = {str(k): round(float((ranks < k).mean()), 4) for k in KS}
    mrr = round(float(np.where(ranks < TOP, 1.0 / (ranks + 1), 0.0).mean()), 4)
    median = float(np.median(ranks + 1))
    if not exact and median > TOP:
        return recall, mrr, None
    return recall, mrr, median


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

    kw_ranks, kw_candidates = _keyword_ranks(
        conn, [r["text"] for r in picked],
        [r["sample_id"] for r in picked], [r["id"] for r in picked])

    # Hybrid: RRF of the two top-10 lists, rank of the target in the fusion.
    # Reuses lexical_candidates() so the exclusion has exactly one definition.
    k_rrf = config.RRF_K
    hy_ranks = np.full(len(picked), TOP + 1, dtype=np.int32)
    for i, r in enumerate(picked):
        fused: dict[int, float] = {}
        for rank, col in enumerate(top_cols[i]):
            fused[int(img.ids[col])] = fused.get(int(img.ids[col]), 0.0) + 1 / (k_rrf + rank + 1)
        for rank, row in enumerate(lexical_candidates(conn, r["text"], r["id"])):
            fused[row["sid"]] = fused.get(row["sid"], 0.0) + 1 / (k_rrf + rank + 1)
        ordered = sorted(fused, key=lambda sid: -fused[sid])[:TOP]
        if r["sample_id"] in ordered:
            hy_ranks[i] = ordered.index(r["sample_id"])

    # How much each path actually got to rank. The semantic path always scores
    # the whole corpus; the lexical path is limited to whatever the conjunctive
    # query matched, which for full-caption queries is usually nothing.
    lexical_mean = round(float(kw_candidates.mean()), 3)
    lexical_empty = round(float((kw_candidates == 0).mean()), 4)
    full_pool = float(img.embeddings.shape[0])

    def result(mode: str, ranks: np.ndarray, exact: bool,
               mean_candidates: float, empty_rate: float) -> EvalModeResult:
        recall, mrr, median = _metrics(ranks, exact)
        return EvalModeResult(mode=mode, recall_at=recall, mrr=mrr, median_rank=median,
                              mean_candidates=mean_candidates, empty_query_rate=empty_rate)

    resp = EvalResponse(
        available=True, sample_size=len(picked),
        pool_size=int(full_pool), depth=TOP,
        mean_query_words=round(float(np.mean([len(r["text"].split()) for r in picked])), 1),
        results=[
            # Semantic ranks come from a full score matrix, so they are exact at
            # any depth; the other two are only computed TOP deep.
            result("semantic", sem_ranks, True, full_pool, 0.0),
            result("keyword", kw_ranks, False, lexical_mean, lexical_empty),
            # Fusion sees the semantic list plus whatever lexical contributed.
            result("hybrid", hy_ranks, False, full_pool + lexical_mean, 0.0),
        ])
    config.ensure_dirs()
    cache.write_text(resp.model_dump_json())
    return resp
