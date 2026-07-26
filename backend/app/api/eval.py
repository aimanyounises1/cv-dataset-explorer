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

Measures the path that ships. The query text is encoded live, through the same
embedder and the same `normalize_query_text` that `/api/search` uses, and the
two rankings are fused to `config.SEARCH_DEPTH` exactly as `run_search` fuses
them. An earlier version took the precomputed caption vector as the query and
fused only the top ten of each path; both are cheaper, and both measured
something the application never runs — the cached vectors are encodings of the
*raw* caption text, which is not what a search request produces.
"""
import json
import sqlite3

import numpy as np
from fastapi import APIRouter, Depends, Query

from .. import config, db
from ..ml import hubness
from ..ml.index import get_caption_index, get_index
from ..ml.prism import MU_FILE, get_prism_index
from ..schemas import EvalModeResult, EvalResponse
from . import search as search_api
from .deps import get_conn

router = APIRouter()

KS = (1, 5, 10)
TOP = 10
ENCODE_BATCH = 128
# Bumped whenever the protocol or reported metrics change, so a cached result
# from an older definition is never served as if it were current.
#   4: queries encoded live (normalized) instead of read from the caption index;
#      fusion at config.SEARCH_DEPTH instead of TOP; mean_candidates reports the
#      matched pool rather than the truncated candidate list.
#   5: the semantic path now carries the hubness correction that /api/search
#      applies, so the benchmark keeps measuring the shipped ranking; and the
#      sample excludes the hubness bank's caption ids, so no query is one of the
#      captions that built the correction being measured.
#   6: when trained PRISM artifacts are present, two paired rows are added on a
#      dedicated test-split query sample — the trained model saw the train split
#      and was selected on validation, so any other query set would grade it on
#      its own training text. The cache key carries the artifact stamp.
#   7: the hubness artifact's fingerprint no longer contains the database's
#      mtime, so the penalty stops being invalidated by ordinary tag edits and
#      the semantic rows are now measured with the correction actually applied.
#      Every version-6 row was recorded under whichever state the artifact
#      happened to be in, so none of them are comparable to these.
PROTOCOL_VERSION = 7


def _cache_path(sample_size: int):
    """Key on everything the result actually depends on.

    The embeddings drive the semantic path, but the captions table and FTS
    index drive the lexical and fused paths, and RRF_K and SEARCH_DEPTH change
    the fusion — so a key over embedding mtimes alone will happily serve a
    result computed against different data or different constants. (Bumping
    PROTOCOL_VERSION by hand is not invalidation; it only covers changes a
    human remembered.)
    """
    stamp = 0.0
    for embs in ("image_embeddings.npy", "caption_embeddings.npy", MU_FILE):
        p = config.EMB_DIR / embs
        if p.exists():
            stamp = max(stamp, p.stat().st_mtime)
    if config.DB_PATH.exists():
        stamp = max(stamp, config.DB_PATH.stat().st_mtime)
    # Presence is keyed explicitly, not only through the mtime: an artifact
    # trained before the embeddings were (re)built would leave the stamp
    # unchanged, and a cached result without the PRISM rows would keep serving
    # as if the trained model did not exist.
    prism = int((config.EMB_DIR / MU_FILE).exists())
    return (config.CACHE_DIR /
            f"eval_v{PROTOCOL_VERSION}_{sample_size}_k{config.RRF_K}"
            f"_d{config.SEARCH_DEPTH}"
            # The hubness constants re-rank the semantic path, so they belong in
            # the key for exactly the reason RRF_K does.
            f"_h{config.HUBNESS_BETA}-{config.HUBNESS_TEMPERATURE}"
            f"-{config.HUBNESS_BANK_SIZE}_p{prism}_{int(stamp)}.json")


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


def lexical_pool_size(conn, text: str, own_caption_id: int) -> int:
    """How many distinct images the lexical query matched at all, before the
    ranking is truncated.

    Reported instead of the length of the returned list, which saturates at the
    fetch depth and would read as "plenty to rank" for any query that matched
    more images than we asked for.
    """
    match = db.fts_escape(text)
    if not match:
        return 0
    return conn.execute(
        "SELECT COUNT(DISTINCT c.sample_id) FROM captions_fts f "
        "JOIN captions c ON c.id = f.rowid WHERE captions_fts MATCH ? "
        "AND c.id != ?", (match, own_caption_id)).fetchone()[0]


def _keyword_paths(conn, texts: list[str], query_caption_ids: list[int],
                   depth: int):
    """Ranked sample ids per query (to `depth`), plus the matched pool size.

    The second return value matters more than it looks. `fts_escape` builds an
    implicit AND, and these queries are whole captions (~12 words), so for most
    of them the conjunction is satisfied by no other caption in the corpus and
    the candidate list comes back empty. A recall number computed over mostly
    empty candidate lists says nothing about BM25's ranking; reporting it
    without that context is how a benchmark misleads.

    Widening the conjunction is measurable and was measured: OR-ing the content
    terms takes keyword R@10 from 3.6% to 53.0% and the empty rate from 90.6% to
    0%, but it *lowers* fused MRR once queries get short (0.1669 vs 0.1831 on
    three-word queries), because a broad lexical list displaces a stronger
    semantic one. The conjunction stays.
    """
    lists, pools = [], np.zeros(len(texts), dtype=np.int64)
    for i, (text, own) in enumerate(zip(texts, query_caption_ids, strict=True)):
        rows = lexical_candidates(conn, text, own, limit=depth)
        lists.append([r["sid"] for r in rows])
        pools[i] = lexical_pool_size(conn, text, own)
    return lists, pools


def _ranks_of(lists: list[list[int]], target_ids: list[int],
              depth: int = TOP) -> np.ndarray:
    """0-based rank of each target within the first `depth` of its list."""
    ranks = np.full(len(lists), TOP + 1, dtype=np.int32)
    for i, (lst, target) in enumerate(zip(lists, target_ids, strict=True)):
        window = lst[:depth]
        if target in window:
            ranks[i] = window.index(target)
    return ranks


def _encode_queries(texts: list[str]) -> np.ndarray | None:
    """Query vectors the way `/api/search` produces them, or None if the model
    stack is unavailable.

    Resolved through `search_api` rather than importing the embedder directly:
    the benchmark's whole claim is that it measures what search does, so the two
    have to reach the model through one seam. Importing `get_embedder` here
    would let a caller substitute an embedder for search and silently benchmark
    a different one.
    """
    embedder = search_api.get_embedder()
    if embedder is None:
        return None
    prompts = [search_api.normalize_query_text(t) for t in texts]
    chunks = [embedder.encode_texts(prompts[i:i + ENCODE_BATCH])
              for i in range(0, len(prompts), ENCODE_BATCH)]
    return np.concatenate(chunks, axis=0)


def _bank_encoder():
    """Encoder for a first-use hubness build, resolved through `search_api` for
    the same reason `_encode_queries` is: one seam to the model."""
    embedder = search_api.get_embedder()
    return None if embedder is None else search_api._encode_for_bank(embedder)


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
    # Hold the hubness bank out of the benchmark. Those captions were used to
    # build the per-image penalty the semantic path now subtracts, so scoring
    # them would be measuring the correction against its own training text —
    # the same class of mistake as the self-retrieval this file already guards.
    bank = set(hubness.bank_caption_ids(conn))
    rows = [r for r in rows if r["id"] not in bank]
    rng = np.random.default_rng(42)
    picked = [rows[i] for i in rng.choice(len(rows), min(sample_size, len(rows)),
                                          replace=False)]

    texts = [r["text"] for r in picked]
    targets = [r["sample_id"] for r in picked]
    target_cols = np.array([img.row_of(r["sample_id"]) for r in picked])
    depth = config.SEARCH_DEPTH
    message = None

    # Query vectors exactly as a search request builds them. Falling back to the
    # stored caption vectors keeps the benchmark available on a machine with no
    # model stack, but those are encodings of the raw caption text, so the
    # response has to say the number is not the shipped path.
    qvecs = _encode_queries(texts)
    if qvecs is None:
        qvecs = cap.embeddings[np.array([cap.row_of(r["id"]) for r in picked])]
        message = ("Embedding model unavailable — queries were taken from the "
                   "stored caption vectors (raw, un-normalized text) instead of "
                   "being encoded the way /api/search encodes them. The semantic "
                   "and hybrid rows understate the shipped path.")

    # Semantic: full score matrix (S x N), exact ranks. The hubness penalty is
    # subtracted here for the same reason the queries are encoded live — the
    # benchmark has to rank the way `index.search` ranks, or it is measuring a
    # path the application does not take.
    scores = qvecs @ img.embeddings.T
    penalty = hubness.get_penalty(conn, _bank_encoder())
    if penalty is not None:
        scores = scores - penalty
    own = scores[np.arange(len(picked)), target_cols]
    sem_ranks = (scores > own[:, None]).sum(axis=1)

    # Semantic id lists for fusion, taken to the same depth run_search fuses to.
    k = min(depth, scores.shape[1])
    top_cols = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    row_order = np.take_along_axis(scores, top_cols, axis=1).argsort(axis=1)[:, ::-1]
    top_cols = np.take_along_axis(top_cols, row_order, axis=1)
    sem_lists = [[int(img.ids[c]) for c in row] for row in top_cols]

    kw_lists, kw_pool = _keyword_paths(conn, texts, [r["id"] for r in picked], depth)
    kw_ranks = _ranks_of(kw_lists, targets)

    # Hybrid: RRF of the two rankings at SEARCH_DEPTH, same constant and same
    # equal weighting run_search uses. Reuses lexical_candidates() via
    # _keyword_paths so the exclusion has exactly one definition.
    k_rrf = config.RRF_K
    hy_lists = []
    for i in range(len(picked)):
        fused: dict[int, float] = {}
        for rank, sid in enumerate(sem_lists[i]):
            fused[sid] = fused.get(sid, 0.0) + 1 / (k_rrf + rank + 1)
        for rank, sid in enumerate(kw_lists[i]):
            fused[sid] = fused.get(sid, 0.0) + 1 / (k_rrf + rank + 1)
        hy_lists.append(sorted(fused, key=lambda sid: -fused[sid])[:TOP])
    hy_ranks = _ranks_of(hy_lists, targets)

    # How much each path actually got to rank. The semantic path always scores
    # the whole corpus; the lexical path is limited to whatever the conjunctive
    # query matched, which for full-caption queries is usually nothing.
    lexical_mean = round(float(kw_pool.mean()), 3)
    lexical_empty = round(float((kw_pool == 0).mean()), 4)
    full_pool = float(img.embeddings.shape[0])

    def result(mode: str, ranks: np.ndarray, exact: bool,
               mean_candidates: float, empty_rate: float,
               queries: int | None = None, note: str | None = None) -> EvalModeResult:
        recall, mrr, median = _metrics(ranks, exact)
        return EvalModeResult(mode=mode, recall_at=recall, mrr=mrr, median_rank=median,
                              mean_candidates=mean_candidates, empty_query_rate=empty_rate,
                              queries=queries, note=note)

    results = [
        # Semantic ranks come from a full score matrix, so they are exact at
        # any depth; the other two are only computed TOP deep.
        result("semantic", sem_ranks, True, full_pool, 0.0),
        result("keyword", kw_ranks, False, lexical_mean, lexical_empty),
        # Fusion sees the semantic list plus whatever lexical contributed.
        result("hybrid", hy_ranks, False, full_pool + lexical_mean, 0.0),
    ]

    # Boosted mode, measured only where measuring it is honest. The PRISM model
    # trained on the train split and was epoch-selected on validation, so the
    # test split holds the only captions it has never seen; grading it on the
    # main sample (~75% train captions) would score it on its own training
    # text — the same class of leak the self-retrieval exclusion above guards.
    # Its rows therefore use a dedicated test-split sample, with a paired
    # semantic row on the *same queries* so the comparison is like-for-like.
    prism = get_prism_index(img)
    if prism is not None:
        t_rows = conn.execute(
            "SELECT c.id, c.sample_id, c.text FROM captions c "
            "JOIN samples s ON s.id = c.sample_id "
            "WHERE s.split = 'test' ORDER BY c.id").fetchall()
        t_rows = [r for r in t_rows if cap.row_of(r["id"]) is not None
                  and img.row_of(r["sample_id"]) is not None
                  and r["id"] not in bank]
        t_picked = [t_rows[i] for i in rng.choice(
            len(t_rows), min(sample_size, len(t_rows)), replace=False)]
        # Live encoding only: boosted mode does not exist without the model
        # stack, so under the stored-vector fallback there is nothing to measure.
        t_vecs = _encode_queries([r["text"] for r in t_picked]) if t_picked else None
        if t_vecs is not None:
            t_targets = np.array([r["sample_id"] for r in t_picked])
            t_cols = np.array([img.row_of(r["sample_id"]) for r in t_picked])
            t_scores = t_vecs @ img.embeddings.T
            if penalty is not None:
                t_scores = t_scores - penalty
            t_own = t_scores[np.arange(len(t_picked)), t_cols]
            sem_t_ranks = (t_scores > t_own[:, None]).sum(axis=1)
            # No hubness penalty on the boosted path — stacking the prior on the
            # trained mu measured worse than the mu alone (see api/search.py).
            boost_ranks = prism.rank_of(t_vecs, t_targets)
            n_t = len(t_picked)
            results += [
                result("semantic (test)", sem_t_ranks, True, full_pool, 0.0,
                       queries=n_t,
                       note="The shipped semantic ranking, restricted to "
                            "test-split queries — the paired baseline for the "
                            "boosted row."),
                result("boosted (test)", boost_ranks, True, full_pool, 0.0,
                       queries=n_t,
                       note="Trained PRISM re-ranking (`python -m "
                            "app.train_prism`). Test-split queries only: the "
                            "model trained on the train split and was selected "
                            "on validation, so any other query set would grade "
                            "it on its own training text."),
            ]

    resp = EvalResponse(
        available=True, message=message, sample_size=len(picked),
        pool_size=int(full_pool), depth=TOP,
        mean_query_words=round(float(np.mean([len(r["text"].split()) for r in picked])), 1),
        results=results)
    config.ensure_dirs()
    cache.write_text(resp.model_dump_json())
    return resp
