"""Search: semantic (SigLIP text->image), keyword (FTS5 BM25 over captions +
VLM tags), or hybrid (reciprocal-rank fusion of both).

Correctness notes:
- Filters are applied INSIDE each ranking (SQL WHERE for keyword, candidate
  mask for semantic), never after a LIMIT — a selective filter can't
  accidentally empty an oversampled result list.
- If the embedding stack is unavailable, semantic/hybrid transparently degrade
  to keyword search and the response says so.
- Each result carries the caption that best explains the match (FTS best hit
  for keyword; most query-similar caption for semantic), the path(s) that
  retrieved it and their ranks, so the UI can show WHY something matched.
- Scores from different modes are not comparable (a text-image cosine and an
  RRF sum live on different scales), so every response names its score basis
  and the UI labels it. Ranks, not scores, are what fusion combines.
"""
import logging
import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from .. import config, db
from ..ml.embedder import get_embedder
from ..ml.index import get_caption_index, get_index
from ..schemas import MatchPath, SearchRequest, SearchResponse, TermStat
from .deps import (
    MAX_ID_LIST,
    SORT_KEYS,
    axis_bounds,
    build_filters,
    filtered_id_set,
    first_captions,
    get_conn,
    id_list,
    id_list_clause,
    parse_id_list,
    row_to_card,
    stage_id_list,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _keyword_ranking(
    conn, q: str, top_k: int,
    split=None, tag=None, vlm_tag=None, attr=None, axes=None, ids=None,
    ids_staged=False,
) -> tuple[list[int], dict[int, str]]:
    """Ranked sample ids + the best-matching caption per sample.
    Filters — including axis ranges — are part of the SQL, applied before LIMIT."""
    match = db.fts_escape(q)
    if not match:
        return [], {}
    where, params = build_filters(split, tag, vlm_tag, attr, axes, ids, ids_staged)
    and_where = where.replace(" WHERE ", " AND ", 1) if where else ""
    rows = conn.execute(
        "SELECT c.sample_id AS sid, MIN(rank) AS best, c.text AS caption_text "
        "FROM captions_fts f JOIN captions c ON c.id = f.rowid "
        "JOIN samples s ON s.id = c.sample_id "
        f"WHERE captions_fts MATCH ?{and_where} "
        "GROUP BY c.sample_id ORDER BY best LIMIT ?",
        [match] + params + [top_k],
    ).fetchall()
    ids = [r["sid"] for r in rows]
    best_caption = {r["sid"]: r["caption_text"] for r in rows}

    # Surface VLM-tag exact hits after caption hits (same filters).
    terms = [t.lower() for t in q.split()]
    if terms and len(ids) < top_k:
        qmarks = ",".join("?" * len(terms))
        tag_rows = conn.execute(
            "SELECT DISTINCT v.sample_id AS sid FROM vlm_tags v "
            "JOIN samples s ON s.id = v.sample_id "
            f"WHERE v.tag IN ({qmarks}){and_where}",
            terms + params,
        ).fetchall()
        seen = set(ids)
        ids += [r["sid"] for r in tag_rows if r["sid"] not in seen][: top_k - len(ids)]
    return ids, best_caption


def _term_stats(conn, q: str) -> list[TermStat]:
    """Document frequency of each content term in the query.

    Two failure modes become visible from this: a term matching nothing (which
    explains an empty result set) and a term matching a large share of the
    corpus (where BM25 has almost nothing to rank on). Counts are corpus-wide
    and stemmed, matching how the query itself is evaluated.
    """
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] or 1
    stats, seen = [], set()
    for raw in q.split():
        term = raw.strip().lower()
        if not term or term in db.STOPWORDS or term in seen:
            continue
        seen.add(term)
        match = db.fts_escape(term)
        if not match:
            continue
        n = conn.execute(
            "SELECT COUNT(DISTINCT c.sample_id) FROM captions_fts f "
            "JOIN captions c ON c.id = f.rowid WHERE captions_fts MATCH ?",
            (match,),
        ).fetchone()[0]
        stats.append(TermStat(term=term, images=n, fraction=round(n / total, 4),
                              common=n / total >= config.DF_WARN_FRACTION))
    return stats


def _sort_by_axis(conn, ids: list[int], sort: str) -> list[int]:
    """Re-order the entire retrieved set by a difficulty axis, in SQL.

    This replaces relevance order rather than refining it: someone sorting by
    difficulty wants the hardest items in the matching set, not the hardest
    among the first page. The response reports the sort so the UI can say what
    the ordering means, since the per-card scores still describe retrieval.
    """
    if sort not in SORT_KEYS or not ids:
        return ids
    axis, direction = sort.rsplit("_", 1)
    qmarks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id FROM samples WHERE id IN ({qmarks}) "
        f"ORDER BY ({axis} IS NULL), {axis} "
        f"{'ASC' if direction == 'asc' else 'DESC'}, id", ids).fetchall()
    return [r["id"] for r in rows]


def _semantic_ranking(q: str, allowed: Optional[set[int]], top_k: int):
    """Ranked (sample_id, score), or None if the embedding stack is down."""
    index = get_index()
    embedder = get_embedder() if index is not None else None
    if index is None or embedder is None:
        return None, None
    qvec = embedder.encode_texts([q])[0]
    return index.search(qvec, top_k=top_k, allowed_ids=allowed), qvec


def _best_captions_for(conn, sample_ids: list[int], qvec) -> dict[int, str]:
    """For semantic results: the caption of each sample most similar to the
    query (uses precomputed caption embeddings; cheap dot products)."""
    cap_index = get_caption_index()
    if cap_index is None or qvec is None or not sample_ids:
        return {}
    qmarks = ",".join("?" * len(sample_ids))
    rows = conn.execute(
        f"SELECT id, sample_id, text FROM captions WHERE sample_id IN ({qmarks})",
        sample_ids).fetchall()
    best: dict[int, tuple[float, str]] = {}
    for r in rows:
        vec = cap_index.vector_of(r["id"])
        if vec is None:
            continue
        score = float(vec @ qvec)
        if r["sample_id"] not in best or score > best[r["sample_id"]][0]:
            best[r["sample_id"]] = (score, r["text"])
    return {sid: text for sid, (_s, text) in best.items()}


def run_search(
    conn: sqlite3.Connection, q: str, mode: str = "hybrid", top_k: int = 60,
    split: Optional[str] = None, tag: Optional[str] = None,
    vlm_tag: Optional[str] = None, attr: Optional[str] = None,
    offset: int = 0, axes: Optional[dict] = None, sort: Optional[str] = None,
    ids: Optional[list[str]] = None,
) -> SearchResponse:
    """Core search service — used by the API endpoint, the export route, and the
    assistant's agent tools (same code path, same behavior).

    Paging contract: both rankings are taken to exactly `config.SEARCH_DEPTH`
    and fused once, then the requested window is sliced out. Fusing to the depth
    of the current page instead would change the ranking as the user pages,
    which shows up as duplicates and gaps.

    The depth is therefore a hard horizon, not a starting point: paging stops at
    SEARCH_DEPTH rather than widening to reach further. An earlier version did
    widen, and it demonstrably returned the same image on two adjacent pages —
    the ranking past row 300 was being recomputed against a different candidate
    pool. A ranking is only defined as deep as it was computed, so results
    beyond that are not offered; raise CVDE_SEARCH_DEPTH to see further.
    """
    depth = config.SEARCH_DEPTH
    # Staged once per request, before any filter builds SQL: past ~10k entries an
    # IN (...) list would exceed SQLite's host-parameter ceiling.
    ids_staged = stage_id_list(conn, ids) if ids else False
    allowed = filtered_id_set(conn, split, tag, vlm_tag, attr, axes, ids, ids_staged)
    # How many pasted entries actually exist here. Reported rather than enforced:
    # a list carried over from a bigger corpus is a normal thing to paste, and
    # the useful response is "412 of your 500 are in this dataset", not an error.
    ids_resolved = None
    if ids:
        clause, id_params = id_list_clause(ids, staged=ids_staged)
        ids_resolved = conn.execute(
            f"SELECT COUNT(*) FROM samples s WHERE {clause}", id_params
        ).fetchone()[0] if clause else 0
    degraded, message = False, None
    scores: dict[int, float] = {}
    match_captions: dict[int, str] = {}
    matched_terms: Optional[list[str]] = None
    paths: dict[int, list[MatchPath]] = {}
    score_basis: Optional[str] = None
    rrf_k: Optional[int] = None

    semantic, qvec = (None, None)
    if mode in ("semantic", "hybrid"):
        semantic, qvec = _semantic_ranking(q, allowed, depth)
        if semantic is None:
            degraded, mode = True, "keyword"
            message = ("Semantic search unavailable (embeddings not computed) — "
                       "using keyword search.")

    def record(path: str, ids) -> None:
        """Ranks are absolute within the full ranking, not within the page, so
        a card on page 3 still reports the rank the user would count to."""
        for rank, sid in enumerate(ids):
            paths.setdefault(sid, []).append(MatchPath(path=path, rank=rank + 1))

    if mode == "semantic":
        ranked = [sid for sid, _ in semantic]
        scores = dict(semantic)
        score_basis = "cosine"
        record("semantic", ranked)
    elif mode == "keyword":
        ranked, match_captions = _keyword_ranking(
            conn, q, depth, split, tag, vlm_tag, attr, axes, ids, ids_staged)
        record("keyword", ranked)
        matched_terms = [t for t in q.split() if t.strip()]
    else:  # hybrid: reciprocal-rank fusion
        keyword, kw_captions = _keyword_ranking(
            conn, q, depth, split, tag, vlm_tag, attr, axes, ids, ids_staged)
        rrf_k = config.RRF_K
        fused: dict[int, float] = {}
        for rank, (sid, _s) in enumerate(semantic):
            fused[sid] = fused.get(sid, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, sid in enumerate(keyword):
            fused[sid] = fused.get(sid, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = sorted(fused, key=lambda sid: -fused[sid])
        scores = fused
        score_basis = "rrf"
        record("semantic", [sid for sid, _ in semantic])
        record("keyword", keyword)
        match_captions = dict(kw_captions)
        matched_terms = [t for t in q.split() if t.strip()]
        logger.debug("Fused %d semantic + %d keyword results with RRF k=%d",
                     len(semantic), len(keyword), rrf_k)

    term_stats = _term_stats(conn, q) if mode in ("keyword", "hybrid") else []
    if sort:
        ranked = _sort_by_axis(conn, ranked, sort)
    # Clamped to the horizon: never serve, or promise, rows the fusion did not rank.
    end = min(offset + top_k, depth)
    window = ranked[offset:end] if offset < depth else []
    has_more = end < min(len(ranked), depth)
    depth_reached = not has_more and len(ranked) >= depth

    # Caption lookups are per-page, not per-ranking: only the window is shown.
    if mode in ("semantic", "hybrid"):
        match_captions = {**_best_captions_for(conn, window, qvec), **match_captions}

    if not window:
        return SearchResponse(items=[], mode_used=mode, degraded=degraded,
                              message=message, score_basis=score_basis,
                              rrf_k=rrf_k, term_stats=term_stats,
                              offset=offset, has_more=False, sort=sort,
                              ids_resolved=ids_resolved, depth_limit=depth,
                              depth_reached=offset >= depth)

    qmarks = ",".join("?" * len(window))
    rows = {r["id"]: r for r in conn.execute(
        f"SELECT * FROM samples WHERE id IN ({qmarks})", window)}
    captions = first_captions(conn, window)
    items = []
    for sid in window:
        if sid not in rows:
            continue
        card = row_to_card(rows[sid], caption=captions.get(sid), score=scores.get(sid))
        card.match_caption = match_captions.get(sid)
        card.matched_terms = matched_terms
        card.match_paths = paths.get(sid)
        items.append(card)
    return SearchResponse(items=items, mode_used=mode, degraded=degraded,
                          message=message, score_basis=score_basis, rrf_k=rrf_k,
                          term_stats=term_stats, offset=offset, has_more=has_more,
                          sort=sort, ids_resolved=ids_resolved,
                          depth_limit=depth, depth_reached=depth_reached)


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1),
    mode: str = Query("hybrid", pattern="^(semantic|keyword|hybrid)$"),
    top_k: int = Query(60, ge=1, le=200),
    offset: int = Query(0, ge=0, le=5000),
    sort: Optional[str] = Query(None, description="<axis>_asc | <axis>_desc"),
    axes: dict = Depends(axis_bounds),
    ids: list = Depends(id_list),
    split: Optional[str] = None,
    tag: Optional[str] = None,
    vlm_tag: Optional[str] = None,
    attr: Optional[str] = None,
    conn: sqlite3.Connection = Depends(get_conn),
):
    return run_search(conn, q, mode=mode, top_k=top_k, split=split, tag=tag,
                      vlm_tag=vlm_tag, attr=attr, offset=offset, axes=axes,
                      sort=sort, ids=ids)


@router.post("/search", response_model=SearchResponse)
def search_post(body: SearchRequest, conn: sqlite3.Connection = Depends(get_conn)):
    """Same search, as a body.

    A URL cannot carry an id list of the size this filter is meant for — sixty
    thousand entries is roughly 400 kB of query string — so a pasted set that
    large has to arrive as a POST. Identical semantics to the GET otherwise:
    both call `run_search`, so there is one ranking implementation, not two.
    """
    entries = parse_id_list(body.ids)
    if len(entries) > MAX_ID_LIST:
        raise HTTPException(
            400, f"Too many entries: {len(entries)}. The limit is {MAX_ID_LIST}.")
    axes = {a: (b.get("min"), b.get("max")) for a, b in (body.axes or {}).items()}
    return run_search(conn, body.q, mode=body.mode, top_k=body.top_k,
                      split=body.split, tag=body.tag, vlm_tag=body.vlm_tag,
                      attr=body.attr, offset=body.offset, axes=axes,
                      sort=body.sort, ids=entries)
