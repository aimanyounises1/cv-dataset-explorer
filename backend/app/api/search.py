"""Search: semantic (SigLIP text->image), keyword (FTS5 BM25 over captions +
VLM tags), or hybrid (reciprocal-rank fusion of both).

Correctness notes:
- Filters are applied INSIDE each ranking (SQL WHERE for keyword, candidate
  mask for semantic), never after a LIMIT — a selective filter can't
  accidentally empty an oversampled result list.
- If the embedding stack is unavailable, semantic/hybrid transparently degrade
  to keyword search and the response says so.
- Each result carries the caption that best explains the match (FTS best hit
  for keyword; most query-similar caption for semantic) so the UI can show
  WHY something matched.
"""
import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, Query

from .. import db
from ..ml.embedder import get_embedder
from ..ml.index import get_caption_index, get_index
from ..schemas import SearchResponse
from .deps import build_filters, filtered_id_set, first_captions, get_conn, row_to_card

router = APIRouter()

RRF_K = 60  # standard reciprocal-rank-fusion constant


def _keyword_ranking(
    conn, q: str, top_k: int,
    split=None, tag=None, vlm_tag=None, attr=None,
) -> tuple[list[int], dict[int, str]]:
    """Ranked sample ids + the best-matching caption per sample.
    Filters are part of the SQL, applied before LIMIT."""
    match = db.fts_escape(q)
    if not match:
        return [], {}
    where, params = build_filters(split, tag, vlm_tag, attr)
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
) -> SearchResponse:
    """Core search service — used by the API endpoint and by the assistant's
    agent tools (same code path, same behavior)."""
    allowed = filtered_id_set(conn, split, tag, vlm_tag, attr)
    degraded, message = False, None
    scores: dict[int, float] = {}
    match_captions: dict[int, str] = {}
    matched_terms: Optional[list[str]] = None

    semantic, qvec = (None, None)
    if mode in ("semantic", "hybrid"):
        semantic, qvec = _semantic_ranking(q, allowed, top_k)
        if semantic is None:
            degraded, mode = True, "keyword"
            message = ("Semantic search unavailable (embeddings not computed) — "
                       "using keyword search.")

    if mode == "semantic":
        ordered = [sid for sid, _ in semantic]
        scores = dict(semantic)
        match_captions = _best_captions_for(conn, ordered, qvec)
    elif mode == "keyword":
        ordered, match_captions = _keyword_ranking(
            conn, q, top_k, split, tag, vlm_tag, attr)
        matched_terms = [t for t in q.split() if t.strip()]
    else:  # hybrid: reciprocal-rank fusion
        keyword, kw_captions = _keyword_ranking(
            conn, q, top_k, split, tag, vlm_tag, attr)
        fused: dict[int, float] = {}
        for rank, (sid, _s) in enumerate(semantic):
            fused[sid] = fused.get(sid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, sid in enumerate(keyword):
            fused[sid] = fused.get(sid, 0.0) + 1.0 / (RRF_K + rank + 1)
        ordered = sorted(fused, key=lambda sid: -fused[sid])[:top_k]
        scores = fused
        match_captions = {**_best_captions_for(conn, ordered, qvec), **kw_captions}
        matched_terms = [t for t in q.split() if t.strip()]

    if not ordered:
        return SearchResponse(items=[], mode_used=mode, degraded=degraded, message=message)

    qmarks = ",".join("?" * len(ordered))
    rows = {r["id"]: r for r in conn.execute(
        f"SELECT * FROM samples WHERE id IN ({qmarks})", ordered)}
    captions = first_captions(conn, ordered)
    items = []
    for sid in ordered:
        if sid not in rows:
            continue
        card = row_to_card(rows[sid], caption=captions.get(sid), score=scores.get(sid))
        card.match_caption = match_captions.get(sid)
        card.matched_terms = matched_terms
        items.append(card)
    return SearchResponse(items=items, mode_used=mode, degraded=degraded, message=message)


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1),
    mode: str = Query("hybrid", pattern="^(semantic|keyword|hybrid)$"),
    top_k: int = Query(60, ge=1, le=200),
    split: Optional[str] = None,
    tag: Optional[str] = None,
    vlm_tag: Optional[str] = None,
    attr: Optional[str] = None,
    conn: sqlite3.Connection = Depends(get_conn),
):
    return run_search(conn, q, mode=mode, top_k=top_k, split=split, tag=tag,
                      vlm_tag=vlm_tag, attr=attr)
