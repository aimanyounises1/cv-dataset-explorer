"""Search: semantic (SigLIP text->image), keyword (FTS5 BM25 over captions +
VLM tags), or hybrid (reciprocal-rank fusion of both).

If the embedding stack is unavailable, semantic/hybrid transparently degrade
to keyword search and the response says so — the UI surfaces the message
instead of erroring.
"""
import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, Query

from .. import db
from ..ml.embedder import get_embedder
from ..ml.index import get_index
from ..schemas import SearchResponse
from .deps import filtered_id_set, first_captions, get_conn, row_to_card

router = APIRouter()

RRF_K = 60  # standard reciprocal-rank-fusion constant


def _keyword_ranking(conn, q: str, allowed: Optional[set[int]], top_k: int) -> list[int]:
    match = db.fts_escape(q)
    if not match:
        return []
    # Best (lowest) BM25 rank across a sample's captions.
    rows = conn.execute(
        "SELECT c.sample_id AS sid, MIN(rank) AS best "
        "FROM captions_fts f JOIN captions c ON c.id = f.rowid "
        "WHERE captions_fts MATCH ? GROUP BY c.sample_id ORDER BY best LIMIT ?",
        (match, top_k * 3),
    ).fetchall()
    ids = [r["sid"] for r in rows]
    # Also surface VLM-tag matches (exact term hits) after caption hits.
    terms = [t.lower() for t in q.split()]
    if terms:
        qmarks = ",".join("?" * len(terms))
        tag_rows = conn.execute(
            f"SELECT DISTINCT sample_id FROM vlm_tags WHERE tag IN ({qmarks})", terms).fetchall()
        seen = set(ids)
        ids += [r["sample_id"] for r in tag_rows if r["sample_id"] not in seen]
    if allowed is not None:
        ids = [i for i in ids if i in allowed]
    return ids[:top_k]


def _semantic_ranking(q: str, allowed: Optional[set[int]], top_k: int):
    index = get_index()
    embedder = get_embedder() if index is not None else None
    if index is None or embedder is None:
        return None  # unavailable
    qvec = embedder.encode_texts([q])[0]
    results = index.search(qvec, top_k=top_k * 3 if allowed else top_k)
    if allowed is not None:
        results = [(sid, s) for sid, s in results if sid in allowed]
    return results[:top_k]


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1),
    mode: str = Query("hybrid", pattern="^(semantic|keyword|hybrid)$"),
    top_k: int = Query(60, ge=1, le=200),
    split: Optional[str] = None,
    tag: Optional[str] = None,
    vlm_tag: Optional[str] = None,
    conn: sqlite3.Connection = Depends(get_conn),
):
    allowed = filtered_id_set(conn, split, tag, vlm_tag)
    degraded, message = False, None
    scores: dict[int, float] = {}

    semantic = _semantic_ranking(q, allowed, top_k) if mode in ("semantic", "hybrid") else None
    if mode in ("semantic", "hybrid") and semantic is None:
        degraded, mode = True, "keyword"
        message = "Semantic search unavailable (embeddings not computed) — using keyword search."

    if mode == "semantic":
        ordered = [sid for sid, _ in semantic]
        scores = dict(semantic)
    elif mode == "keyword":
        ordered = _keyword_ranking(conn, q, allowed, top_k)
    else:  # hybrid: reciprocal-rank fusion
        keyword = _keyword_ranking(conn, q, allowed, top_k)
        fused: dict[int, float] = {}
        for rank, (sid, _s) in enumerate(semantic):
            fused[sid] = fused.get(sid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, sid in enumerate(keyword):
            fused[sid] = fused.get(sid, 0.0) + 1.0 / (RRF_K + rank + 1)
        ordered = sorted(fused, key=lambda sid: -fused[sid])[:top_k]
        scores = fused

    if not ordered:
        return SearchResponse(items=[], mode_used=mode, degraded=degraded, message=message)

    qmarks = ",".join("?" * len(ordered))
    rows = {r["id"]: r for r in conn.execute(
        f"SELECT * FROM samples WHERE id IN ({qmarks})", ordered)}
    captions = first_captions(conn, ordered)
    items = [row_to_card(rows[sid], caption=captions.get(sid), score=scores.get(sid))
             for sid in ordered if sid in rows]
    return SearchResponse(items=items, mode_used=mode, degraded=degraded, message=message)
