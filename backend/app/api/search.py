"""Search: semantic (text->image through the active embedding provider),
keyword (FTS5 BM25 over captions + VLM tags), or hybrid (reciprocal-rank
fusion of both).

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
import io
import json
import logging
import re
import sqlite3
from typing import Optional, Union

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from PIL import Image as PILImage

from .. import config, db
from ..ml import hubness
from ..ml.index import get_caption_index, get_index

# Provider-aware: the active retrieval provider's encoder (Qwen3-VL or SigLIP)
# behind the same duck type. eval.py and hubness reach the model through this
# name, so the whole query path switches provider in one place.
from ..ml.providers import get_encoder as get_embedder
from ..schemas import (
    AnnotationSearchRequest,
    ComposedSearchRequest,
    MatchPath,
    RegionSearchRequest,
    SampleCard,
    ScenarioGroup,
    ScenarioResponse,
    SearchRequest,
    SearchResponse,
    TermStat,
)
from .deps import (
    MAX_ID_LIST,
    MAX_SQLITE_INT,
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
    ids_staged=False, max_agreement=None, album=None,
) -> tuple[list[int], dict[int, str]]:
    """Ranked sample ids + the best-matching caption per sample.
    Filters — including axis ranges — are part of the SQL, applied before LIMIT."""
    match = db.fts_escape(q)
    if not match:
        return [], {}
    where, params = build_filters(split, tag, vlm_tag, attr, axes, ids, ids_staged,
                                  max_agreement, album=album)
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


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_query_text(q: str) -> str:
    """Lowercase and strip punctuation before the text goes to SigLIP.

    SigLIP 2 tokenizes with a *case-sensitive* Gemma SentencePiece vocabulary,
    and `AutoProcessor` does not apply the model's canonical lowercase+depunctuate
    preprocessing for you. A leading capital is therefore its own rare token
    rather than part of the sentence, and it measurably moves the query vector:
    on the 1,000-caption benchmark, encoding the raw text scores R@1 46.0% and
    encoding the normalized text scores 53.2% (MRR 0.5672 -> 0.6280). Lowercasing
    the first character alone recovers 4.6 of those 7.2 points.

    This is query-side only, and it is a no-op on text that is already lowercase
    with no punctuation — the short phrases users actually type are returned
    unchanged, so nothing that works today can regress. The corpus-side caption
    vectors stay as ingested: matching a normalized query against them measured
    *better* than matching a raw one (caption-retrieval MRR 0.4387 -> 0.4572),
    so there is no need to re-embed.

    Not applied to the lexical path: FTS5 does its own tokenization, and the
    conjunctive query is measured against raw text.
    """
    out = _SPACES.sub(" ", _PUNCT.sub(" ", q.lower())).strip()
    # A query that is nothing but punctuation would normalize away entirely;
    # embedding the original is more useful than embedding "".
    return out or q


def _encode_for_bank(embedder):
    """The bank encoder handed to `hubness.build`.

    Defined here, next to `normalize_query_text`, so the hubness bank is encoded
    through exactly the seam a user's query is. A bank built from the stored
    caption vectors — which encode the RAW caption text — measured no gain at
    all, so this is not a stylistic preference.
    """
    def encode(texts: list[str]):
        prompts = [normalize_query_text(t) for t in texts]
        return np.concatenate(
            [embedder.encode_texts(prompts[i:i + 128])
             for i in range(0, len(prompts), 128)], axis=0)
    return encode


def _semantic_ranking(conn, q: str, allowed: Optional[set[int]], top_k: int):
    """Ranked (sample_id, score), the query vector, and the score's basis.

    The basis is returned rather than assumed because the hubness correction
    changes what the number means: with a penalty applied it is a cosine minus a
    per-image constant, which is still comparable across results of the same
    query but is no longer a cosine. Reporting it as one would be a lie the UI
    then prints.
    """
    index = get_index()
    embedder = get_embedder() if index is not None else None
    if index is None or embedder is None:
        return None, None, None
    qvec = embedder.encode_texts([normalize_query_text(q)])[0]
    penalty = hubness.get_penalty(conn, _encode_for_bank(embedder))
    ranked = index.search(qvec, top_k=top_k, allowed_ids=allowed, penalty=penalty)
    return ranked, qvec, ("cosine" if penalty is None else "cosine_adj")


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
    vlm_tag: Optional[str] = None,
    attr: Optional[Union[str, list[str]]] = None,
    offset: int = 0, axes: Optional[dict] = None, sort: Optional[str] = None,
    ids: Optional[list[str]] = None, max_agreement: Optional[float] = None,
    album: Optional[int] = None,
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
    allowed = filtered_id_set(conn, split, tag, vlm_tag, attr, axes, ids, ids_staged,
                              max_agreement, album=album)
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

    semantic, qvec, semantic_basis = (None, None, None)
    if mode in ("semantic", "hybrid"):
        semantic, qvec, semantic_basis = _semantic_ranking(conn, q, allowed, depth)
        if semantic is None:
            degraded, mode = True, "keyword"
            # Appended, not assigned: a request that degraded more than once
            # should say the whole story, not just the last step.
            message = ((message + " ") if message else "") + (
                "Semantic search unavailable (embeddings not computed) — "
                "using keyword search.")

    def record(path: str, ids) -> None:
        """Ranks are absolute within the full ranking, not within the page, so
        a card on page 3 still reports the rank the user would count to."""
        for rank, sid in enumerate(ids):
            paths.setdefault(sid, []).append(MatchPath(path=path, rank=rank + 1))

    if mode == "semantic":
        ranked = [sid for sid, _ in semantic]
        scores = dict(semantic)
        score_basis = semantic_basis
        record("semantic", ranked)
    elif mode == "keyword":
        ranked, match_captions = _keyword_ranking(
            conn, q, depth, split, tag, vlm_tag, attr, axes, ids, ids_staged,
            max_agreement, album)
        record("keyword", ranked)
        # The terms that actually constrained the match, so highlighting
        # marks what was searched for rather than every word typed.
        matched_terms = db.match_terms(q)
    else:  # hybrid: reciprocal-rank fusion
        keyword, kw_captions = _keyword_ranking(
            conn, q, depth, split, tag, vlm_tag, attr, axes, ids, ids_staged,
            max_agreement, album)
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
        # The terms that actually constrained the match, so highlighting
        # marks what was searched for rather than every word typed.
        matched_terms = db.match_terms(q)
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
    max_agreement: Optional[float] = Query(
        None, ge=0.0, le=1.0, allow_inf_nan=False,
        description="Samples with any caption at or below this agreement"),
    axes: dict = Depends(axis_bounds),
    ids: list = Depends(id_list),
    split: Optional[str] = None,
    tag: Optional[str] = None,
    vlm_tag: Optional[str] = None,
    attr: Optional[list[str]] = Query(
        None, description="Attribute facet 'group:label'. Repeatable: several "
                          "are intersected."),
    album: Optional[int] = Query(None, ge=1, le=MAX_SQLITE_INT,
                                 description="Restrict candidates to members "
                                             "of this album"),
    conn: sqlite3.Connection = Depends(get_conn),
):
    return run_search(conn, q, mode=mode, top_k=top_k, split=split, tag=tag,
                      vlm_tag=vlm_tag, attr=attr, offset=offset, axes=axes,
                      sort=sort, ids=ids, max_agreement=max_agreement,
                      album=album)


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
                      sort=body.sort, ids=entries,
                      max_agreement=body.max_agreement, album=body.album)


# A query image is one file, so it needs no form envelope — the raw request
# body avoids the multipart parser, which is a dependency this project does
# not carry. Large enough for any photograph, small enough that a mistaken
# video upload fails fast.
MAX_QUERY_IMAGE_BYTES = 8 * 1024 * 1024


@router.post("/search/by-image", response_model=list[SampleCard])
async def search_by_image(
    request: Request,
    top_k: int = Query(24, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Rank the corpus against an uploaded image: image-to-image retrieval on
    the same embeddings and the same exact index the text search uses.

    Unlike every other search, this one cannot live in a URL — the query is
    the image itself. The result ids can: the UI offers the ranked set as an
    `?ids=` slice, which is the shareable artifact. Scores are plain cosines
    (no hubness penalty): the penalty bank is calibrated on text queries, and
    an uploaded image is not one.
    """
    index = get_index()
    embedder = get_embedder()
    if index is None or embedder is None:
        raise HTTPException(503, "Embeddings not computed yet — run `python -m app.ingest`.")
    body = await request.body()
    if not body:
        raise HTTPException(400, "Send the image bytes as the request body")
    if len(body) > MAX_QUERY_IMAGE_BYTES:
        raise HTTPException(
            413, f"Query image over {MAX_QUERY_IMAGE_BYTES // (1024 * 1024)} MB")
    try:
        img = PILImage.open(io.BytesIO(body))
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(400, "The request body is not a decodable image") from None
    vec = embedder.encode_images([img])[0]
    results = index.search(vec, top_k=top_k)
    ids = [sid for sid, _ in results]
    if not ids:
        return []
    qmarks = ",".join("?" * len(ids))
    rows = {r["id"]: r for r in conn.execute(
        f"SELECT * FROM samples WHERE id IN ({qmarks})", ids)}
    captions = first_captions(conn, ids)
    return [row_to_card(rows[sid], caption=captions.get(sid), score=score)
            for sid, score in results if sid in rows]


# -- composed search ----------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _composed_ranking(conn, body: ComposedSearchRequest):
    """The full composed ranking as (ranked, qvec, None), or (None, None,
    message) when the embedding stack is down.

    Query vector: the normalized mean of (a) the unit text vector, encoded
    through the same `normalize_query_text` seam every text query uses, and
    (b) the unit mean of the positive example image vectors — equal weights,
    either part optional. Negative examples subtract
    0.5 * max(0, max_i cos(img, neg_i)) from each candidate's positive-side
    cosine. The 0.5 is CHOSEN, NOT TUNED: half-weight repulsion, picked for
    being simple to state, with no benchmark run to prefer it over its
    neighbours. The resulting scores carry basis `composed` and are only
    comparable within one response.

    Filters restrict candidates inside the ranking (the allowed-id mask),
    never after a window — and there is no keyword lane here to thread them
    into a second time.
    """
    index = get_index()
    embedder = get_embedder() if index is not None else None
    if index is None or embedder is None:
        return None, None, ("Composed ranking unavailable (embeddings not "
                            "computed) — run `python -m app.ingest`.")

    def vec_of(sid: int, side: str) -> np.ndarray:
        v = index.vector_of(sid)
        if v is None:
            raise HTTPException(
                404, f"{side} example {sid} is not in the embedding index")
        return v

    pos = [vec_of(s, "Positive") for s in body.positive_ids]
    neg = [vec_of(s, "Negative") for s in body.negative_ids]
    parts = []
    if body.text and body.text.strip():
        parts.append(_unit(embedder.encode_texts([normalize_query_text(body.text)])[0]))
    if pos:
        parts.append(_unit(np.mean(np.stack(pos), axis=0)))

    axes = {a: (b.get("min"), b.get("max")) for a, b in (body.axes or {}).items()}
    allowed = filtered_id_set(conn, body.split, body.tag, body.vlm_tag, body.attr,
                              axes, None, False, body.max_agreement, None,
                              body.album)
    if parts:
        qvec = _unit(np.mean(np.stack(parts), axis=0))
        scores = index.embeddings @ qvec
        if neg:
            sims = index.embeddings @ np.stack(neg).T
            scores = scores - 0.5 * np.maximum(0.0, sims.max(axis=1))
    else:
        # Negative-only: nothing to steer toward, so the ranking IS the
        # distance from the excluded examples — unclamped, or the whole far
        # hemisphere would tie at a zero penalty and the order would be
        # arbitrary. The excluded images themselves cannot appear in a search
        # away from themselves.
        qvec = None
        sims = index.embeddings @ np.stack(neg).T
        scores = -sims.max(axis=1)
        neg_rows = [r for r in (index.row_of(s) for s in body.negative_ids)
                    if r is not None]
        scores[neg_rows] = -np.inf
    if allowed is not None:
        allowed_arr = np.fromiter(allowed, dtype=index.ids.dtype,
                                  count=len(allowed))
        scores = np.where(np.isin(index.ids, allowed_arr), scores, -np.inf)
    order = np.argsort(-scores, kind="stable")
    ranked = [(int(index.ids[i]), float(scores[i])) for i in order
              if np.isfinite(scores[i])]
    return ranked, qvec, None


def run_composed(conn: sqlite3.Connection, body: ComposedSearchRequest) -> SearchResponse:
    ranked, qvec, down = _composed_ranking(conn, body)
    if ranked is None:
        if body.text and body.text.strip():
            # The honest fallback run_search's own degradation uses: rank what
            # can still be ranked and say exactly what was lost.
            axes = {a: (b.get("min"), b.get("max"))
                    for a, b in (body.axes or {}).items()}
            out = run_search(conn, body.text, mode="keyword", top_k=body.top_k,
                             split=body.split, tag=body.tag, vlm_tag=body.vlm_tag,
                             attr=body.attr, offset=body.offset, axes=axes,
                             max_agreement=body.max_agreement, album=body.album)
            out.degraded = True
            out.message = (down + " Ranked the text by keyword instead; "
                           "example images were ignored.")
            return out
        return SearchResponse(items=[], mode_used="composed", degraded=True,
                              message=down, offset=body.offset)

    # No fused horizon: this is one exact pass over the whole filtered corpus,
    # so the depth is the candidate count and paging never re-ranks.
    depth = len(ranked)
    end = min(body.offset + body.top_k, depth)
    window = [sid for sid, _ in ranked[body.offset:end]]
    scores = dict(ranked)
    has_more = end < depth
    if not window:
        return SearchResponse(items=[], mode_used="composed", degraded=False,
                              score_basis="composed", offset=body.offset,
                              has_more=False, depth_limit=depth,
                              depth_reached=body.offset >= depth)
    match_captions = _best_captions_for(conn, window, qvec)
    qmarks = ",".join("?" * len(window))
    rows = {r["id"]: r for r in conn.execute(
        f"SELECT * FROM samples WHERE id IN ({qmarks})", window)}
    captions = first_captions(conn, window)
    items = []
    for rank0, sid in enumerate(window):
        if sid not in rows:
            continue
        card = row_to_card(rows[sid], caption=captions.get(sid),
                           score=round(scores[sid], 4))
        card.match_caption = match_captions.get(sid)
        card.match_paths = [MatchPath(path="composed", rank=body.offset + rank0 + 1)]
        items.append(card)
    return SearchResponse(items=items, mode_used="composed", score_basis="composed",
                          offset=body.offset, has_more=has_more,
                          depth_limit=depth, depth_reached=end >= depth,
                          # qvec is None exactly when the query was negative-only:
                          # say what this ranking IS, because "results for
                          # nothing, minus X" is not guessable from the grid.
                          message=(None if qvec is not None else
                                   "Ranked by distance from the excluded "
                                   "example(s) — add text or a positive "
                                   "reference to steer toward something."))


@router.post("/search/by-region", response_model=SearchResponse)
def search_by_region(body: RegionSearchRequest,
                     conn: sqlite3.Connection = Depends(get_conn)):
    """A region of an existing sample as search evidence — positive ("more
    like this crop", optionally blended with steering text through the same
    composed arithmetic) or negative ("away from this crop").

    The server crops the original file from the request's normalized geometry,
    so the query is reproducible without shipping pixels. Scores carry basis
    `composed`; the negative role ranks by distance, unclamped, and never
    returns the source sample it was told to move away from.
    """
    index = get_index()
    embedder = get_embedder()
    if index is None or embedder is None:
        raise HTTPException(503, "Embeddings not computed yet — run `python -m app.ingest`.")
    row = conn.execute("SELECT filename FROM samples WHERE id = ?",
                       (body.sample_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Sample not found")
    path = config.IMAGES_DIR / row["filename"]
    try:
        img = PILImage.open(path).convert("RGB")
    except OSError as exc:
        raise HTTPException(503, f"Image file unreadable: {row['filename']}") from exc
    W, H = img.size
    box = (int(body.x * W), int(body.y * H),
           max(int((body.x + body.w) * W), int(body.x * W) + 8),
           max(int((body.y + body.h) * H), int(body.y * H) + 8))
    crop_vec = _unit(embedder.encode_images([img.crop(box)])[0])

    if body.role == "positive":
        parts = [crop_vec]
        if body.text and body.text.strip():
            parts.append(_unit(
                embedder.encode_texts([normalize_query_text(body.text)])[0]))
        qvec = _unit(np.mean(np.stack(parts), axis=0))
        scores = index.embeddings @ qvec
        message = None
    else:
        qvec = None
        scores = -(index.embeddings @ crop_vec)
        src = index.row_of(body.sample_id)
        if src is not None:
            scores[src] = -np.inf
        message = ("Ranked by distance from the marked region — add text or "
                   "a positive reference to steer toward something.")

    order = np.argsort(-scores, kind="stable")
    ranked = [(int(index.ids[i]), float(scores[i])) for i in order
              if np.isfinite(scores[i])]
    depth = len(ranked)
    end = min(body.offset + body.top_k, depth)
    window = [sid for sid, _ in ranked[body.offset:end]]
    if not window:
        return SearchResponse(items=[], mode_used="composed", degraded=False,
                              score_basis="composed", offset=body.offset,
                              has_more=False, depth_limit=depth,
                              depth_reached=True, message=message)
    qmarks = ",".join("?" * len(window))
    rows = {r["id"]: r for r in conn.execute(
        f"SELECT * FROM samples WHERE id IN ({qmarks})", window)}
    captions = first_captions(conn, window)
    smap = dict(ranked)
    items = []
    for rank0, sid in enumerate(window):
        if sid not in rows:
            continue
        card = row_to_card(rows[sid], caption=captions.get(sid),
                           score=round(smap[sid], 4))
        card.match_paths = [MatchPath(path="composed", rank=body.offset + rank0 + 1)]
        items.append(card)
    return SearchResponse(items=items, mode_used="composed",
                          score_basis="composed", offset=body.offset,
                          has_more=end < depth, depth_limit=depth,
                          depth_reached=end >= depth, message=message)


def run_annotation_search(
    conn: sqlite3.Connection,
    body: AnnotationSearchRequest,
) -> SearchResponse:
    """Rank full images against one persisted masked object.

    When the annotation has a semantic leaf label, its unit text vector and the
    unit masked-object vector receive equal weight through the same normalized
    mean used by composed region search. This is query-time evidence against the
    existing full-image index, not a precomputed object-patch index.
    """
    index = get_index()
    embedder = get_embedder() if index is not None else None
    if index is None or embedder is None:
        raise HTTPException(
            503, "Embeddings not computed yet — run `python -m app.ingest`.")
    row = conn.execute(
        "SELECT a.sample_id, a.geometry, s.filename, m.png, m.width, m.height, "
        "l.name AS label_name "
        "FROM annotations a "
        "JOIN samples s ON s.id = a.sample_id "
        "LEFT JOIN annotation_masks m ON m.annotation_id = a.id "
        "LEFT JOIN annotation_object_labels al ON al.annotation_id = a.id "
        "LEFT JOIN object_labels l ON l.id = al.label_id "
        "WHERE a.id = ?",
        (body.annotation_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Annotation not found")
    if row["png"] is None:
        raise HTTPException(422, "Annotation has no persisted mask")
    path = config.IMAGES_DIR / row["filename"]
    try:
        image = PILImage.open(path).convert("RGB")
        mask = PILImage.open(io.BytesIO(row["png"])).convert("L")
    except OSError as exc:
        raise HTTPException(503, "Annotation image or mask is unreadable") from exc
    if mask.size != image.size or mask.size != (row["width"], row["height"]):
        raise HTTPException(503, "Persisted mask dimensions do not match its image")
    try:
        bbox = json.loads(row["geometry"])
        x0 = int(bbox["x"] * image.width)
        y0 = int(bbox["y"] * image.height)
        x1 = max(int((bbox["x"] + bbox["w"]) * image.width), x0 + 1)
        y1 = max(int((bbox["y"] + bbox["h"]) * image.height), y0 + 1)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(503, "Persisted mask bbox is invalid") from exc
    crop_box = (max(x0, 0), max(y0, 0),
                min(x1, image.width), min(y1, image.height))
    crop = image.crop(crop_box)
    alpha = mask.crop(crop_box)
    neutral = PILImage.new("RGB", crop.size, (128, 128, 128))
    masked_object = PILImage.composite(crop, neutral, alpha)
    parts = [_unit(embedder.encode_images([masked_object])[0])]
    label_name = row["label_name"]
    if label_name:
        parts.append(_unit(embedder.encode_texts(
            [normalize_query_text(label_name)])[0]))
    query = _unit(np.mean(np.stack(parts), axis=0))
    scores = index.embeddings @ query
    source_row = index.row_of(row["sample_id"])
    if source_row is not None:
        scores[source_row] = -np.inf
    order = np.argsort(-scores, kind="stable")
    ranked = [(int(index.ids[i]), float(scores[i])) for i in order
              if np.isfinite(scores[i])]
    depth = len(ranked)
    end = min(body.offset + body.top_k, depth)
    window = [sid for sid, _ in ranked[body.offset:end]]
    basis = "segment_composed" if label_name else "segment_cosine"
    message = (
        f"Masked-object image evidence blended equally with leaf label "
        f"{label_name!r}; ranked against the full-image index."
        if label_name else
        "Masked-object image evidence ranked against the full-image index."
    )
    if not window:
        return SearchResponse(
            items=[], mode_used="segment", score_basis=basis,
            offset=body.offset, has_more=False, depth_limit=depth,
            depth_reached=True, message=message)
    qmarks = ",".join("?" * len(window))
    rows = {r["id"]: r for r in conn.execute(
        f"SELECT * FROM samples WHERE id IN ({qmarks})", window)}
    captions = first_captions(conn, window)
    score_map = dict(ranked)
    items = []
    for rank0, sample_id in enumerate(window):
        sample = rows.get(sample_id)
        if sample is None:
            continue
        card = row_to_card(
            sample, caption=captions.get(sample_id),
            score=round(score_map[sample_id], 4))
        card.match_paths = [
            MatchPath(path="segment", rank=body.offset + rank0 + 1)]
        items.append(card)
    return SearchResponse(
        items=items, mode_used="segment", score_basis=basis,
        offset=body.offset, has_more=end < depth, depth_limit=depth,
        depth_reached=end >= depth, message=message)


@router.post("/search/by-annotation", response_model=SearchResponse)
def search_by_annotation(
    body: AnnotationSearchRequest,
    conn: sqlite3.Connection = Depends(get_conn),
):
    return run_annotation_search(conn, body)


@router.post("/search/composed", response_model=SearchResponse)
def search_composed(body: ComposedSearchRequest,
                    conn: sqlite3.Connection = Depends(get_conn)):
    """Text and example images fused into one ranking, with negative examples
    pushing results away. See `_composed_ranking` for the arithmetic and for
    why the negative weight is described as chosen, not tuned."""
    return run_composed(conn, body)


# -- scenario groups ----------------------------------------------------------

SCENARIO_DEPTH = 200
SCENARIO_MIN_RESULTS = 8
# A label part must describe a MAJORITY of its group. A title is read as a claim
# about the images under it — "water · black" says these are the black ones — so
# a part that holds for a large minority mislabels the rest. At 0.35 a group of
# 80 titled "water · black" carried 45 black ones and 35 that were not, which is
# what a reader sees as a misclassification rather than a summary.
SCENARIO_MIN_SHARE = 0.5
# ...and it must be this much more common here than across the whole ranked pool.
# The same group's "black" ran 56% here against 51% across the results — a lift
# of 1.10, which distinguishes nothing while sounding like a category.
# Swept against five real queries: 1.25 drops the overclaiming parts and leaves
# the distinguishing ones (biker/dirt, trick, indoor, night); 1.5 starts
# dissolving good labels into "mixed". Chosen against that sweep, not tuned.
SCENARIO_MIN_LIFT = 1.25
SCENARIO_MAX_PARTS = 3
# How much more distinctive a looser trait must be to outrank a tighter one.
#
# The three sources are not equally good evidence about what separates a
# cluster. An attribute label is a corpus-wide facet applied to every image the
# same way, so "environment:snow" names an axis a researcher can filter, export
# and reason about. A caption word is one annotator's wording in one of five
# captions, so "white" can win on lift while naming the snow it was describing —
# a group titled by a colour term tells you nothing you can act on. At
# comparable lift the facet is the better description, so each step away from
# structure must clear this factor to lead a label. Ranking only: the evidence
# line still quotes the raw counts and shares, unchanged.
SCENARIO_SOURCE_MARGIN = 1.4
SCENARIO_BASIS = ("k-means over the top-200 composed ranking; each label names the traits most "
                  "over-represented in its group versus the whole result set — counted, "
                  "never generated")


# Grammar, not scenery. These pass db.STOPWORDS on purpose — that list serves
# SEARCH, where "through" is a legitimate query term and dropping it would
# change what the FTS index answers. Titling a group is a different job:
# "through · running · grass" names a preposition, not a scenario. Label-only,
# so no search behaviour moves. Content words a caption uses to describe a
# scene (running, grass, snowy) are deliberately absent.
SCENARIO_LABEL_SKIP = frozenset("""
    through across around along past against among amid behind beside between
    beyond during under above below within without toward towards upon
    while when where which who whom whose what because although though unless
    until whether and but nor yet
    are was were been being have has had does did doing will would can could
    should may might must
    his her its their our your my this that these those some any each every
    another other both few more most much such
    very just also then than only even still back again
""".split())


def _kmeans_assign(X: np.ndarray, k: int, seed: int = 0, iters: int = 20) -> np.ndarray:
    """Plain k-means over unit vectors: fixed seed, bounded iterations, empty
    clusters keep their center — deterministic by construction, so the same
    query always yields the same groups."""
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    assign = np.full(len(X), -1)
    for _ in range(iters):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
        for c in range(k):
            members = X[assign == c]
            if len(members):
                centers[c] = members.mean(axis=0)
    return assign


def _stem(value: str) -> str:
    """The first three letters of a value, used only to keep one label from
    saying the same word twice ("runs · running", "child · children")."""
    return "".join(ch for ch in value.lower() if ch.isalnum())[:3]


def _pool_terms(conn, pool_ids: list[int]) -> dict[str, tuple[int, str, str, set[int]]]:
    """Every candidate trait over the whole pool being grouped, measured once:
    `name -> (source rank, facet, display value, the samples carrying it)`.

    A trait is an attribute label, a VLM tag, or a caption content word. Every
    attribute label counts, not just its group's dominant one — which label is
    over-represented is precisely what distinguishes one cluster from another.
    The facet is the attribute group (so a label never reads "field/park ·
    street") or the word stem for the rest."""
    qmarks = ",".join("?" * len(pool_ids))
    terms: dict[str, tuple[int, str, str, set[int]]] = {}

    def add(name: str, src: int, facet: str, value: str, sid: int) -> None:
        terms.setdefault(name, (src, facet, value, set()))[3].add(sid)

    for r in conn.execute(
            f"SELECT sample_id, grp, label FROM attributes WHERE sample_id IN ({qmarks})",
            pool_ids):
        add(f"{r['grp']}:{r['label']}", 0, r["grp"], r["label"], r["sample_id"])
    for r in conn.execute(
            f"SELECT sample_id, tag FROM vlm_tags WHERE sample_id IN ({qmarks})", pool_ids):
        add(f"vlm:{r['tag']}", 1, _stem(r["tag"]), r["tag"], r["sample_id"])
    for r in conn.execute(
            f"SELECT sample_id, text FROM captions WHERE sample_id IN ({qmarks})", pool_ids):
        for tok in set(r["text"].split()):
            bare = "".join(ch for ch in tok.lower() if ch.isalnum())
            if (len(bare) >= 3 and bare not in db.STOPWORDS
                    and bare not in SCENARIO_LABEL_SKIP):
                add(f"caption:{bare}", 2, _stem(bare), bare, r["sample_id"])
    return terms


def _distinctive_candidates(terms: dict[str, tuple[int, str, str, set[int]]],
                            member_ids: list[int],
                            pool_size: int) -> list[tuple]:
    """The group's traits ranked by over-representation, most distinctive first.

    A trait qualifies when it describes at least `SCENARIO_MIN_SHARE` of the
    group (a handful of members never titles it) *and* is at least
    `SCENARIO_MIN_LIFT` times as common inside the group as across the whole
    ranked pool — a trait no more common here than everywhere else
    distinguishes nothing at all. Ranking is by lift (p_in / p_bg), so the
    traits the whole result set shares — "dog" under a dog query — score ~1
    and lose to whatever is peculiar to this group.
    Ties break on (in-group count, source, name): fully deterministic.

    Each entry is (lift, n_in, p_in, n_bg, p_bg, source, name, facet, value).
    """
    count = len(member_ids)
    members = set(member_ids)
    cands: list[tuple] = []
    for name, (src, facet, value, sids) in terms.items():
        n_in = len(members & sids)
        p_in = n_in / count
        if p_in < SCENARIO_MIN_SHARE:
            continue
        n_bg = len(sids)
        p_bg = n_bg / pool_size
        lift = p_in / p_bg
        if lift < SCENARIO_MIN_LIFT:
            continue
        cands.append((lift, n_in, p_in, n_bg, p_bg, src, name, facet, value))
    # Rank on lift discounted by distance from structured data, so a facet leads
    # unless a caption word is decisively more distinctive than it.
    cands.sort(key=lambda c: (-c[0] / SCENARIO_SOURCE_MARGIN ** c[5], -c[1], c[5], c[6]))
    return cands


def _evidence_for(cand: tuple, count: int) -> str:
    """One checkable claim: both counts and both shares, e.g.
    "24/61 caption:people — 39% here vs 12% across the results"."""
    _lift, n_in, p_in, _n_bg, p_bg, _src, name, _facet, _value = cand
    return f"{n_in}/{count} {name} — {p_in:.0%} here vs {p_bg:.0%} across the results"


def _label_parts(cands: list[tuple], count: int,
                 taken_leads: set[str]) -> tuple[list[str], str]:
    """(label parts, evidence) for one group, given the leads other groups
    already claimed.

    The leading part must be a trait no other group is titled with, so two
    groups can never read the same: the list is descended until an unclaimed
    lead is found. A group whose every distinctive trait already heads another
    group — or which has none at all — is called "mixed", with the measurement
    that says why. Later parts only avoid repeating this label's own facets."""
    parts: list[str] = []
    evidence: list[str] = []
    used: set[str] = set()
    for cand in cands:
        facet, value = cand[7], cand[8]
        keys = {facet, _stem(value)}
        if keys & used:
            continue
        if not parts and _stem(value) in taken_leads:
            continue          # another group is already titled with this trait
        used |= keys
        parts.append(value)
        evidence.append(_evidence_for(cand, count))
        if len(parts) == SCENARIO_MAX_PARTS:
            break
    if parts:
        return parts, "; ".join(evidence)
    if cands:
        return [], ("every trait over-represented here already titles another group "
                    f"(strongest: {_evidence_for(cands[0], count)})")
    return [], (f"no trait describes {SCENARIO_MIN_SHARE:.0%} of these {count} images "
                f"and is {SCENARIO_MIN_LIFT}x as common here as across the results")


@router.post("/search/scenarios", response_model=ScenarioResponse)
def search_scenarios(body: ComposedSearchRequest,
                     conn: sqlite3.Connection = Depends(get_conn)):
    """The composed ranking, grouped into at most three scenario clusters.

    Ranks to min(200, corpus) — `top_k`/`offset` are ignored here, grouping is
    over the head of the ranking — then k-means with k = min(3, n // 8): a
    grouping is only offered when groups would average eight members, and
    fewer than eight results yields no groups at all. The fixed seed makes the
    whole pipeline deterministic: same query, same groups, byte for byte.

    Labels answer "how is this group different from the rest of the results",
    not "what is in it": a trait is scored by how over-represented it is in the
    group against the same ranked pool (see `_distinctive_candidates`), so the
    query's own subject cannot title all three groups. The most distinctive
    group names itself first and reserves its leading trait, which is what
    keeps the labels apart.
    """
    ranked, _qvec, down = _composed_ranking(conn, body)
    if ranked is None:
        return ScenarioResponse(groups=[], basis=SCENARIO_BASIS,
                                degraded=True, message=down)
    index = get_index()
    top = ranked[:min(SCENARIO_DEPTH, len(ranked))]
    n = len(top)
    if n < SCENARIO_MIN_RESULTS:
        return ScenarioResponse(
            groups=[], basis=SCENARIO_BASIS,
            message=f"Only {n} results — fewer than {SCENARIO_MIN_RESULTS} "
                    "needed to group.")
    k = min(3, n // SCENARIO_MIN_RESULTS)
    X = np.stack([index.vector_of(sid) for sid, _ in top])
    assign = _kmeans_assign(X, k)
    pool_ids = [sid for sid, _ in top]
    terms = _pool_terms(conn, pool_ids)          # the background, measured once
    clusters = []
    for c in range(k):
        members = [top[i] for i in range(n) if assign[i] == c]  # score order
        if not members:
            continue
        ids = [sid for sid, _ in members]
        clusters.append((ids, _distinctive_candidates(terms, ids, len(pool_ids))))
    # Strongest lift first, so the group with the clearest identity claims its
    # trait before a weaker group can; size then first id keep it deterministic.
    order = sorted(range(len(clusters)),
                   key=lambda i: (-(clusters[i][1][0][0] if clusters[i][1] else 0.0),
                                  -len(clusters[i][0]), clusters[i][0][0]))
    labelled: dict[int, tuple[list[str], str]] = {}
    taken_leads: set[str] = set()
    for i in order:
        ids, cands = clusters[i]
        parts, evidence = _label_parts(cands, len(ids), taken_leads)
        if parts:
            taken_leads.add(_stem(parts[0]))
        labelled[i] = (parts, evidence)
    groups = []
    for i, (ids, _cands) in enumerate(clusters):
        parts, evidence = labelled[i]
        label = (" · ".join(parts) if parts else "mixed") + f" — {len(ids)} images"
        # The FULL membership, in ranking order: "save this group as an album"
        # files every member, so a preview here would silently truncate it.
        groups.append(ScenarioGroup(label=label, evidence=evidence,
                                    count=len(ids), sample_ids=ids))
    groups.sort(key=lambda g: (-g.count, g.sample_ids[0] if g.sample_ids else 0))
    return ScenarioResponse(groups=groups[:3], basis=SCENARIO_BASIS)
