"""Search: semantic (SigLIP text->image), keyword (FTS5 BM25 over captions +
VLM tags), hybrid (reciprocal-rank fusion of both), or boosted (semantic
ranking replaced by the trained PRISM speaker models, when artifacts exist).

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
import logging
import re
import sqlite3
from typing import Optional, Union

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from PIL import Image as PILImage

from .. import config, db
from ..ml import hubness
from ..ml.embedder import get_embedder
from ..ml.index import get_caption_index, get_index
from ..ml.prism import get_prism_index
from ..schemas import MatchPath, SampleCard, SearchRequest, SearchResponse, TermStat
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


def _boosted_ranking(conn, q: str, allowed: Optional[set[int]], top_k: int):
    """Ranked (sample_id, score) under the trained PRISM speaker models, or
    (None, None, None) when the artifacts (or the embedding stack) are absent.

    Deliberately NO hubness penalty on this path. Stacking the Bayes prior on
    top of the trained mu measured *worse* than the trained mu alone — R@1
    51.6% -> 50.3%, paired delta -1.3 pts, CI95 [-2.0, -0.5] — because training
    already absorbs the hub structure the penalty models. One correction ranks;
    they do not stack. (`data/cache/prism_eval.json`, rows A1 vs A3.)

    The score is a log-likelihood (constant dropped), not a cosine: comparable
    within one result list, meaningless against any other basis — the response
    says `prism_ll` so the UI never prints it as a similarity.
    """
    index = get_index()
    embedder = get_embedder() if index is not None else None
    if index is None or embedder is None:
        return None, None, None
    prism = get_prism_index(index)   # id-validated against the live corpus
    if prism is None:
        return None, None, None
    qvec = embedder.encode_texts([normalize_query_text(q)])[0]
    ranked = prism.search(qvec, top_k=top_k, allowed_ids=allowed)
    return ranked, qvec, "prism_ll"


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
    boosted, boosted_basis = None, None
    if mode == "boosted":
        boosted, qvec, boosted_basis = _boosted_ranking(conn, q, allowed, depth)
        if boosted is None:
            degraded, mode = True, "semantic"
            # Only name the missing artifacts when they are the cause; if the
            # whole embedding stack is down, the semantic fallback below says so.
            if get_index() is not None and get_embedder() is not None:
                message = ("Boosted ranking unavailable (no trained PRISM model "
                           "for this corpus — run `python -m app.train_prism "
                           "--no-sigma`) — using semantic search.")
    if mode in ("semantic", "hybrid"):
        semantic, qvec, semantic_basis = _semantic_ranking(conn, q, allowed, depth)
        if semantic is None:
            degraded, mode = True, "keyword"
            # Appended, not assigned: a boosted request that degraded twice
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
    elif mode == "boosted":
        ranked = [sid for sid, _ in boosted]
        scores = dict(boosted)
        score_basis = boosted_basis
        record("boosted", ranked)
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
    if mode in ("semantic", "hybrid", "boosted"):
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
    mode: str = Query("hybrid", pattern="^(semantic|keyword|hybrid|boosted)$"),
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
