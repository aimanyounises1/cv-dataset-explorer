"""Browse, inspect, similarity, and subset export endpoints."""
import csv
import io
import json
import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from .. import config
from ..db import AXES
from ..ml import providers
from ..ml.index import get_index
from ..schemas import CaptionOut, SampleCard, SampleDetail, SampleList
from .deps import (
    MAX_SQLITE_INT,
    axis_bounds,
    axis_scores,
    build_filters,
    embeddings_fingerprint,
    first_captions,
    get_conn,
    id_list,
    image_url,
    order_by_album_position,
    order_by_axis,
    row_to_card,
    stage_id_list,
    thumb_url,
)

router = APIRouter()


@router.get("/samples", response_model=SampleList)
def list_samples(
    # Bounded above as well: page * per_page becomes a SQL OFFSET, and 2^63
    # used to arrive there as a bare 500. The ceiling covers any real corpus
    # by orders of magnitude.
    page: int = Query(1, ge=1, le=1_000_000),
    per_page: int = Query(60, ge=1, le=200),
    split: Optional[str] = None,
    tag: Optional[str] = None,
    vlm_tag: Optional[str] = None,
    attr: Optional[list[str]] = Query(
        None, description="Attribute facet 'group:label'. Repeatable: several "
                          "are intersected, so attr=time_of_day:night&"
                          "attr=setting:indoor is the images that are both."),
    sort: Optional[str] = Query(None, description="<axis>_asc | <axis>_desc"),
    max_agreement: Optional[float] = Query(
        None, ge=0.0, le=1.0, allow_inf_nan=False,
        description="Samples with any caption at or below this agreement"),
    cluster: Optional[int] = Query(None, description="k-means group id"),
    album: Optional[int] = Query(None, ge=1, le=MAX_SQLITE_INT,
                                 description="Restrict to members of this album"),
    axes: dict = Depends(axis_bounds),
    ids: list = Depends(id_list),
    conn: sqlite3.Connection = Depends(get_conn),
):
    staged = stage_id_list(conn, ids) if ids else False
    where, params = build_filters(split, tag, vlm_tag, attr, axes, ids, staged,
                                  max_agreement, cluster, album)
    total = conn.execute(f"SELECT COUNT(*) FROM samples s{where}", params).fetchone()[0]
    # Sorting is over the whole filtered set, in SQL — never over the page.
    # An explicit axis sort wins; otherwise an album filter presents its
    # members in the album's own order, because that order is user intent.
    order, order_params = order_by_axis(sort), []
    if not order:
        order, order_params = order_by_album_position(album)
    order = order or " ORDER BY s.id"
    rows = conn.execute(
        f"SELECT s.* FROM samples s{where}{order} LIMIT ? OFFSET ?",
        params + order_params + [per_page, (page - 1) * per_page],
    ).fetchall()
    captions = first_captions(conn, [r["id"] for r in rows])
    items = [row_to_card(r, caption=captions.get(r["id"])) for r in rows]
    return SampleList(items=items, total=total, page=page, per_page=per_page)


@router.get("/samples/{sample_id}", response_model=SampleDetail)
def get_sample(sample_id: int, conn: sqlite3.Connection = Depends(get_conn)):
    row = conn.execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Sample not found")
    captions = [CaptionOut(text=r["text"],
                           agreement=round(r["agreement"], 4) if r["agreement"] is not None else None)
                for r in conn.execute(
        "SELECT text, agreement FROM captions WHERE sample_id = ? ORDER BY idx", (sample_id,))]
    tags = [r["name"] for r in conn.execute(
        "SELECT t.name FROM tags t JOIN sample_tags st ON st.tag_id = t.id "
        "WHERE st.sample_id = ? ORDER BY t.name", (sample_id,))]
    vlm_tags = [r["tag"] for r in conn.execute(
        "SELECT tag FROM vlm_tags WHERE sample_id = ? ORDER BY tag", (sample_id,))]
    attributes = {r["grp"]: r["label"] for r in conn.execute(
        "SELECT grp, label FROM attributes WHERE sample_id = ?", (sample_id,))}
    consistency = row["caption_consistency"] if "caption_consistency" in row.keys() else None
    return SampleDetail(
        id=row["id"], filename=row["filename"], split=row["split"],
        width=row["width"], height=row["height"], filesize=row["filesize"],
        image_url=image_url(row["filename"]), thumb_url=thumb_url(row["filename"]),
        captions=captions, tags=tags, vlm_tags=vlm_tags, attributes=attributes,
        cluster=row["cluster"],
        caption_consistency=round(consistency, 4) if consistency is not None else None,
        axes=axis_scores(row),
    )


@router.get("/samples/{sample_id}/similar", response_model=list[SampleCard])
def similar_samples(
    sample_id: int,
    top_k: int = Query(12, ge=1, le=60),
    conn: sqlite3.Connection = Depends(get_conn),
):
    index = get_index()
    if index is None:
        raise HTTPException(503, "Embeddings not computed yet — run `python -m app.ingest`.")
    results = index.similar_to(sample_id, top_k=top_k)
    if not results:
        return []
    ids = [sid for sid, _ in results]
    qmarks = ",".join("?" * len(ids))
    rows = {r["id"]: r for r in conn.execute(f"SELECT * FROM samples WHERE id IN ({qmarks})", ids)}
    captions = first_captions(conn, ids)
    return [row_to_card(rows[sid], caption=captions.get(sid), score=score)
            for sid, score in results if sid in rows]


@router.get("/export")
def export_subset(
    q: Optional[str] = None,
    # Kept in step with /api/search deliberately: export exists to hand over
    # *the set on screen*, so a mode the gallery can produce and the export
    # cannot is a 422 on a button the user can see.
    mode: str = Query("hybrid", pattern="^(semantic|keyword|hybrid|boosted)$"),
    top_k: int = Query(500, ge=1, le=5000),
    split: Optional[str] = None,
    tag: Optional[str] = None,
    vlm_tag: Optional[str] = None,
    attr: Optional[list[str]] = Query(
        None, description="Attribute facet 'group:label'. Repeatable: several "
                          "are intersected, so attr=time_of_day:night&"
                          "attr=setting:indoor is the images that are both."),
    fmt: str = Query("json", alias="format", pattern="^(json|jsonl|csv)$"),
    max_agreement: Optional[float] = Query(
        None, ge=0.0, le=1.0, allow_inf_nan=False,
        description="Samples with any caption at or below this agreement"),
    cluster: Optional[int] = Query(None, description="k-means group id"),
    album: Optional[int] = Query(None, ge=1, le=MAX_SQLITE_INT,
                                 description="Restrict to members of this album"),
    scores: bool = Query(
        True, description="Include the computed per-sample signals "
                          "(difficulty axes, cluster, caption agreement)"),
    axes: dict = Depends(axis_bounds),
    ids: list = Depends(id_list),
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Manifest of the current subset — filters *or* a search result set — for
    handing a curated slice to a training pipeline.

    With `q`, this delegates to the same `run_search` the UI and the agent tools
    call, so an export is exactly the result set the user was looking at, in the
    same order. Without `q`, it is the filtered corpus by id.

    The manifest records the query that produced it, including the embedding
    model, because a slice you cannot regenerate is not a curated dataset.
    """
    # A filter export is complete; a search export stops where the fusion
    # stopped ranking. The manifest says which one this file is, because a
    # consumer cannot tell a full set from a truncated ranking by looking.
    ranked_depth = None
    truncated = None
    if q:
        from .search import run_search

        result = run_search(conn, q, mode=mode, top_k=top_k, split=split,
                            tag=tag, vlm_tag=vlm_tag, attr=attr, axes=axes, ids=ids,
                            max_agreement=max_agreement, album=album)
        ranked_depth = result.depth_limit
        truncated = bool(result.depth_reached)
        ids = [it.id for it in result.items]
        rows_by_id = {}
        if ids:
            qmarks = ",".join("?" * len(ids))
            rows_by_id = {r["id"]: r for r in conn.execute(
                f"SELECT * FROM samples WHERE id IN ({qmarks})", ids)}
        rows = [rows_by_id[i] for i in ids if i in rows_by_id]  # ranked order
        ids = [r["id"] for r in rows]
    else:
        staged = stage_id_list(conn, ids) if ids else False
        where, params = build_filters(split, tag, vlm_tag, attr, axes, ids, staged,
                                      max_agreement, cluster, album)
        # A filter export of an album keeps the album's own order — the export
        # exists to hand over the set as curated, and the order is part of that.
        order, order_params = order_by_album_position(album)
        rows = conn.execute(
            f"SELECT s.* FROM samples s{where}{order or ' ORDER BY s.id'}",
            params + order_params).fetchall()
        ids = [r["id"] for r in rows]
    caps: dict[int, list[str]] = {}
    tag_map: dict[int, list[str]] = {}
    attr_map: dict[int, dict[str, str]] = {}
    agree_map: dict[int, float] = {}
    if ids:
        qmarks = ",".join("?" * len(ids))
        for c in conn.execute(
            f"SELECT sample_id, text FROM captions WHERE sample_id IN ({qmarks}) "
            "ORDER BY sample_id, idx", ids):
            caps.setdefault(c["sample_id"], []).append(c["text"])
        for t in conn.execute(
            f"SELECT st.sample_id, t.name FROM sample_tags st "
            f"JOIN tags t ON t.id = st.tag_id WHERE st.sample_id IN ({qmarks})", ids):
            tag_map.setdefault(t["sample_id"], []).append(t["name"])
        if scores:
            for a in conn.execute(
                f"SELECT sample_id, grp, label FROM attributes "
                f"WHERE sample_id IN ({qmarks})", ids):
                attr_map.setdefault(a["sample_id"], {})[a["grp"]] = a["label"]
            for r in conn.execute(
                f"SELECT sample_id, AVG(agreement) AS m FROM captions "
                f"WHERE sample_id IN ({qmarks}) AND agreement IS NOT NULL "
                f"GROUP BY sample_id", ids):
                agree_map[r["sample_id"]] = r["m"]

    def record(r) -> dict:
        """One exported row.

        With `scores`, it carries the signals this tool computed rather than only
        the ones it was given. Exporting "the 866 hardest images" in a file that
        cannot say how hard any one of them is forces every downstream consumer
        to re-derive work already done — and curriculum ordering, loss weighting
        and stratified splits all need exactly these numbers.
        """
        out = {"id": r["id"], "filename": r["filename"], "split": r["split"],
               "captions": caps.get(r["id"], []), "tags": tag_map.get(r["id"], [])}
        if not scores:
            return out
        keys = r.keys()
        out["axes"] = {a: r[a] for a in AXES if a in keys and r[a] is not None}
        if "cluster" in keys and r["cluster"] is not None:
            out["cluster"] = r["cluster"]
        if "caption_consistency" in keys and r["caption_consistency"] is not None:
            out["caption_consistency"] = round(r["caption_consistency"], 4)
        if r["id"] in agree_map:
            out["mean_agreement"] = round(agree_map[r["id"]], 4)
        if r["id"] in attr_map:
            out["attributes"] = attr_map[r["id"]]
        return out

    samples = [record(r) for r in rows]
    query = {"q": q, "mode": mode if q else None, "top_k": top_k if q else None,
             "split": split, "tag": tag, "vlm_tag": vlm_tag, "attr": attr,
             "cluster": cluster, "album": album,
             "axes": {a: list(b) for a, b in axes.items()},
             "max_agreement": max_agreement, "scores": scores,
             # Named so a consumer can tell whether two exports are comparable:
             # the axes are percentile ranks over whatever corpus was loaded,
             # and the fingerprint pins which embeddings produced the scores.
             "embed_model": providers.active_model_id(),
             "corpus_samples": conn.execute(
                 "SELECT COUNT(*) FROM samples").fetchone()[0],
             "embeddings_fingerprint": embeddings_fingerprint(),
             # Search exports only: how deep the ranking went, and whether more
             # images matched than the depth could hold. Null for filter
             # exports, which are complete by construction.
             "ranked_depth": ranked_depth,
             "truncated": truncated,
             "axis_semantics": "0-10 percentile ranks over this corpus; "
                               "not comparable across datasets"}

    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        # CSV is positional, so score columns are appended after the original
        # five rather than interleaved: an existing consumer reading by index
        # keeps working, and `scores=false` reproduces the old header exactly.
        head = ["id", "filename", "split", "captions", "tags"]
        if scores:
            head += list(AXES) + ["cluster", "caption_consistency", "mean_agreement"]
        w.writerow(head)
        for s in samples:
            row = [s["id"], s["filename"], s["split"],
                   " | ".join(s["captions"]), " ".join(s["tags"])]
            if scores:
                row += [s.get("axes", {}).get(a) for a in AXES]
                row += [s.get("cluster"), s.get("caption_consistency"),
                        s.get("mean_agreement")]
            w.writerow(row)
        return _download(buf.getvalue(), "text/csv", "csv")
    if fmt == "jsonl":
        # Query provenance rides as the first record so the file stays valid JSONL.
        lines = [json.dumps({"_manifest": query, "count": len(samples)})]
        lines += [json.dumps(s) for s in samples]
        return _download("\n".join(lines) + "\n", "application/x-ndjson", "jsonl")
    return {"count": len(samples), "filters": query, "samples": samples}


def _download(body: str, media_type: str, ext: str) -> Response:
    return Response(
        content=body, media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="cvde-export.{ext}"'},
    )
