"""Browse, inspect, similarity, and subset export endpoints."""
import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..ml.index import get_index
from ..schemas import CaptionOut, SampleCard, SampleDetail, SampleList
from .deps import build_filters, first_captions, get_conn, image_url, row_to_card, thumb_url

router = APIRouter()


@router.get("/samples", response_model=SampleList)
def list_samples(
    page: int = Query(1, ge=1),
    per_page: int = Query(60, ge=1, le=200),
    split: Optional[str] = None,
    tag: Optional[str] = None,
    vlm_tag: Optional[str] = None,
    attr: Optional[str] = None,
    conn: sqlite3.Connection = Depends(get_conn),
):
    where, params = build_filters(split, tag, vlm_tag, attr)
    total = conn.execute(f"SELECT COUNT(*) FROM samples s{where}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT s.* FROM samples s{where} ORDER BY s.id LIMIT ? OFFSET ?",
        params + [per_page, (page - 1) * per_page],
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
    split: Optional[str] = None,
    tag: Optional[str] = None,
    vlm_tag: Optional[str] = None,
    attr: Optional[str] = None,
    conn: sqlite3.Connection = Depends(get_conn),
):
    """JSON manifest of the current filtered subset (ids, filenames, captions,
    user tags), e.g. for handing a curated slice to a training pipeline."""
    where, params = build_filters(split, tag, vlm_tag, attr)
    rows = conn.execute(f"SELECT s.* FROM samples s{where} ORDER BY s.id", params).fetchall()
    ids = [r["id"] for r in rows]
    caps: dict[int, list[str]] = {}
    tag_map: dict[int, list[str]] = {}
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
    return {
        "count": len(rows),
        "filters": {"split": split, "tag": tag, "vlm_tag": vlm_tag, "attr": attr},
        "samples": [
            {"id": r["id"], "filename": r["filename"], "split": r["split"],
             "captions": caps.get(r["id"], []), "tags": tag_map.get(r["id"], [])}
            for r in rows
        ],
    }
