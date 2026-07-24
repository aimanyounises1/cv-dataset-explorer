"""Annotation QA endpoints (CLIPScore-style caption auditing).

Low image-caption agreement => likely weak/wrong caption. The sibling mean
contextualizes it: low agreement + high sibling mean = the caption is the
outlier; all-low = the image itself is unusual (a different kind of find).
"""
import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..schemas import QASummary, SuspectCaption
from .deps import build_filters, get_conn, row_to_card

router = APIRouter()


@router.get("/qa/summary", response_model=QASummary)
def qa_summary(conn: sqlite3.Connection = Depends(get_conn)):
    row = conn.execute(
        "SELECT COUNT(*) AS n, AVG(agreement) AS mean FROM captions "
        "WHERE agreement IS NOT NULL").fetchone()
    return QASummary(
        available=row["n"] > 0, scored_captions=row["n"],
        mean_agreement=round(row["mean"], 4) if row["mean"] is not None else None,
    )


@router.get("/qa/captions", response_model=list[SuspectCaption])
def suspect_captions(
    limit: int = Query(50, ge=1, le=200),
    order: str = Query("asc", pattern="^(asc|desc)$"),
    split: Optional[str] = None,
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Captions ranked by image-caption agreement (asc = most suspect first)."""
    where, params = build_filters(split, None, None)
    and_where = where.replace(" WHERE ", " AND ", 1) if where else ""
    rows = conn.execute(
        "SELECT c.id AS cid, c.text, c.agreement, c.sample_id, s.* "
        "FROM captions c JOIN samples s ON s.id = c.sample_id "
        f"WHERE c.agreement IS NOT NULL{and_where} "
        f"ORDER BY c.agreement {'ASC' if order == 'asc' else 'DESC'} LIMIT ?",
        params + [limit],
    ).fetchall()
    if not rows:
        return []

    sample_ids = list({r["sample_id"] for r in rows})
    qmarks = ",".join("?" * len(sample_ids))
    sib = {}
    for r in conn.execute(
        f"SELECT sample_id, id, agreement FROM captions "
        f"WHERE sample_id IN ({qmarks}) AND agreement IS NOT NULL", sample_ids):
        sib.setdefault(r["sample_id"], []).append((r["id"], r["agreement"]))

    out = []
    for r in rows:
        others = [a for cid, a in sib.get(r["sample_id"], []) if cid != r["cid"]]
        out.append(SuspectCaption(
            caption=r["text"],
            agreement=round(r["agreement"], 4),
            sibling_mean=round(sum(others) / len(others), 4) if others else None,
            sample=row_to_card(r, caption=r["text"]),
        ))
    return out


@router.get("/qa/consistency", response_model=list[SuspectCaption])
def inconsistent_samples(
    limit: int = Query(30, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Samples whose 5 captions disagree most with each other (ambiguous images
    or outlier captions). Reuses SuspectCaption with consistency as the score."""
    rows = conn.execute(
        "SELECT * FROM samples WHERE caption_consistency IS NOT NULL "
        "ORDER BY caption_consistency ASC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        first = conn.execute(
            "SELECT text FROM captions WHERE sample_id = ? ORDER BY idx LIMIT 1",
            (r["id"],)).fetchone()
        out.append(SuspectCaption(
            caption=first["text"] if first else "",
            agreement=round(r["caption_consistency"], 4),
            sample=row_to_card(r, caption=first["text"] if first else None),
        ))
    return out
