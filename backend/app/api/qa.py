"""Annotation QA endpoints (CLIPScore-style caption auditing).

Low image-caption agreement => likely weak/wrong caption. The sibling mean
contextualizes it: low agreement + high sibling mean = the caption is the
outlier; all-low = the image itself is unusual (a different kind of find).
"""
import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..schemas import QASelection, QASummary, SuspectCaption
from .deps import build_filters, get_conn, row_to_card

router = APIRouter()


HIST_BINS = 40


@router.get("/qa/summary", response_model=QASummary)
def qa_summary(conn: sqlite3.Connection = Depends(get_conn)):
    row = conn.execute(
        "SELECT COUNT(*) AS n, AVG(agreement) AS mean, MIN(agreement) AS lo, "
        "MAX(agreement) AS hi FROM captions WHERE agreement IS NOT NULL").fetchone()
    if not row["n"]:
        return QASummary(available=False, scored_captions=0)

    lo, hi = row["lo"], row["hi"]
    span = (hi - lo) or 1.0
    # Binned in SQL rather than by streaming 40,000 floats to Python: the width
    # is derived from the observed range so the bars describe this dataset.
    width = span / HIST_BINS
    counts = {r["b"]: r["n"] for r in conn.execute(
        "SELECT CAST((agreement - ?) / ? AS INTEGER) AS b, COUNT(*) AS n "
        "FROM captions WHERE agreement IS NOT NULL GROUP BY b", (lo, width))}
    histogram = [
        {"lo": round(lo + i * width, 4),
         "hi": round(lo + (i + 1) * width, 4),
         # The topmost value lands in bin HIST_BINS by integer truncation; fold
         # it into the last real bin rather than emitting an off-by-one tail.
         "count": counts.get(i, 0) + (counts.get(HIST_BINS, 0) if i == HIST_BINS - 1 else 0)}
        for i in range(HIST_BINS)
    ]
    return QASummary(
        available=True, scored_captions=row["n"],
        mean_agreement=round(row["mean"], 4),
        histogram=histogram,
        min_agreement=round(lo, 4), max_agreement=round(hi, 4),
    )


@router.get("/qa/selection", response_model=QASelection)
def qa_selection(
    max_agreement: float = Query(..., description="The review threshold"),
    split: Optional[str] = None,
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Size of the set a threshold selects, in both units that matter.

    A threshold is drawn over captions but acted on as images, and the two counts
    differ because one image can have several weak captions. Reporting both keeps
    the hand-off button honest: "review 388 images" is what the gallery will show,
    even when 444 captions are below the line.
    """
    where, params = build_filters(split, None, None)
    and_where = where.replace(" WHERE ", " AND ", 1) if where else ""
    row = conn.execute(
        "SELECT COUNT(*) AS captions, COUNT(DISTINCT c.sample_id) AS samples "
        "FROM captions c JOIN samples s ON s.id = c.sample_id "
        f"WHERE c.agreement IS NOT NULL AND c.agreement <= ?{and_where}",
        [max_agreement] + params).fetchone()
    return QASelection(max_agreement=max_agreement,
                       captions=row["captions"], samples=row["samples"])


@router.get("/qa/captions", response_model=list[SuspectCaption])
def suspect_captions(
    limit: int = Query(50, ge=1, le=200),
    order: str = Query("asc", pattern="^(asc|desc)$"),
    split: Optional[str] = None,
    max_agreement: Optional[float] = Query(
        None, description="Only captions at or below this agreement — the review threshold"),
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Captions ranked by image-caption agreement (asc = most suspect first)."""
    where, params = build_filters(split, None, None)
    and_where = where.replace(" WHERE ", " AND ", 1) if where else ""
    threshold = ""
    if max_agreement is not None:
        threshold = " AND c.agreement <= ?"
        params = params + [max_agreement]
    rows = conn.execute(
        "SELECT c.id AS cid, c.text, c.agreement, c.sample_id, s.* "
        "FROM captions c JOIN samples s ON s.id = c.sample_id "
        f"WHERE c.agreement IS NOT NULL{and_where}{threshold} "
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
    split: Optional[str] = None,
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Samples whose 5 captions disagree most with each other (ambiguous images
    or outlier captions). Reuses SuspectCaption with consistency as the score."""
    where, params = build_filters(split, None, None)
    and_where = where.replace(" WHERE ", " AND ", 1) if where else ""
    rows = conn.execute(
        f"SELECT s.* FROM samples s WHERE s.caption_consistency IS NOT NULL{and_where} "
        "ORDER BY s.caption_consistency ASC LIMIT ?", params + [limit]).fetchall()
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
