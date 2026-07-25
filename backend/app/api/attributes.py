"""Zero-shot attribute coverage (dataset composition / long-tail surfacing)."""
import sqlite3

from fastapi import APIRouter, Depends

from ..schemas import AttributeGroup, AttributeLabel
from .deps import get_conn

router = APIRouter()


@router.get("/attributes/coverage", response_model=list[AttributeGroup])
def coverage(conn: sqlite3.Connection = Depends(get_conn)):
    """Per-group label counts, plus how much of the corpus the classifier
    declined to label at all.

    The abstention count is the point of this endpoint as much as the
    histogram: a group where 10% of images were too ambiguous to call is a
    different research finding from one where every image got a confident
    label, and a bar chart that silently omits the abstentions reads as the
    second when it is the first.
    """
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] or 1
    rows = conn.execute(
        "SELECT grp, label, COUNT(*) AS n FROM attributes "
        "GROUP BY grp, label ORDER BY grp, n DESC").fetchall()
    stats = {r["grp"]: r for r in conn.execute(
        "SELECT grp, COUNT(*) AS n, AVG(confidence) AS mc FROM attributes GROUP BY grp")}
    groups: dict[str, list[AttributeLabel]] = {}
    for r in rows:
        groups.setdefault(r["grp"], []).append(AttributeLabel(
            label=r["label"], count=r["n"], fraction=round(r["n"] / total, 4)))
    out = []
    for g, labels in groups.items():
        s = stats.get(g)
        labelled = s["n"] if s else 0
        out.append(AttributeGroup(
            grp=g, labels=labels, labelled=labelled,
            abstained=max(0, total - labelled),
            mean_confidence=round(s["mc"], 4) if s and s["mc"] is not None else None))
    return out
