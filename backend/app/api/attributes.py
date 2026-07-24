"""Zero-shot attribute coverage (dataset composition / long-tail surfacing)."""
import sqlite3

from fastapi import APIRouter, Depends

from ..schemas import AttributeGroup, AttributeLabel
from .deps import get_conn

router = APIRouter()


@router.get("/attributes/coverage", response_model=list[AttributeGroup])
def coverage(conn: sqlite3.Connection = Depends(get_conn)):
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] or 1
    rows = conn.execute(
        "SELECT grp, label, COUNT(*) AS n FROM attributes "
        "GROUP BY grp, label ORDER BY grp, n DESC").fetchall()
    groups: dict[str, list[AttributeLabel]] = {}
    for r in rows:
        groups.setdefault(r["grp"], []).append(AttributeLabel(
            label=r["label"], count=r["n"], fraction=round(r["n"] / total, 4)))
    return [AttributeGroup(grp=g, labels=labels) for g, labels in groups.items()]
