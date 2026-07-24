"""Shared FastAPI dependencies and row helpers."""
import sqlite3
from typing import Iterator, Optional

from .. import db
from ..schemas import SampleCard


def get_conn() -> Iterator[sqlite3.Connection]:
    with db.get_db() as conn:
        yield conn


def thumb_url(filename: str) -> str:
    return f"/media/thumbs/{filename}"


def image_url(filename: str) -> str:
    return f"/media/images/{filename}"


def row_to_card(row: sqlite3.Row, caption: Optional[str] = None,
                score: Optional[float] = None) -> SampleCard:
    return SampleCard(
        id=row["id"], filename=row["filename"], split=row["split"],
        width=row["width"], height=row["height"],
        thumb_url=thumb_url(row["filename"]), caption=caption, score=score,
    )


def build_filters(split: Optional[str], tag: Optional[str], vlm_tag: Optional[str],
                  attr: Optional[str] = None):
    """Compose WHERE clauses + params for common sample filters.
    `attr` is a zero-shot attribute facet encoded as "group:label"."""
    clauses, params = [], []
    if split:
        clauses.append("s.split = ?")
        params.append(split)
    if tag:
        clauses.append(
            "s.id IN (SELECT st.sample_id FROM sample_tags st "
            "JOIN tags t ON t.id = st.tag_id WHERE t.name = ?)")
        params.append(tag)
    if vlm_tag:
        clauses.append("s.id IN (SELECT sample_id FROM vlm_tags WHERE tag = ?)")
        params.append(vlm_tag)
    if attr and ":" in attr:
        grp, label = attr.split(":", 1)
        clauses.append("s.id IN (SELECT sample_id FROM attributes WHERE grp = ? AND label = ?)")
        params.extend([grp, label])
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def filtered_id_set(conn, split, tag, vlm_tag, attr=None) -> Optional[set[int]]:
    """Set of sample ids matching filters, or None when unfiltered."""
    if not (split or tag or vlm_tag or attr):
        return None
    where, params = build_filters(split, tag, vlm_tag, attr)
    rows = conn.execute(f"SELECT s.id FROM samples s{where}", params)
    return {r["id"] for r in rows}


def first_captions(conn, sample_ids: list[int]) -> dict[int, str]:
    if not sample_ids:
        return {}
    qmarks = ",".join("?" * len(sample_ids))
    rows = conn.execute(
        f"SELECT sample_id, MIN(idx) AS mi, text FROM captions "
        f"WHERE sample_id IN ({qmarks}) GROUP BY sample_id", sample_ids)
    return {r["sample_id"]: r["text"] for r in rows}
