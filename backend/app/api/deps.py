"""Shared FastAPI dependencies and row helpers."""
import json
import sqlite3
from typing import Iterator, Optional

from fastapi import Query

from .. import db
from ..db import AXES
from ..schemas import AxisScores, SampleCard

# Sort keys the API accepts, as "<axis>_<direction>". Whitelisted rather than
# parsed, because the axis name reaches SQL as an identifier.
SORT_KEYS = tuple(f"{a}_{d}" for a in AXES for d in ("asc", "desc"))


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
        axes=axis_scores(row),
    )


def axis_bounds(
    legibility_min: Optional[int] = Query(None, ge=0, le=10),
    legibility_max: Optional[int] = Query(None, ge=0, le=10),
    rarity_min: Optional[int] = Query(None, ge=0, le=10),
    rarity_max: Optional[int] = Query(None, ge=0, le=10),
    difficulty_min: Optional[int] = Query(None, ge=0, le=10),
    difficulty_max: Optional[int] = Query(None, ge=0, le=10),
    clutter_min: Optional[int] = Query(None, ge=0, le=10),
    clutter_max: Optional[int] = Query(None, ge=0, le=10),
) -> dict[str, tuple[Optional[int], Optional[int]]]:
    """Difficulty-axis range constraints, as one injectable dependency.

    Stated explicitly rather than as a free-form expression so the bounds are
    validated by FastAPI and visible in the OpenAPI schema.
    """
    raw = {
        "legibility": (legibility_min, legibility_max),
        "rarity": (rarity_min, rarity_max),
        "difficulty": (difficulty_min, difficulty_max),
        "clutter": (clutter_min, clutter_max),
    }
    return {a: b for a, b in raw.items() if b[0] is not None or b[1] is not None}


def build_filters(split: Optional[str], tag: Optional[str], vlm_tag: Optional[str],
                  attr: Optional[str] = None,
                  axes: Optional[dict[str, tuple[Optional[int], Optional[int]]]] = None):
    """Compose WHERE clauses + params for common sample filters.

    `attr` is a zero-shot attribute facet encoded as "group:label"; `axes` maps
    a difficulty axis to (min, max) and emits real range predicates, so an axis
    constraint narrows the candidate set *before* any ranking or LIMIT — the
    same contract the set-membership filters have always honoured.
    """
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
    for axis, (lo, hi) in (axes or {}).items():
        if axis not in AXES:        # never interpolate an unvalidated identifier
            continue
        if lo is not None:
            clauses.append(f"s.{axis} >= ?")
            params.append(lo)
        if hi is not None:
            clauses.append(f"s.{axis} <= ?")
            params.append(hi)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def filtered_id_set(conn, split, tag, vlm_tag, attr=None, axes=None) -> Optional[set[int]]:
    """Set of sample ids matching filters, or None when unfiltered."""
    if not (split or tag or vlm_tag or attr or axes):
        return None
    where, params = build_filters(split, tag, vlm_tag, attr, axes)
    rows = conn.execute(f"SELECT s.id FROM samples s{where}", params)
    return {r["id"] for r in rows}


def order_by_axis(sort: Optional[str]) -> str:
    """SQL ORDER BY for an axis sort key, or "" when unsorted.

    Unscored samples sort last in both directions: a NULL axis means "not
    measured", which is not the same as "measured as zero" and should not be
    allowed to head a ranking.
    """
    if not sort or sort not in SORT_KEYS:
        return ""
    axis, direction = sort.rsplit("_", 1)
    return (f" ORDER BY (s.{axis} IS NULL), s.{axis} "
            f"{'ASC' if direction == 'asc' else 'DESC'}, s.id")


def axis_scores(row: sqlite3.Row) -> Optional[AxisScores]:
    """Axis scores off a samples row, with the raw components that justify them."""
    keys = row.keys()
    if not any(a in keys for a in AXES):
        return None
    values = {a: row[a] for a in AXES if a in keys}
    if all(v is None for v in values.values()):
        return None
    detail = {}
    if "axis_detail" in keys and row["axis_detail"]:
        try:
            detail = json.loads(row["axis_detail"])
        except (ValueError, TypeError):
            detail = {}
    return AxisScores(**values, detail=detail)


def first_captions(conn, sample_ids: list[int]) -> dict[int, str]:
    if not sample_ids:
        return {}
    qmarks = ",".join("?" * len(sample_ids))
    rows = conn.execute(
        f"SELECT sample_id, MIN(idx) AS mi, text FROM captions "
        f"WHERE sample_id IN ({qmarks}) GROUP BY sample_id", sample_ids)
    return {r["sample_id"]: r["text"] for r in rows}
