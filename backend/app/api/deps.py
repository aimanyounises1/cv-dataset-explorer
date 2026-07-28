"""Shared FastAPI dependencies and row helpers."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Annotated, Iterator, Optional, Union

from fastapi import HTTPException, Path, Query
from pydantic import Field

from .. import db
from ..db import AXES
from ..schemas import AxisScores, SampleCard

# Sort keys the API accepts, as "<axis>_<direction>". Whitelisted rather than
# parsed, because the axis name reaches SQL as an identifier.
SORT_KEYS = tuple(f"{a}_{d}" for a in AXES for d in ("asc", "desc"))

# SQLite INTEGER is signed 64-bit; a Python int past 2^63-1 raises
# OverflowError at bind time, which would surface as a 500. Bounding turns the
# hostile probe into a 422.
MAX_SQLITE_INT = 2**63 - 1
PathId = Annotated[int, Path(ge=1, le=MAX_SQLITE_INT)]
BoundedId = Annotated[int, Field(ge=1, le=MAX_SQLITE_INT)]


def record_activity(conn: sqlite3.Connection, kind: str, payload: dict) -> None:
    """Append one activity event. The caller owns the transaction: an event
    rides inside the mutation that caused it, so history can never claim
    something happened that in fact rolled back."""
    conn.execute(
        "INSERT INTO activity_events(kind, payload, created_at) VALUES (?,?,?)",
        (kind, json.dumps(payload, separators=(",", ":")),
         datetime.now(timezone.utc).isoformat()))


def get_conn() -> Iterator[sqlite3.Connection]:
    with db.get_db() as conn:
        yield conn


def embeddings_fingerprint() -> str:
    """Size+mtime of the embedding artifacts, hashed short.

    Deliberately not the database file: SQLite's mtime moves on every tag
    edit, which would cry "environment differs" over saved views whose results
    are identical. The embedding files change exactly when a re-ingest or
    re-embed changes what a query returns — the event worth catching.
    """
    from ..ml import providers

    emb_dir = providers.active_emb_dir()
    # Provider-scoped: the same corpus embedded by two providers is two
    # different retrieval environments, so the provider name (and, for
    # prompt-bearing providers, the prompt version) joins the hash.
    parts = [f"provider:{providers.active_provider()}"]
    manifest = providers.read_manifest(emb_dir)
    if manifest and manifest.get("prompt_version"):
        parts.append(f"prompt:{manifest['prompt_version']}")
    for name in ("image_embeddings.npy", "caption_embeddings.npy"):
        try:
            st = (emb_dir / name).stat()
            parts.append(f"{name}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            parts.append(f"{name}:absent")
    return hashlib.sha1(";".join(parts).encode()).hexdigest()[:12]


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


# A pasted list is how a researcher brings back a set computed somewhere else —
# worst-loss samples from a training run, a regression list, another team's
# output. The cap is generous because the list is machine-produced.
MAX_ID_LIST = 60_000

# SQLite binds each `IN (?, ?, ...)` entry as a host parameter, and the default
# SQLITE_LIMIT_VARIABLE_NUMBER is 32766 — so a list anywhere near the cap above
# would fail with "too many SQL variables" rather than working. Past this
# threshold the entries go into a temp table instead, which costs no host
# parameters at all; the threshold leaves headroom for the other filters'
# parameters in the same statement.
ID_PARAM_LIMIT = 10_000
ID_STAGE_TABLE = "_id_filter"


def parse_id_list(text: Optional[str]) -> list[str]:
    """Split a pasted list into entries.

    Accepts commas or newlines (real pastes contain both), tolerates blank
    entries and stray whitespace, and preserves order while removing repeats so
    a duplicated line cannot weight a sample twice.
    """
    if not text:
        return []
    seen, out = set(), []
    for raw in text.replace(",", "\n").splitlines():
        entry = raw.strip()
        if entry and entry not in seen:
            seen.add(entry)
            out.append(entry)
    return out


def stage_id_list(conn, entries: list[str]) -> bool:
    """Put a large id list in a temp table; return True if it was staged.

    Temp tables are per-connection and `get_conn` hands out a fresh connection
    per request, so this lives exactly as long as the request that made it and
    cannot leak into another one.
    """
    if len(entries) <= ID_PARAM_LIMIT:
        return False
    conn.execute(f"DROP TABLE IF EXISTS temp.{ID_STAGE_TABLE}")
    conn.execute(f"CREATE TEMP TABLE {ID_STAGE_TABLE} (entry TEXT PRIMARY KEY)")
    conn.executemany(
        f"INSERT OR IGNORE INTO {ID_STAGE_TABLE}(entry) VALUES (?)",
        [(e,) for e in entries])
    return True


def id_list_clause(entries: list[str], staged: bool = False) -> tuple[str, list]:
    """SQL predicate matching either a sample id or a filename.

    Both are accepted in one list because both are things a researcher already
    has: ids come out of this tool's own export, filenames out of anything that
    touched the images on disk. Entries that match nothing are ignored rather
    than rejected — a list pasted from a larger corpus is a normal thing to do,
    and failing the whole query over one stale line would be hostile.
    """
    if not entries:
        return "", []
    if staged:
        # Zero host parameters: the entries are already rows. Matching on the
        # text column for both cases keeps one table doing both jobs, and the
        # id side is compared as text so no CAST of untrusted input is needed.
        return (f"(CAST(s.id AS TEXT) IN (SELECT entry FROM {ID_STAGE_TABLE})"
                f" OR s.filename IN (SELECT entry FROM {ID_STAGE_TABLE}))"), []
    # `isdigit()` is true for non-ASCII digits ('٣') that int() also accepts, so
    # the split below is deliberately str.isascii()-guarded: a non-ASCII numeral
    # is treated as a filename, where it simply matches nothing.
    numeric = [int(e) for e in entries if e.lstrip("-").isascii() and e.lstrip("-").isdigit()]
    names = [e for e in entries
             if not (e.lstrip("-").isascii() and e.lstrip("-").isdigit())]
    parts, params = [], []
    if numeric:
        parts.append(f"s.id IN ({','.join('?' * len(numeric))})")
        params.extend(numeric)
    if names:
        parts.append(f"s.filename IN ({','.join('?' * len(names))})")
        params.extend(names)
    if not parts:
        return "", []
    return "(" + " OR ".join(parts) + ")", params


def id_list(ids: Optional[str] = Query(
        None, description=f"Sample ids or filenames, comma/newline separated "
                          f"(max {MAX_ID_LIST}). Large lists need POST /api/search: "
                          f"a URL cannot carry tens of thousands of entries.")
) -> list[str]:
    entries = parse_id_list(ids)
    if len(entries) > MAX_ID_LIST:
        raise HTTPException(
            400, f"Too many entries: {len(entries)}. The limit is {MAX_ID_LIST}.")
    return entries


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
                  attr: Optional[Union[str, list[str]]] = None,
                  axes: Optional[dict[str, tuple[Optional[int], Optional[int]]]] = None,
                  ids: Optional[list[str]] = None, ids_staged: bool = False,
                  max_agreement: Optional[float] = None,
                  cluster: Optional[int] = None,
                  album: Optional[int] = None):
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
    if cluster is not None:
        # k-means group, computed in the full 768-d space and merely drawn on the
        # map. It was colourable there long before it was filterable anywhere,
        # which meant the only way to reach a cluster was to lasso it by hand.
        clauses.append("s.cluster = ?")
        params.append(cluster)
    if album is not None:
        # Album membership. An id that names no album matches nothing — an
        # honest empty result, the same contract a tag that tags nothing has.
        clauses.append("s.id IN (SELECT sample_id FROM album_items WHERE album_id = ?)")
        params.append(album)
    # One attribute facet or several, intersected. `attr` was a bare `str`, so
    # FastAPI bound only the LAST occurrence: `?attr=time_of_day:night&
    # attr=environment:indoors` silently returned the 706 indoor images instead
    # of the 99 that are both, and reversing the order returned 392. No error,
    # just a different answer than the URL asked for — which also made the
    # describe panel's drill-through land on a set 6.7x larger than the row it
    # came from, because a facet could not be added to the selection already in
    # play.
    for one in ([attr] if isinstance(attr, str) else (attr or [])):
        if not one or ":" not in one:
            continue
        grp, label = one.split(":", 1)
        clauses.append(
            "s.id IN (SELECT sample_id FROM attributes WHERE grp = ? AND label = ?)")
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
    if ids:
        clause, id_params = id_list_clause(ids, staged=ids_staged)
        if clause:
            clauses.append(clause)
            params.extend(id_params)
    if max_agreement is not None:
        # Samples with *any* caption at or below the threshold. This is what
        # makes the Quality page's brush a real filter rather than a view-local
        # setting: the same predicate then works in the gallery and in export,
        # so a triage selection can leave the page it was made on.
        clauses.append(
            "s.id IN (SELECT sample_id FROM captions "
            "WHERE agreement IS NOT NULL AND agreement <= ?)")
        params.append(max_agreement)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def filtered_id_set(conn, split, tag, vlm_tag, attr=None, axes=None,
                    ids=None, ids_staged=False, max_agreement=None,
                    cluster=None, album=None) -> Optional[set[int]]:
    """Set of sample ids matching filters, or None when unfiltered."""
    if not (split or tag or vlm_tag or attr or axes or ids
            or max_agreement is not None or cluster is not None
            or album is not None):
        return None
    where, params = build_filters(split, tag, vlm_tag, attr, axes, ids, ids_staged,
                                  max_agreement, cluster, album)
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


def order_by_album_position(album: Optional[int]) -> tuple[str, list]:
    """ORDER BY the album's own item order, or ("", []) when no album filter.

    An album is an ordered collection — its position is user intent, so the
    filter path presents members in that order rather than by id. Explicit sort
    keys and query rankings still win; callers consult this only when neither
    is in force. Correlated rather than joined so the caller's FROM clause
    stays untouched.
    """
    if album is None:
        return "", []
    return (" ORDER BY (SELECT ai.position FROM album_items ai "
            "WHERE ai.album_id = ? AND ai.sample_id = s.id), s.id", [album])


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
