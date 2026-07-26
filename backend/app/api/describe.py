"""Describe a set: "what do these samples have in common?"

Every other endpoint runs the arrow forwards — given a filter, return the
samples. This runs it backwards. Given any selection the product can already
make (a lasso, a quality brush, a pasted id list, a search, a saved view), it
reports what characterises that selection relative to the corpus.

The arithmetic is a lift: how much more often a facet appears inside the set than
in the dataset as a whole. Lift alone is a machine for producing confident
nonsense on small sets — 3 of 20 samples being "night" is a 6x lift and also
three photographs. So every facet is tested before it is reported, and the raw
count travels with the multiplier so a reader can always see what it rests on.

Nothing here is computed at request time that was not already stored: the
attribute labels, the axis buckets and the cluster assignments all come from
`python -m app.analyze`. The endpoint is an aggregation, not an analysis.
"""
import math
import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..db import AXES
from ..schemas import DescribeResponse, FacetLift, SetAxis
from .deps import (
    axis_bounds,
    build_filters,
    get_conn,
    id_list,
    stage_id_list,
)

router = APIRouter()

# A facet needs to clear both bars to be reported. The count floor is about
# meaningfulness (a 3-image pattern is an anecdote regardless of significance);
# the z threshold is about noise. Neither alone is enough.
MIN_FACET_COUNT = 5
MIN_ABS_Z = 2.0
MAX_FACETS = 12


def _z_score(k: int, n: int, big_k: int, big_n: int) -> float:
    """How surprising `k` hits in a set of `n` is, given `big_k` of `big_n` overall.

    Hypergeometric rather than binomial, because the set is drawn from the corpus
    *without replacement* — it is a subset, not a sample with replacement. The
    finite-population correction matters enormously here: filtering to
    `split=train` selects 6,000 of 8,000 samples, and without the correction that
    set's every facet would look wildly significant purely because n is large,
    when in fact a set containing three quarters of the corpus cannot differ much
    from it.
    """
    if n <= 1 or big_n <= 1 or n >= big_n:
        return 0.0
    p = big_k / big_n
    expected = n * p
    fpc = (big_n - n) / (big_n - 1)          # → 0 as the set approaches the corpus
    var = n * p * (1 - p) * fpc
    if var <= 0:
        return 0.0
    return (k - expected) / math.sqrt(var)


@router.get("/describe", response_model=DescribeResponse)
def describe_set(
    split: Optional[str] = None,
    tag: Optional[str] = None,
    vlm_tag: Optional[str] = None,
    attr: Optional[list[str]] = Query(
        None, description="Attribute facet 'group:label'. Repeatable: several "
                          "are intersected, so attr=time_of_day:night&"
                          "attr=setting:indoor is the images that are both."),
    cluster: Optional[int] = None,
    max_agreement: Optional[float] = Query(None),
    axes: dict = Depends(axis_bounds),
    ids: list = Depends(id_list),
    conn: sqlite3.Connection = Depends(get_conn),
):
    """What distinguishes the selected samples from the dataset as a whole.

    Takes exactly the parameters `/api/samples` takes, so any view that can build
    a selection can describe it without translating anything.
    """
    staged = stage_id_list(conn, ids) if ids else False
    where, params = build_filters(split, tag, vlm_tag, attr, axes, ids, staged,
                                  max_agreement, cluster)

    corpus_n = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    set_n = conn.execute(f"SELECT COUNT(*) FROM samples s{where}", params).fetchone()[0]

    if set_n == 0:
        return DescribeResponse(set_size=0, corpus_size=corpus_n, filtered=bool(where),
                                message="Nothing matches this selection.")
    if not where:
        return DescribeResponse(
            set_size=set_n, corpus_size=corpus_n, filtered=False,
            message="This is the whole dataset — there is nothing to compare it "
                    "against. Filter, search, lasso a region or brush a threshold "
                    "first.")

    # ---------------------------------------------------------------- facets
    corpus_facets = {
        (r["grp"], r["label"]): r["n"] for r in conn.execute(
            "SELECT grp, label, COUNT(*) AS n FROM attributes GROUP BY grp, label")}
    set_facets = {
        (r["grp"], r["label"]): r["n"] for r in conn.execute(
            f"SELECT a.grp AS grp, a.label AS label, COUNT(*) AS n "
            f"FROM attributes a JOIN samples s ON s.id = a.sample_id{where} "
            f"GROUP BY a.grp, a.label", params)}

    # A group the caller already filtered by can only report tautologies, so the
    # whole group is dropped rather than just the label that was asked for.
    #
    # `attributes` is PRIMARY KEY (sample_id, grp) — one label per group per
    # sample — so filtering to setting:indoor makes the set 100% indoor and
    # necessarily 0% every other `setting` label. The panel was spending its two
    # strongest slots saying so: "night x20.41" at the top of over-represented,
    # then "outdoor x0.00", "day x0.00", "dusk x0.00" filling under-represented.
    # All true, none of it a finding, and each offered a drill that could only
    # return the same set or nothing at all.
    applied = {one.split(":", 1)[0] for one in (attr or []) if ":" in one}

    over: list[FacetLift] = []
    under: list[FacetLift] = []
    for key, big_k in corpus_facets.items():
        grp, label = key
        if grp in applied:
            continue
        k = set_facets.get(key, 0)
        z = _z_score(k, set_n, big_k, corpus_n)
        if abs(z) < MIN_ABS_Z:
            continue
        corpus_share = big_k / corpus_n
        item = FacetLift(
            group=grp, label=label, count=k, set_share=round(k / set_n, 4),
            corpus_share=round(corpus_share, 4),
            lift=round((k / set_n) / corpus_share, 2) if corpus_share else 0.0,
            z=round(z, 1), drill=f"attr={grp}:{label}")
        # An absent facet is a real finding ("no night images at all here"), but
        # only when enough were expected for the absence to mean anything.
        if z > 0 and k >= MIN_FACET_COUNT:
            over.append(item)
        elif z < 0 and set_n * corpus_share >= MIN_FACET_COUNT:
            under.append(item)

    over.sort(key=lambda f: -f.lift)
    under.sort(key=lambda f: f.lift)

    # ------------------------------------------------------------------ axes
    axis_rows = []
    corpus_means = conn.execute(
        "SELECT " + ", ".join(f"AVG({a}) AS {a}" for a in AXES) + " FROM samples"
    ).fetchone()
    set_means = conn.execute(
        "SELECT " + ", ".join(f"AVG(s.{a}) AS {a}" for a in AXES)
        + f" FROM samples s{where}", params).fetchone()
    for a in AXES:
        if set_means[a] is None or corpus_means[a] is None:
            continue
        axis_rows.append(SetAxis(
            axis=a, set_mean=round(set_means[a], 2),
            corpus_mean=round(corpus_means[a], 2),
            delta=round(set_means[a] - corpus_means[a], 2)))
    axis_rows.sort(key=lambda x: -abs(x.delta))

    # -------------------------------------------------------- splits/clusters
    # `where` is guaranteed non-empty here (the unfiltered case returned above),
    # so it can be spliced as an AND — the same idiom the QA and search modules
    # use. Building it conditionally would be dead code pretending to be careful.
    and_where = where.replace(" WHERE ", " AND ", 1)

    splits = {r["split"]: r["n"] for r in conn.execute(
        f"SELECT s.split AS split, COUNT(*) AS n FROM samples s{where} "
        f"GROUP BY s.split", params)}
    clusters = [
        {"cluster": r["cluster"], "count": r["n"]}
        for r in conn.execute(
            f"SELECT s.cluster AS cluster, COUNT(*) AS n FROM samples s "
            f"WHERE s.cluster IS NOT NULL{and_where} "
            f"GROUP BY s.cluster ORDER BY n DESC LIMIT 6", params)]

    # ---------------------------------------------------------------- caption
    agree = conn.execute(
        f"SELECT AVG(c.agreement) AS m FROM captions c "
        f"JOIN samples s ON s.id = c.sample_id "
        f"WHERE c.agreement IS NOT NULL{and_where}", params).fetchone()
    corpus_agree = conn.execute(
        "SELECT AVG(agreement) FROM captions WHERE agreement IS NOT NULL").fetchone()[0]

    return DescribeResponse(
        set_size=set_n, corpus_size=corpus_n, filtered=True,
        over=over[:MAX_FACETS], under=under[:MAX_FACETS],
        axes=axis_rows, splits=splits, clusters=clusters,
        mean_agreement=round(agree["m"], 4) if agree and agree["m"] is not None else None,
        corpus_mean_agreement=round(corpus_agree, 4) if corpus_agree is not None else None,
        min_facet_count=MIN_FACET_COUNT, min_abs_z=MIN_ABS_Z,
    )
