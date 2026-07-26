"""Train/test leakage: held-out images that have a near-duplicate in training.

This is the failure mode with the largest documented effect on reported accuracy
that a dataset tool can detect from embeddings alone. Barz & Denzler found 3.3%
of CIFAR-10 and 10% of CIFAR-100 test images have near-duplicates in train, and
that removing them costs 9-14% relative accuracy — meaning a reported number was
partly measuring memorisation (*"Do we train on test data?"*, J. Imaging 6(6):41).

The tool already computed every embedding it needs. What it lacked was the split
comparison: the existing duplicate view shows the strongest 200 pairs as
thumbnails and never says which split either side came from, so the question
"how much of my held-out set is contaminated?" could not be asked.

**Why this endpoint returns a curve and not a number.** The answer depends
entirely on where the threshold is put, and on this corpus that dependence is
violent — measured here: at cosine 0.95 exactly 16 held-out images (0.80%) have a
train near-duplicate, at 0.90 it is 241 (12.05%). A single headline figure at a
hard-coded threshold would be an arbitrary choice presented as a measurement, so
every response carries the whole ladder and the caller picks. The pair thumbnails
are the point of appeal: 0.90-cosine "duplicates" must be looked at before they
are believed.
"""
import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from .. import config
from ..ml.index import get_index
from ..schemas import LeakagePair, LeakagePoint, LeakageReport
from .deps import get_conn, thumb_url

router = APIRouter()

# The ladder every response reports. Fixed rather than caller-supplied so two
# reports are always comparable, and floored at 0.85 because below it "near
# duplicate" stops meaning anything on SigLIP cosines.
LADDER = (0.85, 0.90, 0.92, 0.95, 0.97, 0.99)
FLOOR = LADDER[0]

# Cached on the pair list, which is the expensive part; every threshold in the
# ladder is then a filter over it. Keyed by the embedding index identity so a
# reload invalidates it.
_cache: dict[int, list[tuple[int, int, float]]] = {}


def _pairs(index) -> list[tuple[int, int, float]]:
    key = id(index)
    if key not in _cache:
        _cache.clear()                    # only ever one index generation is live
        _cache[key] = index.all_pairs_above(FLOOR)
    return _cache[key]


@router.get("/stats/leakage", response_model=LeakageReport)
def leakage_report(
    threshold: float = Query(0.90, ge=FLOOR, le=1.0),
    limit: int = Query(40, ge=0, le=200),
    split: Optional[str] = Query(None, description="Restrict the held-out side"),
    conn: sqlite3.Connection = Depends(get_conn),
):
    index = get_index()
    if index is None:
        raise HTTPException(
            503, "Leakage detection needs image embeddings — run "
                 "`python -m app.ingest` first.")

    pairs = _pairs(index)
    splits = {r["id"]: r["split"] for r in conn.execute("SELECT id, split FROM samples")}
    held_out = {"test", "validation"} if split is None else {split}

    def summarise(th: float) -> tuple[int, int, set[int]]:
        """(pairs, cross-split pairs, held-out ids with a train near-duplicate)."""
        total = cross = 0
        contaminated: set[int] = set()
        for a, b, score in pairs:
            if score <= th:
                break                     # sorted by score, so the rest are below
            total += 1
            sa, sb = splits.get(a), splits.get(b)
            if sa != sb:
                cross += 1
            for x, y in ((a, b), (b, a)):
                if splits.get(x) in held_out and splits.get(y) == "train":
                    contaminated.add(x)
        return total, cross, contaminated

    curve = []
    for th in LADDER:
        total, cross, contaminated = summarise(th)
        curve.append(LeakagePoint(threshold=th, pairs=total, cross_split=cross,
                                  contaminated=len(contaminated)))

    total, cross, contaminated = summarise(threshold)
    held_out_total = sum(1 for s in splits.values() if s in held_out)

    # Cross-split breakdown at the chosen threshold, and the examples to show.
    by_pair: dict[str, int] = {}
    shortlist: list[tuple[int, int, float]] = []
    for a, b, score in pairs:
        if score <= threshold:
            break
        sa, sb = splits.get(a, "?"), splits.get(b, "?")
        if sa != sb:
            key = "~".join(sorted((sa, sb)))
            by_pair[key] = by_pair.get(key, 0) + 1
            # Cross-split pairs lead the list: they are the ones that matter, and
            # a page of within-train duplicates would bury them.
            if len(shortlist) < limit:
                shortlist.append((a, b, score))
    if len(shortlist) < limit:
        for a, b, score in pairs:
            if score <= threshold or len(shortlist) >= limit:
                break
            if splits.get(a) == splits.get(b):
                shortlist.append((a, b, score))

    # One query for every thumbnail, not one per pair.
    wanted = {i for pair in shortlist for i in pair[:2]}
    rows = {}
    if wanted:
        qmarks = ",".join("?" * len(wanted))
        rows = {r["id"]: r for r in conn.execute(
            f"SELECT id, filename, split FROM samples WHERE id IN ({qmarks})",
            list(wanted))}
    listed = [
        LeakagePair(
            a_id=a, b_id=b, score=round(score, 4),
            a_split=rows[a]["split"], b_split=rows[b]["split"],
            a_thumb=thumb_url(rows[a]["filename"]),
            b_thumb=thumb_url(rows[b]["filename"]),
            cross_split=rows[a]["split"] != rows[b]["split"])
        for a, b, score in shortlist if a in rows and b in rows
    ]

    return LeakageReport(
        threshold=threshold, floor=FLOOR,
        pairs=total, cross_split_pairs=cross, by_split_pair=by_pair,
        contaminated=len(contaminated), held_out_total=held_out_total,
        contaminated_fraction=round(len(contaminated) / held_out_total, 4)
                              if held_out_total else 0.0,
        curve=curve, examples=listed,
        default_threshold=config.DUPLICATE_THRESHOLD,
        caveat=(
            "Near-duplicate means SigLIP cosine above the threshold, which is not "
            "a calibrated notion of 'the same image' — look at the pairs before "
            "believing a count. The split assignments come from this copy of the "
            "dataset and are not independently verified, so this describes this "
            "copy. Every figure moves with the threshold: see the curve."),
    )
