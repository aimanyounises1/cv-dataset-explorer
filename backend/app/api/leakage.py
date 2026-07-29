"""Train/test leakage: held-out images that have a near-duplicate in training.

This is the failure mode with the largest documented effect on reported accuracy
that a dataset tool can detect from embeddings alone. Barz & Denzler found 3.3%
of CIFAR-10 and 10% of CIFAR-100 test images have near-duplicates in train, and
that removing them costs 9-14% relative accuracy — meaning a reported number was
partly measuring memorisation (*"Do we train on test data?"*, J. Imaging 6(6):41).

**What that citation does and does not license here.** In CIFAR the duplicated
thing is the photograph, which is why the accuracy it buys is memorisation. A
cosine cannot tell you the same is true of this corpus, because SigLIP is built
to be invariant to exactly what separates two photographs of one scene. So the
conclusion is not assumed: `/stats/leakage/pixel` hashes the flagged pairs
perceptually and reports what they actually share. On Flickr8k the answer is a
subject, not a frame — the contamination is real and worth excluding from a
benchmark, but "memorisation" would be borrowed rather than measured.

The tool already computed every embedding it needs. What it lacked was the split
comparison: the existing duplicate view shows the strongest 200 pairs as
thumbnails and never says which split either side came from, so the question
"how much of my held-out set is contaminated?" could not be asked.

**Why this endpoint returns a curve and not a number.** The answer depends
entirely on the threshold and the active embedding generation. A single
headline figure at a fixed threshold would present an arbitrary choice as a
measurement, so every response carries the whole ladder and the caller picks.
The pair thumbnails are the point of appeal: candidate near-duplicates must be
looked at before they are believed.
"""
import sqlite3
import threading
from typing import Literal, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from .. import config
from ..ml import phash
from ..ml.index import get_index
from ..schemas import (
    LeakageContamination,
    LeakagePair,
    LeakagePoint,
    LeakageReport,
)
from .deps import MAX_ID_LIST, get_conn, thumb_url

router = APIRouter()

# The ladder every response reports. Fixed rather than caller-supplied so two
# reports are always comparable, and floored at 0.85 because below it "near
# duplicate" stops meaning anything on SigLIP cosines.
LADDER = (0.85, 0.90, 0.92, 0.95, 0.97, 0.99)
FLOOR = LADDER[0]
HeldOutSplit = Literal["test", "validation"]

# The O(n²) scan is cached for exactly one live index object. A strong reference
# matters here: two providers deliberately have the same ordered sample IDs but
# different vectors, so an ID-derived key would mix their leakage results. It
# also prevents CPython from reusing the cached object's address. The lock keeps
# the report and contaminated-id requests from launching the cold scan twice.
_cache_lock = threading.Lock()
_cached_index = None
_cached_pairs: list[tuple[int, int, float]] | None = None


def _pairs(index) -> list[tuple[int, int, float]]:
    global _cached_index, _cached_pairs
    with _cache_lock:
        if _cached_index is not index:
            pairs = index.all_pairs_above(FLOOR)
            _cached_index = index
            _cached_pairs = pairs
        assert _cached_pairs is not None
        return _cached_pairs


def clear_cache() -> None:
    """Drop the cached pair list. Called by /api/admin/reload.

    Clearing both references releases the superseded index and its potentially
    large pair list together.
    """
    global _cached_index, _cached_pairs
    with _cache_lock:
        _cached_index = None
        _cached_pairs = None


def _contaminated_ids(pairs: list[tuple[int, int, float]],
                      splits: dict[int, str],
                      held_out: set[str],
                      threshold: float) -> set[int]:
    """Held-out ids having at least one training near-duplicate above `threshold`.

    The set, not its size. Reporting only the count answers "how bad is it?" and
    leaves "which ones?" — the question you act on — to a re-derivation the tool
    already did. One implementation, shared by the report and the id endpoint,
    so the number on the page and the ids you export can never disagree.
    """
    out: set[int] = set()
    for a, b, score in pairs:
        if score <= threshold:
            break                         # sorted by score, so the rest are below
        for x, y in ((a, b), (b, a)):
            if splits.get(x) in held_out and splits.get(y) == "train":
                out.add(x)
    return out


@router.get("/stats/leakage/contaminated", response_model=LeakageContamination)
def contaminated_ids(
    threshold: float = Query(0.90, ge=FLOOR, le=1.0),
    split: Optional[HeldOutSplit] = Query(
        None, description="Restrict the held-out side"
    ),
    conn: sqlite3.Connection = Depends(get_conn),
):
    """The contaminated held-out ids themselves, as a set you can act on.

    `/stats/leakage` measures the problem; this hands it over. The ids go
    straight into `?ids=` (the gallery's own selection vocabulary) and into
    `/api/export?ids=`, so "12% of my test split is contaminated" becomes a
    slice a researcher can open, tag, or exclude — without re-deriving in numpy
    what this endpoint already computed.

    Capped at the same ceiling the rest of the id vocabulary uses, and says so
    when it truncates rather than silently handing back a short list.
    """
    index = get_index()
    if index is None:
        raise HTTPException(
            503, "Leakage detection needs image embeddings — run "
                 "`python -m app.ingest` first.")

    splits = {r["id"]: r["split"] for r in conn.execute("SELECT id, split FROM samples")}
    held_out = {"test", "validation"} if split is None else {split}
    found = sorted(_contaminated_ids(_pairs(index), splits, held_out, threshold))
    return LeakageContamination(
        threshold=threshold,
        held_out_split=split,
        total=len(found),
        ids=found[:MAX_ID_LIST],
        truncated=len(found) > MAX_ID_LIST,
    )


@router.get("/stats/leakage", response_model=LeakageReport)
def leakage_report(
    threshold: float = Query(0.90, ge=FLOOR, le=1.0),
    limit: int = Query(40, ge=0, le=200),
    split: Optional[HeldOutSplit] = Query(
        None, description="Restrict the held-out side"
    ),
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
        for a, b, score in pairs:
            if score <= th:
                break                     # sorted by score, so the rest are below
            total += 1
            sa, sb = splits.get(a), splits.get(b)
            if sa != sb:
                cross += 1
        return total, cross, _contaminated_ids(pairs, splits, held_out, th)

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


# ---------------------------------------------------------------------------
# Pixel-level check: are these the same photograph, or only the same scene?
# ---------------------------------------------------------------------------

# Models live here rather than in `schemas.py` because they describe one
# endpoint's answer and nothing else reads them; `detect.py` sets the same
# precedent. The rest of the leakage contract stays where it was.

class PixelPair(BaseModel):
    a_id: int
    b_id: int
    score: float
    distance: int
    duplicate_frame: bool
    a_split: str
    b_split: str
    a_thumb: str
    b_thumb: str


class PixelReport(BaseModel):
    threshold: float
    pairs_measured: int
    duplicate_frames: int
    duplicate_frame_fraction: float
    #: The cut, and what it was derived from — so the number is never a
    #: bare threshold a reader has to take on trust.
    duplicate_frame_max_distance: int
    median_distance: float
    #: Chance, measured on this corpus in the same request.
    unrelated_median_distance: float
    correlation_with_cosine: float
    closest: list[PixelPair] = []
    reading: str


_phash_lock = threading.Lock()
_phash_cache: dict[int, bytes] = {}


def _hash_for(sample_id: int, filename: str) -> bytes:
    with _phash_lock:
        cached = _phash_cache.get(sample_id)
    if cached is not None:
        return cached
    digest = phash.dhash(config.IMAGES_DIR / filename)
    with _phash_lock:
        _phash_cache[sample_id] = digest
    return digest


@router.get("/stats/leakage/pixel", response_model=PixelReport)
def pixel_report(
    threshold: float = Query(0.90, ge=FLOOR, le=1.0),
    limit: int = Query(12, ge=0, le=60),
    conn: sqlite3.Connection = Depends(get_conn),
):
    """Whether the cross-split near-duplicates are duplicated *photographs*.

    The leakage report borrows its stakes from Barz & Denzler, whose CIFAR
    finding is about the same photograph appearing in both splits — there,
    reported accuracy is partly memorisation. That reading transfers to this
    corpus only if these pairs are the same photograph too, and cosine cannot
    say: SigLIP is trained to be invariant to exactly what separates two shots
    of one scene.

    So this measures it instead of assuming it, and reports the null in the
    same breath: if the median distance between flagged pairs is the same as
    the median between randomly chosen images, then whatever these pairs share
    is not their pixels.

    Costs one 0.8 ms hash per distinct image in a flagged pair, once per
    process. It is a separate endpoint precisely because it reads image files
    while `/stats/leakage` reads only vectors — that one stays fast.
    """
    index = get_index()
    if index is None:
        raise HTTPException(
            503, "The pixel check needs the embedding index to know which pairs "
                 "to look at — run `python -m app.ingest` first.")

    rows = {r["id"]: r for r in conn.execute(
        "SELECT id, split, filename FROM samples")}
    cross = [(a, b, s) for a, b, s in _pairs(index)
             if s > threshold and a in rows and b in rows
             and rows[a]["split"] != rows[b]["split"]]

    measured: list[tuple[int, int, float, int]] = []
    digests: list[bytes] = []
    seen: set[int] = set()
    for a, b, score in cross:
        try:
            ha, hb = (_hash_for(i, rows[i]["filename"]) for i in (a, b))
        except OSError:
            # An unreadable file is one pair we cannot speak for, not a failed
            # request — the count of what WAS measured is reported either way.
            continue
        measured.append((a, b, score, phash.hamming(ha, hb)))
        for i, h in ((a, ha), (b, hb)):
            if i not in seen:
                seen.add(i)
                digests.append(h)

    if not measured:
        return PixelReport(
            threshold=threshold, pairs_measured=0, duplicate_frames=0,
            duplicate_frame_fraction=0.0,
            duplicate_frame_max_distance=phash.DUPLICATE_FRAME_MAX,
            median_distance=0.0, unrelated_median_distance=0.0,
            correlation_with_cosine=0.0, closest=[],
            reading=("No cross-split pair sits above this threshold, so there is "
                     "nothing to check. Lower the threshold to widen the net."))

    distances = np.array([d for *_, d in measured], dtype=float)
    scores = np.array([s for _, _, s, _ in measured], dtype=float)
    same = int((distances <= phash.DUPLICATE_FRAME_MAX).sum())
    null = phash.null_distance(digests)
    # A constant column has no correlation to report; numpy would warn and
    # return nan. Saying 0.0 there is not a fudge: nothing varies, so nothing
    # co-varies.
    corr = (float(np.corrcoef(scores, distances)[0, 1])
            if distances.std() > 0 and scores.std() > 0 else 0.0)

    measured.sort(key=lambda m: m[3])
    closest = [
        PixelPair(
            a_id=a, b_id=b, score=round(s, 4), distance=d,
            duplicate_frame=d <= phash.DUPLICATE_FRAME_MAX,
            a_split=rows[a]["split"], b_split=rows[b]["split"],
            a_thumb=thumb_url(rows[a]["filename"]),
            b_thumb=thumb_url(rows[b]["filename"]))
        for a, b, s, d in measured[:limit]
    ]

    median = float(np.median(distances))
    if same == 0:
        reading = (
            f"None of the {len(measured)} cross-split pairs above {threshold:g} "
            f"is a duplicate frame. Their median distance is {median:.0f} of "
            f"{phash.HASH_BITS} bits, against {null:.0f} for randomly paired "
            f"images from this corpus — so at the pixel level these are no more "
            f"alike than any two images here. They share a subject, not a frame: "
            f"accuracy on them measures generalisation across near-identical "
            f"scenes, not memorisation of an image already seen.")
    else:
        subject = ("1 of {} cross-split pairs above {:g} is a duplicate frame"
                   if same == 1 else
                   "{2} of {0} cross-split pairs above {1:g} are duplicate frames")
        reading = (
            subject.format(len(measured), threshold, same)
            + f" — within {phash.DUPLICATE_FRAME_MAX} of {phash.HASH_BITS} bits, "
            f"meaning one shot rescaled, re-saved, or a burst neighbour. The full "
            f"set has a median distance of {median:.0f} against {null:.0f} for "
            f"randomly paired images here, so the rest share a subject rather "
            f"than a frame. Open the closest pairs before reading either number "
            f"as memorisation.")

    return PixelReport(
        threshold=threshold, pairs_measured=len(measured), duplicate_frames=same,
        duplicate_frame_fraction=round(same / len(measured), 4),
        duplicate_frame_max_distance=phash.DUPLICATE_FRAME_MAX,
        median_distance=median, unrelated_median_distance=null,
        correlation_with_cosine=round(corr, 3), closest=closest,
        reading=reading)
