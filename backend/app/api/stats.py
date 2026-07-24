"""Dataset statistics: overview, caption distributions, near-duplicates."""
import json
import re
import sqlite3
from collections import Counter
from typing import Optional

from fastapi import APIRouter, Depends

from .. import config
from ..db import STOPWORDS
from ..ml.index import get_index
from ..schemas import CaptionStats, DuplicatePair, StatsOverview
from .deps import first_captions, get_conn, row_to_card

router = APIRouter()

_caption_stats_cache: Optional[CaptionStats] = None


def clear_caches() -> None:
    """Reset in-memory + on-disk stats caches (called by /api/admin/reload)."""
    global _caption_stats_cache
    _caption_stats_cache = None
    if config.CACHE_DIR.exists():
        for f in config.CACHE_DIR.glob("*.json"):
            f.unlink(missing_ok=True)


@router.get("/stats/overview", response_model=StatsOverview)
def overview(conn: sqlite3.Connection = Depends(get_conn)):
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    total_caps = conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0]
    splits = {r["split"]: r["n"] for r in conn.execute(
        "SELECT split, COUNT(*) AS n FROM samples GROUP BY split")}
    avg_len = conn.execute(
        "SELECT AVG(LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1) FROM captions"
    ).fetchone()[0] or 0.0
    buckets: Counter = Counter()
    for r in conn.execute("SELECT width, height FROM samples"):
        if not r["width"] or not r["height"]:
            continue
        long_side = max(r["width"], r["height"])
        if long_side < 400:
            buckets["<400px"] += 1
        elif long_side < 500:
            buckets["400-499px"] += 1
        else:
            buckets["500px+"] += 1
    enriched = conn.execute("SELECT COUNT(DISTINCT sample_id) FROM vlm_tags").fetchone()[0]
    return StatsOverview(
        total_samples=total, total_captions=total_caps, splits=splits,
        avg_caption_length_words=round(float(avg_len), 2),
        image_size_buckets=dict(buckets),
        embeddings_available=get_index() is not None,
        vlm_enriched=enriched,
    )


@router.get("/stats/captions", response_model=CaptionStats)
def caption_stats(conn: sqlite3.Connection = Depends(get_conn)):
    global _caption_stats_cache
    if _caption_stats_cache is not None:
        return _caption_stats_cache

    lengths: Counter = Counter()
    words: Counter = Counter()
    for r in conn.execute("SELECT text FROM captions"):
        tokens = re.findall(r"[a-z']+", r["text"].lower())
        bucket = min(len(tokens) // 5 * 5, 40)
        lengths[bucket] += 1
        words.update(t for t in tokens if t not in STOPWORDS and len(t) > 2)

    _caption_stats_cache = CaptionStats(
        length_histogram=[
            {"bucket": f"{b}-{b + 4}" if b < 40 else "40+", "count": c}
            for b, c in sorted(lengths.items())
        ],
        top_words=[{"word": w, "count": c} for w, c in words.most_common(30)],
    )
    return _caption_stats_cache


@router.get("/stats/duplicates", response_model=list[DuplicatePair])
def duplicates(conn: sqlite3.Connection = Depends(get_conn)):
    index = get_index()
    if index is None:
        return []
    cache_file = config.CACHE_DIR / f"duplicates_{config.DUPLICATE_THRESHOLD}.json"
    if cache_file.exists():
        pairs = json.loads(cache_file.read_text())
    else:
        pairs = index.duplicate_pairs()
        config.ensure_dirs()
        cache_file.write_text(json.dumps(pairs))

    ids = sorted({i for p in pairs for i in (p[0], p[1])})
    if not ids:
        return []
    qmarks = ",".join("?" * len(ids))
    rows = {r["id"]: r for r in conn.execute(
        f"SELECT * FROM samples WHERE id IN ({qmarks})", ids)}
    captions = first_captions(conn, ids)
    out = []
    for a, b, sim in pairs:
        if a in rows and b in rows:
            out.append(DuplicatePair(
                a=row_to_card(rows[a], caption=captions.get(a)),
                b=row_to_card(rows[b], caption=captions.get(b)),
                similarity=round(sim, 4),
            ))
    return out
