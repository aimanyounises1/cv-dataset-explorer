"""Embedding map: 2-D UMAP projection of every image, for the scatter view.

Each point carries the dimensions the UI can colour by. Cluster id alone was
not enough: "cluster 7" is a nominal label a researcher cannot interpret, so
the map could not answer the question it is best placed to answer — where in
embedding space is the model weakest? Agreement and the difficulty axes make
that a colour.
"""
import sqlite3

from fastapi import APIRouter, Depends

from ..schemas import MapPoint
from .deps import get_conn, thumb_url

router = APIRouter()


@router.get("/map", response_model=list[MapPoint])
def embedding_map(conn: sqlite3.Connection = Depends(get_conn)):
    # Agreement is per-caption, so it is averaged per sample here rather than in
    # Python: one grouped subquery beats 8,000 round trips.
    rows = conn.execute(
        "SELECT s.id, s.filename, s.umap_x, s.umap_y, s.cluster, s.split, "
        "       s.legibility, s.rarity, s.difficulty, s.clutter, a.mean_agreement "
        "FROM samples s "
        "LEFT JOIN (SELECT sample_id, AVG(agreement) AS mean_agreement "
        "           FROM captions WHERE agreement IS NOT NULL "
        "           GROUP BY sample_id) a ON a.sample_id = s.id "
        "WHERE s.umap_x IS NOT NULL AND s.umap_y IS NOT NULL"
    ).fetchall()
    return [
        MapPoint(
            id=r["id"], x=r["umap_x"], y=r["umap_y"],
            cluster=r["cluster"] if r["cluster"] is not None else 0,
            thumb_url=thumb_url(r["filename"]),
            split=r["split"],
            agreement=round(r["mean_agreement"], 4) if r["mean_agreement"] is not None else None,
            legibility=r["legibility"], rarity=r["rarity"],
            difficulty=r["difficulty"], clutter=r["clutter"],
        )
        for r in rows
    ]
