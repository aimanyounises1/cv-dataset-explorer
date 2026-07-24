"""Embedding map: 2-D UMAP projection of every image, for the scatter view."""
import sqlite3

from fastapi import APIRouter, Depends

from ..schemas import MapPoint
from .deps import get_conn, thumb_url

router = APIRouter()


@router.get("/map", response_model=list[MapPoint])
def embedding_map(conn: sqlite3.Connection = Depends(get_conn)):
    rows = conn.execute(
        "SELECT id, filename, umap_x, umap_y, cluster FROM samples "
        "WHERE umap_x IS NOT NULL AND umap_y IS NOT NULL"
    ).fetchall()
    return [
        MapPoint(id=r["id"], x=r["umap_x"], y=r["umap_y"],
                 cluster=r["cluster"] if r["cluster"] is not None else 0,
                 thumb_url=thumb_url(r["filename"]))
        for r in rows
    ]
