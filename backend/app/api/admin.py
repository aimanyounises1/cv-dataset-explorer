"""Operational endpoints: pick up new ingestion/analysis output without a
server restart."""
from fastapi import APIRouter

from ..ml.index import get_caption_index, get_index, invalidate_index

router = APIRouter()


@router.post("/admin/reload")
def reload_indexes():
    from . import stats

    invalidate_index()
    stats.clear_caches()
    return {
        "ok": True,
        "image_index": get_index() is not None,
        "caption_index": get_caption_index() is not None,
    }
