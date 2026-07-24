"""API response models."""
from typing import Optional

from pydantic import BaseModel


class SampleCard(BaseModel):
    id: int
    filename: str
    split: str
    width: Optional[int] = None
    height: Optional[int] = None
    thumb_url: str
    caption: Optional[str] = None       # representative caption
    score: Optional[float] = None       # search relevance, when applicable


class SampleList(BaseModel):
    items: list[SampleCard]
    total: int
    page: int
    per_page: int


class SampleDetail(BaseModel):
    id: int
    filename: str
    split: str
    width: Optional[int]
    height: Optional[int]
    filesize: Optional[int]
    image_url: str
    thumb_url: str
    captions: list[str]
    tags: list[str]
    vlm_tags: list[str]
    cluster: Optional[int] = None


class SearchResponse(BaseModel):
    items: list[SampleCard]
    mode_used: str                      # actual mode after any fallback
    degraded: bool = False              # true if semantic search was unavailable
    message: Optional[str] = None


class StatsOverview(BaseModel):
    total_samples: int
    total_captions: int
    splits: dict[str, int]
    avg_caption_length_words: float
    image_size_buckets: dict[str, int]
    embeddings_available: bool
    vlm_enriched: int


class CaptionStats(BaseModel):
    length_histogram: list[dict]        # [{bucket, count}]
    top_words: list[dict]               # [{word, count}]


class DuplicatePair(BaseModel):
    a: SampleCard
    b: SampleCard
    similarity: float


class MapPoint(BaseModel):
    id: int
    x: float
    y: float
    cluster: int
    thumb_url: str


class TagInfo(BaseModel):
    name: str
    count: int
