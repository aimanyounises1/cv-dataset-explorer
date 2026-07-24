"""API response models."""
from typing import Optional

from pydantic import BaseModel


class MatchPath(BaseModel):
    """Which retrieval path produced this result, and where it placed."""
    path: str                           # "keyword" | "semantic"
    rank: int                           # 1-based rank within that path


class SampleCard(BaseModel):
    id: int
    filename: str
    split: str
    width: Optional[int] = None
    height: Optional[int] = None
    thumb_url: str
    caption: Optional[str] = None       # representative caption
    score: Optional[float] = None       # search relevance, when applicable
    match_caption: Optional[str] = None  # the caption that explains this match
    matched_terms: Optional[list[str]] = None  # terms to highlight (keyword hits)
    match_paths: Optional[list[MatchPath]] = None  # why this result is here


class SampleList(BaseModel):
    items: list[SampleCard]
    total: int
    page: int
    per_page: int


class CaptionOut(BaseModel):
    text: str
    agreement: Optional[float] = None   # SigLIP image-caption similarity


class SampleDetail(BaseModel):
    id: int
    filename: str
    split: str
    width: Optional[int]
    height: Optional[int]
    filesize: Optional[int]
    image_url: str
    thumb_url: str
    captions: list[CaptionOut]
    tags: list[str]
    vlm_tags: list[str]
    attributes: dict[str, str] = {}
    cluster: Optional[int] = None
    caption_consistency: Optional[float] = None


class TermStat(BaseModel):
    """Document frequency of one query term, so the user can see when keyword
    ranking has nothing to discriminate on."""
    term: str
    images: int                         # images with the term in any caption
    fraction: float                     # images / corpus size
    common: bool                        # at or above the warning threshold


class SearchResponse(BaseModel):
    items: list[SampleCard]
    mode_used: str                      # actual mode after any fallback
    degraded: bool = False              # true if semantic search was unavailable
    message: Optional[str] = None
    score_basis: Optional[str] = None   # what `score` means: cosine | rrf | None
    rrf_k: Optional[int] = None         # fusion constant, when fusion ran
    term_stats: list[TermStat] = []     # per-term document frequency (lexical modes)


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


class SuspectCaption(BaseModel):
    caption: str
    agreement: float
    sibling_mean: Optional[float] = None  # mean agreement of the other 4 captions
    sample: SampleCard


class QASummary(BaseModel):
    available: bool
    scored_captions: int
    mean_agreement: Optional[float] = None


class AttributeLabel(BaseModel):
    label: str
    count: int
    fraction: float


class AttributeGroup(BaseModel):
    grp: str
    labels: list[AttributeLabel]


class EvalModeResult(BaseModel):
    mode: str
    recall_at: dict[str, float]         # {"1": .., "5": .., "10": ..}
    mrr: float = 0.0                    # mean reciprocal rank within the depth
    median_rank: Optional[float] = None  # None when the median falls past depth


class EvalResponse(BaseModel):
    available: bool
    message: Optional[str] = None
    sample_size: int = 0                # number of caption queries run
    pool_size: int = 0                  # candidate images each query ranks against
    depth: int = 0                      # rank depth MRR/median are computed to
    results: list[EvalModeResult] = []


class ChatMessage(BaseModel):
    role: str                            # "user" | "assistant"
    content: str


class ChatTraceStep(BaseModel):
    agent: str
    tool: str
    input: str


class ChatResponse(BaseModel):
    reply: str
    samples: list[SampleCard] = []
    trace: list[ChatTraceStep] = []
