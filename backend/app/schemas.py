"""API response models."""
from typing import Optional

from pydantic import BaseModel


class AxisScores(BaseModel):
    """Difficulty axes, 0-10 percentile buckets over this dataset.

    Not measurements — ranks. A 7 means "harder than roughly 70% of *this*
    corpus", so the numbers do not transfer to another dataset. `detail` carries
    the raw components behind each axis so a score can be explained in place.
    """
    legibility: Optional[int] = None    # Clear(0) -> Blind(10)
    rarity: Optional[int] = None        # Not rare(0) -> Very rare(10)
    difficulty: Optional[int] = None    # Routine(0) -> Hard(10)
    clutter: Optional[int] = None       # Simple(0) -> Busy(10)
    detail: dict = {}


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
    axes: Optional[AxisScores] = None   # why this item is interesting


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
    axes: Optional[AxisScores] = None


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
    offset: int = 0                     # window start within the full ranking
    has_more: bool = False              # a further page exists
    sort: Optional[str] = None          # axis sort in force; None = relevance order
    # How many pasted id-list entries exist in this dataset. Reported, not
    # enforced: a list carried over from a larger corpus is normal.
    ids_resolved: Optional[int] = None


class SearchRequest(BaseModel):
    """POST body for search, carrying what a URL cannot.

    Mirrors the GET parameters exactly; it exists only because an id list of the
    size this tool accepts does not fit in a query string.
    """
    q: str
    mode: str = "hybrid"
    top_k: int = 60
    offset: int = 0
    split: Optional[str] = None
    tag: Optional[str] = None
    vlm_tag: Optional[str] = None
    attr: Optional[str] = None
    sort: Optional[str] = None
    ids: Optional[str] = None            # raw pasted text, parsed server-side
    axes: dict[str, dict[str, Optional[int]]] = {}   # {"difficulty": {"min": 8}}


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
    # Label fractions are of the whole corpus, so they sum to less than 1 and the
    # shortfall is exactly the abstention rate. Renormalising over labelled
    # samples would make every group look like full coverage.
    labelled: int = 0
    abstained: int = 0                    # below the decisiveness margin
    mean_confidence: Optional[float] = None


class EvalModeResult(BaseModel):
    mode: str
    recall_at: dict[str, float]         # {"1": .., "5": .., "10": ..}
    mrr: float = 0.0                    # mean reciprocal rank within the depth
    median_rank: Optional[float] = None  # None when the median falls past depth
    # What this mode actually got to rank. A recall figure is only about
    # ranking quality if there was something to rank.
    mean_candidates: float = 0.0
    empty_query_rate: float = 0.0       # fraction of queries with no candidates


class EvalResponse(BaseModel):
    available: bool
    message: Optional[str] = None
    sample_size: int = 0                # number of caption queries run
    pool_size: int = 0                  # corpus size (the semantic candidate pool)
    depth: int = 0                      # rank depth MRR/median are computed to
    mean_query_words: float = 0.0       # queries are whole captions, not phrases
    results: list[EvalModeResult] = []


class SavedView(BaseModel):
    """A named filter set, held as the URL query string that produced it.

    `query_string` is opaque to the server: storing the URL rather than a parsed
    filter object means a view survives the UI gaining filters the API has no
    column for, at the cost of not being able to validate or migrate one.
    """
    name: str
    query_string: str
    created_at: str                      # ISO-8601, UTC


class SavedViewCreate(BaseModel):
    name: str
    query_string: str


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
