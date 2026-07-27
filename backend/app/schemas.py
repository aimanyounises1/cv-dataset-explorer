"""API response models."""
from typing import Optional, Union

from pydantic import BaseModel, Field


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
    degraded: bool = False              # true if the requested ranking fell back
    message: Optional[str] = None
    # What `score` means: cosine | cosine_adj | rrf | prism_ll | None
    score_basis: Optional[str] = None
    rrf_k: Optional[int] = None         # fusion constant, when fusion ran
    term_stats: list[TermStat] = []     # per-term document frequency (lexical modes)
    offset: int = 0                     # window start within the full ranking
    has_more: bool = False              # a further page exists
    sort: Optional[str] = None          # axis sort in force; None = relevance order
    # How many pasted id-list entries exist in this dataset. Reported, not
    # enforced: a list carried over from a larger corpus is normal.
    ids_resolved: Optional[int] = None
    # How deep the fusion ranked, and whether the caller has reached it. Paging
    # stops here rather than widening, because a widened pool re-ranks the tail
    # and would repeat images across adjacent pages.
    depth_limit: int = 0
    depth_reached: bool = False


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
    # One facet or several, intersected. A bare string is still accepted so the
    # POST body stays compatible with clients that only ever sent one.
    attr: Optional[Union[str, list[str]]] = None
    sort: Optional[str] = None
    ids: Optional[str] = None            # raw pasted text, parsed server-side
    axes: dict[str, dict[str, Optional[int]]] = {}   # {"difficulty": {"min": 8}}
    max_agreement: Optional[float] = None


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
    """One image in the projection, with the dimensions worth colouring by.

    Sent flat rather than nested: this is 8,000 rows, and a nested object per
    point roughly doubles the payload for no gain.
    """
    id: int
    x: float
    y: float
    cluster: int
    thumb_url: str
    split: str
    agreement: Optional[float] = None     # mean image-caption agreement
    legibility: Optional[int] = None
    rarity: Optional[int] = None
    difficulty: Optional[int] = None
    clutter: Optional[int] = None


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
    # The distribution, so a reviewer can see where a sensible cutoff sits
    # instead of trusting a fixed top-50. Bins are over the observed range,
    # because agreement is a cosine that occupies a narrow slice of [0,1] and
    # binning across the full interval would put every caption in one bar.
    histogram: list[dict] = []           # [{lo, hi, count}]
    min_agreement: Optional[float] = None
    max_agreement: Optional[float] = None


class QASelection(BaseModel):
    """How large the current review threshold's selection is, in both units."""
    max_agreement: float
    captions: int
    samples: int


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
    # Set on rows measured over a different query set than the main sample
    # (the PRISM rows use test-split queries only). A number without its
    # sample size and its reason invites a comparison it cannot support.
    queries: Optional[int] = None
    note: Optional[str] = None


class EvalResponse(BaseModel):
    available: bool
    message: Optional[str] = None
    sample_size: int = 0                # number of caption queries run
    pool_size: int = 0                  # corpus size (the semantic candidate pool)
    depth: int = 0                      # rank depth MRR/median are computed to
    mean_query_words: float = 0.0       # queries are whole captions, not phrases
    results: list[EvalModeResult] = []


class LeakagePoint(BaseModel):
    """One rung of the threshold ladder. Reported as a curve because the answer
    moves violently with the cut: on this corpus 0.80% of held-out images look
    contaminated at cosine 0.95 and 12.05% at 0.90."""
    threshold: float
    pairs: int
    cross_split: int
    contaminated: int


class LeakagePair(BaseModel):
    a_id: int
    b_id: int
    score: float
    a_split: str
    b_split: str
    a_thumb: str
    b_thumb: str
    cross_split: bool


class LeakageReport(BaseModel):
    threshold: float
    floor: float
    pairs: int                           # total near-duplicate pairs, uncapped
    cross_split_pairs: int
    by_split_pair: dict[str, int] = {}   # {"test~train": n, ...}
    # The statistic that matters: distinct held-out images with at least one
    # training near-duplicate. Reported accuracy on those is partly memorisation.
    contaminated: int
    held_out_total: int
    contaminated_fraction: float
    curve: list[LeakagePoint] = []
    examples: list[LeakagePair] = []
    default_threshold: float
    caveat: str


class FacetLift(BaseModel):
    """One attribute label, and how over- or under-represented it is in a set.

    `count` travels with `lift` on purpose: a 6x multiplier over three images and
    a 6x multiplier over three hundred are different findings, and a reader shown
    only the multiplier cannot tell them apart. `z` is the hypergeometric score
    the facet had to clear to be reported at all.
    """
    group: str
    label: str
    count: int
    set_share: float
    corpus_share: float
    lift: float
    z: float
    drill: str                           # gallery query string for this facet


class SetAxis(BaseModel):
    axis: str
    set_mean: float
    corpus_mean: float
    delta: float


class DescribeResponse(BaseModel):
    """What characterises a selection, relative to the whole dataset."""
    set_size: int
    corpus_size: int
    filtered: bool
    message: Optional[str] = None
    over: list[FacetLift] = []
    under: list[FacetLift] = []
    axes: list[SetAxis] = []
    splits: dict[str, int] = {}
    clusters: list[dict] = []
    mean_agreement: Optional[float] = None
    corpus_mean_agreement: Optional[float] = None
    # The thresholds a facet had to clear, reported so the absence of a facet is
    # interpretable rather than mysterious.
    min_facet_count: int = 0
    min_abs_z: float = 0.0


class SavedView(BaseModel):
    """A named filter set, held as the URL query string that produced it.

    `query_string` is opaque to the server: storing the URL rather than a parsed
    filter object means a view survives the UI gaining filters the API has no
    column for, at the cost of not being able to validate or migrate one.
    """
    name: str
    query_string: str
    created_at: str                      # ISO-8601, UTC
    # A human-readable warning when the view was saved under a different
    # embedding model or corpus than the one now loaded. None when the
    # environments match — or when the view predates fingerprinting and there
    # is nothing to compare, in which case silence is the honest answer.
    stale_env: Optional[str] = None


class SavedViewCreate(BaseModel):
    # Bounded because unbounded was measured: a 200,000-character name was
    # accepted and stored. 200 covers any label a human types; the query
    # string ceiling sits above the ~64 KiB URL transport limit, so no view
    # a browser can actually carry is refused.
    name: str = Field(max_length=200)
    query_string: str = Field(max_length=100_000)


class ChatMessage(BaseModel):
    role: str                            # "user" | "assistant"
    content: str


class ChatTraceStep(BaseModel):
    agent: str
    tool: str
    input: str
    # Wall clock of the specialist lane this step ran in, so the trace shows
    # where a slow turn went rather than only what it did.
    lane_seconds: Optional[float] = None


class ChatResponse(BaseModel):
    reply: str
    samples: list[SampleCard] = []
    trace: list[ChatTraceStep] = []
    # Renderable visualizations the tools produced, in the order produced. Typed
    # as dicts here and validated against `app.agent.blocks.Block` in the chat
    # endpoint: keeping the discriminated union out of this module means the REST
    # schemas do not depend on the optional agent package.
    blocks: list[dict] = []
    # Which specialists ran and which failed. Surfaced so a partial answer is
    # visibly partial — the alternative is a reply that silently covered half the
    # request and reads as though it covered all of it.
    lanes: list[str] = []
    lanes_failed: list[str] = []
    elapsed_s: Optional[float] = None
