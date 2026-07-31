"""API response models."""
from __future__ import annotations

import math
from typing import Annotated, Literal, Optional, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


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
    score_basis: Optional[str] = None   # what `score` is, on endpoints whose
                                        # response is a bare list with no
                                        # envelope to carry the basis
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
    # What `score` means: cosine | cosine_adj | rrf | composed | None.
    # `composed` is a cosine minus a chosen negative-example penalty — only
    # comparable within its own response, never against any other basis.
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


AxisName = Literal["legibility", "rarity", "difficulty", "clutter"]
SearchMode = Literal["semantic", "keyword", "hybrid"]
SearchSort = Literal[
    "legibility_asc",
    "legibility_desc",
    "rarity_asc",
    "rarity_desc",
    "difficulty_asc",
    "difficulty_desc",
    "clutter_asc",
    "clutter_desc",
]


class AxisRange(BaseModel):
    """One validated 0–10 dataset-relative axis interval."""

    model_config = ConfigDict(extra="forbid")

    min: Optional[int] = Field(None, strict=True, ge=0, le=10)
    max: Optional[int] = Field(None, strict=True, ge=0, le=10)


class SearchRequest(BaseModel):
    """POST body for search, carrying what a URL cannot.

    Mirrors the GET parameters exactly; it exists only because an id list of the
    size this tool accepts does not fit in a query string.
    """
    q: str = Field(..., strict=True, min_length=1)
    mode: SearchMode = "hybrid"
    top_k: int = Field(60, strict=True, ge=1, le=200)
    offset: int = Field(0, strict=True, ge=0, le=5000)
    split: Optional[str] = None
    tag: Optional[str] = None
    vlm_tag: Optional[str] = None
    # One facet or several, intersected. A bare string is still accepted so the
    # POST body stays compatible with clients that only ever sent one.
    attr: Optional[Union[str, list[str]]] = None
    sort: Optional[SearchSort] = None
    ids: Optional[str] = None            # raw pasted text, parsed server-side
    axes: dict[AxisName, AxisRange] = Field(default_factory=dict)
    max_agreement: Optional[float] = Field(
        None,
        strict=True,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    # One k-means cluster. Bounded to what SQLite can bind and no narrower —
    # cluster ids are assigned at ingest and carry no sign convention.
    cluster: Optional[int] = Field(
        None,
        strict=True,
        ge=-(2**63),
        le=2**63 - 1,
    )
    # Membership filter, bounded to SQLite's signed 64-bit range like every
    # album id. An id naming no album matches nothing — honest empty, no error.
    album: Optional[int] = Field(
        None,
        strict=True,
        ge=1,
        le=2**63 - 1,
    )


class StatsOverview(BaseModel):
    total_samples: int
    total_captions: int
    splits: dict[str, int]
    avg_caption_length_words: float
    image_size_buckets: dict[str, int]
    embeddings_available: bool
    vlm_enriched: int
    # Retrieval provider truth (additive): which provider is configured as
    # preferred, which one is actually serving, with what model/dimension, and
    # — when they differ — the named reason. The UI must never have to guess.
    embed_preferred: Optional[str] = None
    embed_provider: Optional[str] = None
    embed_model: Optional[str] = None
    embed_dim: Optional[int] = None
    embed_index_ready: bool = False
    embed_fallback_reason: Optional[str] = None
    # Measured 10th-percentile nearest-neighbour cosine of the ACTIVE index
    # (from its manifest); None for the legacy flat layout, where the UI keeps
    # its documented SigLIP-derived default.
    sim_floor: Optional[float] = None
    vlm_model: Optional[str] = None
    # Is the enrichment model pulled in Ollama right now? Distinct from
    # vlm_enriched (past work) — a corpus can be tagged while the model is
    # gone, or untagged while the model waits.
    vlm_ready: bool = False
    chat_model: Optional[str] = None


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
    # Mean cosine *distance* to the 10 nearest images in the 768-D SigLIP space
    # (`analyze._embedding_isolation`, stored per sample). The one signal the map
    # cannot derive from what it already sends: 2-D UMAP distance is not a
    # similarity, so "which images sit in a sparse region of the corpus?" has to
    # be answered from the original space or not at all. `rarity` will not do —
    # it is a percentile that averages this with caption word rarity.
    isolation: Optional[float] = None


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
    # Set on rows measured over a different query set than the main sample.
    # A number without its sample size and its reason invites a comparison it
    # cannot support.
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
    """One rung of the threshold ladder.

    The complete curve keeps the threshold-dependent result explicit instead
    of baking one corpus/model generation's count into the API contract.
    """
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


class LeakageContamination(BaseModel):
    """The contaminated held-out ids, not just how many there are.

    `LeakageReport.contaminated` answers "how bad is it?". This answers "which
    ones?" — the question a researcher acts on, and the one that otherwise sends
    them back to numpy to re-derive a set the tool already had in memory. The
    ids are the gallery's own `?ids=` vocabulary, so the answer arrives as a
    slice that can be opened, exported, tagged or excluded.
    """
    threshold: float
    held_out_split: Optional[Literal["test", "validation"]] = None
    total: int                             # before any cap
    ids: list[int]
    # Never a silently short list: a caller that would page has to be told.
    truncated: bool = False


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


class AlbumSummary(BaseModel):
    """One album on the shelf. `origin` is provenance — 'manual' | 'tag' |
    'agent' — kept so an agent-proposed album stays distinguishable from the
    user's own curation when a review flow arrives."""
    id: int
    name: str
    summary: Optional[str] = None
    category: Optional[str] = None
    origin: str
    item_count: int
    # Thumb of the chosen cover, else the first item; None only when empty.
    cover: Optional[str] = None
    created_at: str                      # ISO-8601, UTC


class AlbumDetail(AlbumSummary):
    notes: Optional[str] = None
    cover_sample_id: Optional[int] = None  # None: cover falls back to first item
    updated_at: str
    items: list[SampleCard] = []         # in album order


class AlbumCreate(BaseModel):
    # Name shares the saved-view ceiling; the prose fields are bounded because
    # an unbounded text field was measured as a real hole once already (the
    # 200,000-character view name).
    name: str = Field(max_length=200)
    summary: Optional[str] = Field(None, max_length=2000)
    category: Optional[str] = Field(None, max_length=2000)
    notes: Optional[str] = Field(None, max_length=2000)


class AlbumUpdate(BaseModel):
    """PATCH body: an absent field means "leave alone", an explicit null means
    "clear" — the endpoint reads `model_dump(exclude_unset=True)` to keep the
    two distinguishable."""
    name: Optional[str] = Field(None, max_length=200)
    summary: Optional[str] = Field(None, max_length=2000)
    category: Optional[str] = Field(None, max_length=2000)
    notes: Optional[str] = Field(None, max_length=2000)
    # Bounded to SQLite's signed 64-bit range: a larger int overflows at bind
    # time and would surface as a 500 instead of a 422.
    cover_sample_id: Optional[int] = Field(None, ge=1, le=2**63 - 1)


# Example ids are few by design: an example is something the user hand-picked,
# not a pasted set — the id-list filter is the tool for sets.
ExampleId = Annotated[int, Field(ge=1, le=2**63 - 1)]


class ComposedSearchRequest(BaseModel):
    """POST body for composed search: text and image examples fused into one
    query, with negative examples pushing results away. At least one of `text`
    / `positive_ids` is required — an all-negative query has no direction."""
    text: Optional[str] = Field(None, max_length=500)
    positive_ids: list[ExampleId] = Field(default_factory=list, max_length=16)
    negative_ids: list[ExampleId] = Field(default_factory=list, max_length=16)
    top_k: int = Field(60, ge=1, le=200)
    offset: int = Field(0, ge=0, le=5000)
    split: Optional[str] = None
    tag: Optional[str] = None
    vlm_tag: Optional[str] = None
    # One facet or several, intersected — same contract as SearchRequest.
    attr: Optional[Union[str, list[str]]] = None
    album: Optional[int] = Field(None, ge=1, le=2**63 - 1)
    max_agreement: Optional[float] = Field(None, ge=0.0, le=1.0, allow_inf_nan=False)
    axes: dict[AxisName, AxisRange] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _needs_a_direction(self):
        # Negative-only is a real direction — "away from this" — and ranks by
        # distance from the excluded examples. Only a fully empty query has
        # nothing to rank by.
        if (not (self.text and self.text.strip()) and not self.positive_ids
                and not self.negative_ids):
            raise ValueError(
                "Provide text, a positive example id, or a negative example id")
        return self


class RegionSearchRequest(BaseModel):
    """A region of an existing sample used as search evidence.

    The server crops the ORIGINAL image itself — the client sends geometry,
    never pixels — so the evidence is reproducible from the request alone.
    Coordinates are normalized 0..1 like annotations; slivers are rejected
    because a 3-pixel crop embeds as noise dressed up as a query.
    """
    sample_id: int = Field(..., ge=1, le=2**63 - 1)
    x: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    y: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    w: float = Field(..., gt=0.0, le=1.0, allow_inf_nan=False)
    h: float = Field(..., gt=0.0, le=1.0, allow_inf_nan=False)
    role: Literal["positive", "negative"] = "positive"
    text: Optional[str] = Field(None, max_length=500)
    top_k: int = Field(24, ge=1, le=100)
    offset: int = Field(0, ge=0, le=100_000)

    @model_validator(mode="after")
    def _fits_and_not_a_sliver(self):
        if self.x + self.w > 1.0001 or self.y + self.h > 1.0001:
            raise ValueError("Region extends outside the image")
        if self.w < 0.02 or self.h < 0.02:
            raise ValueError("Region too small to embed meaningfully "
                             "(minimum 2% of each dimension)")
        return self


class AnnotationSearchRequest(BaseModel):
    """A persisted object mask used as image evidence."""
    annotation_id: int = Field(..., ge=1, le=2**63 - 1)
    top_k: int = Field(24, ge=1, le=100)
    offset: int = Field(0, ge=0, le=100_000)


class ScenarioGroup(BaseModel):
    # Templated from what makes this group DIFFERENT from the rest of the results,
    # e.g. "street · people — 43 images"; "mixed" when nothing distinguishes it.
    label: str
    # The measurement behind every part, in-group and background both stated so the
    # label is checkable: "24/61 caption:people — 39% here vs 12% across the results".
    evidence: str
    count: int                # always equals len(sample_ids)
    # ALL member ids, in ranking order — a group is saved whole (as an album),
    # so this is the full membership, never a preview.
    sample_ids: list[int] = []


class ScenarioResponse(BaseModel):
    groups: list[ScenarioGroup] = []   # never more than 3
    # How the groups were made, stated so nobody mistakes them for a model's
    # judgment: clustering is arithmetic and the labels are counted, not written.
    basis: str
    degraded: bool = False
    message: Optional[str] = None


class ActivityEvent(BaseModel):
    id: int
    kind: str
    payload: dict = {}
    created_at: str                      # ISO-8601, UTC


class ActivityCreate(BaseModel):
    """Client-written snapshot events. `album_*` kinds are deliberately not
    accepted here: those are written by the endpoints that performed the
    action, and a client claiming one would forge server history.
    `tag_approval` is a client kind on purpose — approving an assistant's tag
    proposal happens in the browser, on the user's click, so the client is the
    honest witness for it."""
    kind: Literal["search_snapshot", "image_search", "composed_search",
                  "tag_approval"]
    payload: dict = {}


def _finite01(v) -> bool:
    """A JSON number in [0, 1]. bool is excluded (it is an int in Python);
    NaN/inf are excluded because Python's json parser accepts them even though
    JSON forbids them, and a NaN coordinate would poison every comparison."""
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and 0.0 <= v <= 1.0)


class AnnotationCreate(BaseModel):
    """A region drawn over a sample, in NORMALIZED 0..1 coordinates so the
    geometry survives any rendered size. Rows, never pixels — the source
    image is immutable."""
    kind: Literal["rect", "polygon"]
    geometry: dict
    label: Optional[str] = Field(None, max_length=200)

    @model_validator(mode="after")
    def _valid_geometry(self):
        g = self.geometry
        if self.kind == "rect":
            if set(g.keys()) != {"x", "y", "w", "h"}:
                raise ValueError("rect geometry must be exactly {x, y, w, h}")
            if not all(_finite01(g[k]) for k in ("x", "y", "w", "h")):
                raise ValueError("rect coordinates must be numbers in [0, 1]")
            if g["w"] <= 0 or g["h"] <= 0:
                raise ValueError("rect width and height must be > 0")
            if g["x"] + g["w"] > 1.0001 or g["y"] + g["h"] > 1.0001:
                raise ValueError("rect extends outside the image")
            self.geometry = {k: float(g[k]) for k in ("x", "y", "w", "h")}
        else:
            points = g.get("points")
            if set(g.keys()) != {"points"} or not isinstance(points, list):
                raise ValueError("polygon geometry must be exactly {points: [[x,y], ...]}")
            if not 3 <= len(points) <= 100:
                raise ValueError("polygon needs 3 to 100 points")
            for p in points:
                if not (isinstance(p, list) and len(p) == 2
                        and _finite01(p[0]) and _finite01(p[1])):
                    raise ValueError("each polygon point must be [x, y] in [0, 1]")
            self.geometry = {"points": [[float(x), float(y)] for x, y in points]}
        return self


class AnnotationOut(BaseModel):
    id: int
    sample_id: int
    kind: str
    geometry: dict
    label: Optional[str] = None
    created_at: str
    label_name: Optional[str] = None
    parent_name: Optional[str] = None
    label_path: list[str] = Field(default_factory=list)
    points: list[SegmentPoint] = Field(default_factory=list)
    box: Optional[SegmentBox] = None
    bbox: Optional[SegmentBox] = None
    mask_data_url: Optional[str] = None
    mask_url: Optional[str] = None
    cutout_url: Optional[str] = None
    artifact_package_url: Optional[str] = None
    mask_width: Optional[int] = None
    mask_height: Optional[int] = None
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    prompt: Optional[SegmentPrompt] = None
    proposal_source: Optional[DetectionProposalSource] = None
    predicted_iou: Optional[float] = None


class SegmentPoint(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    y: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    label: Literal[0, 1]

    @field_validator("label", mode="before")
    @classmethod
    def _label_is_an_integer(cls, value):
        if isinstance(value, bool):
            raise ValueError("point label must be 0 (background) or 1 (foreground)")
        return value


class SegmentBox(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    y: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    w: float = Field(..., gt=0.0, le=1.0, allow_inf_nan=False)
    h: float = Field(..., gt=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _fits(self):
        if self.x + self.w > 1.0001 or self.y + self.h > 1.0001:
            raise ValueError("segment box extends outside the image")
        return self


class DetectionProposalSource(BaseModel):
    """Immutable detector evidence that led to an accepted mask.

    The accepted taxonomy is stored separately. That keeps a reviewer relabel
    visible instead of rewriting what the detector originally proposed.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["detector"] = "detector"
    model_id: str = Field(..., min_length=1, max_length=200)
    model_revision: str = Field(
        ...,
        min_length=40,
        max_length=40,
        pattern=r"^[0-9a-f]{40}$",
    )
    queries: str = Field(..., min_length=3, max_length=300)
    original_label: str = Field(..., min_length=1, max_length=100)
    proposed_label: str = Field(..., min_length=1, max_length=100)
    score: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    box: SegmentBox


class SegmentPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points: list[SegmentPoint] = Field(default_factory=list, max_length=16)
    box: Optional[SegmentBox] = None

    @model_validator(mode="after")
    def _has_foreground(self):
        if self.box is None and not any(p.label == 1 for p in self.points):
            raise ValueError("Provide a box or at least one foreground point")
        return self


# AnnotationOut is declared before the segment types because it is also the
# response contract for manual rectangles and polygons. Resolve those typed
# forward references once every segment contract exists.
AnnotationOut.model_rebuild()


def _clean_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = " ".join(value.strip().lower().split())
    if not cleaned:
        raise ValueError("label names cannot be blank")
    return cleaned


class SegmentRequest(SegmentPrompt):
    sample_id: int = Field(..., ge=1, le=2**63 - 1)
    label_name: Optional[str] = Field(None, max_length=100)
    parent_name: Optional[str] = Field(None, max_length=100)

    @field_validator("label_name", "parent_name")
    @classmethod
    def _normalize_labels(cls, value):
        return _clean_label(value)

    @model_validator(mode="after")
    def _valid_label_pair(self):
        if self.parent_name and not self.label_name:
            raise ValueError("parent_name requires label_name")
        if self.parent_name == self.label_name and self.parent_name is not None:
            raise ValueError("an object label cannot be its own parent")
        return self


class SegmentAcceptRequest(SegmentPrompt):
    label_name: str = Field(..., min_length=1, max_length=100)
    parent_name: Optional[str] = Field(None, max_length=100)
    # The preview token authenticates the exact mask bytes the reviewer saw.
    # They remain optional at the schema seam only so the endpoint can return
    # one actionable "generate and review a preview" error for a missing pair;
    # acceptance rejects both a missing pair and a partial pair.
    preview_token: Optional[str] = Field(None, min_length=64, max_length=16_384)
    mask_data_url: Optional[str] = Field(
        None,
        min_length=len("data:image/png;base64,") + 4,
        max_length=4_000_000,
    )
    # Opaque, server-issued evidence returned by POST /api/detect. The server
    # verifies and resolves it; callers never submit model identity/confidence
    # as trusted provenance.
    proposal_token: Optional[str] = Field(None, min_length=64, max_length=4_096)

    @field_validator("label_name", "parent_name")
    @classmethod
    def _normalize_labels(cls, value):
        return _clean_label(value)

    @model_validator(mode="after")
    def _not_its_own_parent(self):
        if self.parent_name == self.label_name:
            raise ValueError("an object label cannot be its own parent")
        return self


class SegmentPreview(BaseModel):
    sample_id: int
    model: str
    model_revision: str
    preview_token: str = Field(..., min_length=64, max_length=16_384)
    source_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    mask_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    prompt: SegmentPrompt
    predicted_iou: float
    bbox: dict
    area_fraction: float
    mask_width: int
    mask_height: int
    mask_data_url: str
    label_name: Optional[str] = None
    parent_name: Optional[str] = None
    label_path: list[str] = Field(default_factory=list)


class ModelCapabilityStatus(BaseModel):
    ready: bool
    reason: str | None = None
    model: str
    revision: str | None = None
    measured: str


class DetectBoxOut(SegmentBox):
    label: str
    score: float
    label_name: str | None = None
    parent_name: str | None = None
    label_path: list[str] = Field(default_factory=list)
    proposal_token: str = Field(..., min_length=64, max_length=4_096)
    # What a second, independent model sees in this crop. `score` above is the
    # detector's phrase-alignment, which has no way to say "absent" — these say
    # whether the phrase actually beat the alternatives. None = not checked.
    verified: bool | None = None
    verified_score: float | None = None
    best_alternative: str | None = None
    alternative_score: float | None = None


class DetectResponse(BaseModel):
    sample_id: int
    model: str
    revision: str
    queries: str
    boxes: list[DetectBoxOut]
    note: str


class ObjectLabelOut(BaseModel):
    id: int
    name: str
    parent_id: Optional[int] = None
    path: list[str] = Field(default_factory=list)


# ---- Local vision inspection ------------------------------------------------

VisionTask = Literal["scene", "road_scene", "caption_audit", "ocr", "question"]
VisionSetting = Literal["indoor", "outdoor", "mixed", "unknown"]
VisionLighting = Literal[
    "daylight",
    "low_light",
    "artificial_light",
    "backlit",
    "mixed",
    "unknown",
]


class VisionModelStatus(BaseModel):
    """One configured local Ollama artifact and its measured capabilities."""

    model_config = ConfigDict(extra="forbid")

    name: str
    ready: bool
    reason: str | None = None
    digest: str | None = None
    family: str | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None
    capabilities: list[str] = Field(default_factory=list)


class VisionPairCapabilityStatus(BaseModel):
    """The one server-selected pair adapter, separate from model discovery."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    reason: str | None = None
    provider: Literal["ollama"] = "ollama"
    model: str | None = None
    model_digest: str | None = None
    runtime: Literal["ollama"] = "ollama"
    runtime_version: str | None = None
    adapter_id: Literal["ollama_sequential_frames"] = "ollama_sequential_frames"
    adapter_version: int = 1
    protocol: Literal["sequential_frames_v1"] | None = None


class VisionModelsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_model: str | None = None
    models: list[VisionModelStatus]
    pair_comparison: VisionPairCapabilityStatus


class VisionInspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: int = Field(..., strict=True, ge=1, le=2**63 - 1)
    model: str = Field(..., min_length=1, max_length=200)
    task: VisionTask
    question: str | None = Field(None, max_length=500)

    @model_validator(mode="after")
    def _question_matches_task(self) -> "VisionInspectRequest":
        question = " ".join(self.question.split()) if self.question else None
        if self.task == "question" and not question:
            raise ValueError("question is required for the question task")
        if self.task != "question" and question:
            raise ValueError("question is only accepted for the question task")
        self.question = question
        return self


class VisionVisibleObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=100)
    attributes: list[str] = Field(default_factory=list, max_length=8)
    location: Literal[
        "foreground",
        "midground",
        "background",
        "throughout",
        "unknown",
    ] = "unknown"


class VisionSceneProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["scene"] = "scene"
    summary: str = Field(..., min_length=1, max_length=800)
    objects: list[VisionVisibleObject] = Field(default_factory=list, max_length=24)
    setting: VisionSetting
    lighting: VisionLighting
    surface_conditions: list[str] = Field(default_factory=list, max_length=12)
    visible_text: list[str] = Field(default_factory=list, max_length=20)
    uncertainties: list[str] = Field(default_factory=list, max_length=12)
    search_terms: list[str] = Field(default_factory=list, max_length=16)


class VisionRoadActor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "pedestrian",
        "cyclist",
        "motorcyclist",
        "vehicle",
        "animal",
        "other",
    ]
    description: str = Field(..., min_length=1, max_length=240)
    location: Literal[
        "road",
        "road_edge",
        "sidewalk",
        "crossing",
        "off_road",
        "unknown",
    ] = "unknown"


class VisionRoadSceneProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["road_scene"] = "road_scene"
    road_scene: bool
    summary: str = Field(..., min_length=1, max_length=800)
    actors: list[VisionRoadActor] = Field(default_factory=list, max_length=24)
    traffic_controls: list[str] = Field(default_factory=list, max_length=16)
    surface_conditions: list[str] = Field(default_factory=list, max_length=12)
    visibility_limitations: list[str] = Field(default_factory=list, max_length=12)
    uncertainties: list[str] = Field(default_factory=list, max_length=12)
    search_terms: list[str] = Field(default_factory=list, max_length=16)


class VisionCaptionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caption_index: int = Field(..., strict=True, ge=0, le=100)
    status: Literal["supported", "partly_supported", "unsupported", "uncertain"]
    visible_evidence: str = Field(..., min_length=1, max_length=500)


class VisionCaptionAuditProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["caption_audit"] = "caption_audit"
    assessments: list[VisionCaptionAssessment] = Field(
        default_factory=list,
        max_length=100,
    )
    discrepancies: list[str] = Field(default_factory=list, max_length=20)
    uncertainties: list[str] = Field(default_factory=list, max_length=12)
    search_terms: list[str] = Field(default_factory=list, max_length=16)


class VisionTextRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=500)
    location: Literal[
        "top_left",
        "top",
        "top_right",
        "left",
        "center",
        "right",
        "bottom_left",
        "bottom",
        "bottom_right",
        "unknown",
    ] = "unknown"
    legibility: Literal["clear", "partial", "uncertain"]


class VisionOcrProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["ocr"] = "ocr"
    regions: list[VisionTextRegion] = Field(default_factory=list, max_length=40)
    uncertainties: list[str] = Field(default_factory=list, max_length=12)
    search_terms: list[str] = Field(default_factory=list, max_length=16)


class VisionQuestionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["question"] = "question"
    answer: str = Field(..., min_length=1, max_length=1_200)
    visible_evidence: list[str] = Field(default_factory=list, max_length=16)
    uncertainties: list[str] = Field(default_factory=list, max_length=12)


VisionPairChange = Literal[
    "presence",
    "count",
    "position",
    "pose",
    "appearance",
    "background",
    "text",
    "other",
]


def _strip_pair_text(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("pair proposal text must not be blank")
    return value


VisionPairSubject = Annotated[
    str,
    Field(min_length=1, max_length=160),
    AfterValidator(_strip_pair_text),
]
VisionPairEvidence = Annotated[
    str,
    Field(min_length=1, max_length=400),
    AfterValidator(_strip_pair_text),
]
VisionGroundingTerm = Annotated[
    str,
    Field(min_length=1, max_length=160),
    AfterValidator(_strip_pair_text),
]
VisionPairSummary = Annotated[
    str,
    Field(min_length=1, max_length=1_000),
    AfterValidator(_strip_pair_text),
]


class VisionPairDifference(BaseModel):
    """One concrete, directly visible difference between frame A and frame B."""

    model_config = ConfigDict(extra="forbid")

    subject: VisionPairSubject
    change_type: VisionPairChange
    image_a: VisionPairEvidence = Field(
        ...,
        min_length=1,
        max_length=400,
        description=(
            "Concrete visible state in image A; never merely the label 'Frame A'."
        ),
    )
    image_b: VisionPairEvidence = Field(
        ...,
        min_length=1,
        max_length=400,
        description=(
            "Concrete visible state in image B; never merely the label 'Frame B'."
        ),
    )


class VisionPairProposal(BaseModel):
    """A reviewable semantic comparison, never a pixel or corruption metric."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["pair_comparison"] = "pair_comparison"
    summary: VisionPairSummary
    shared: list[VisionPairEvidence] = Field(..., max_length=16)
    only_a: list[VisionPairEvidence] = Field(..., max_length=16)
    only_b: list[VisionPairEvidence] = Field(..., max_length=16)
    differences: list[VisionPairDifference] = Field(
        ...,
        max_length=24,
    )
    uncertainties: list[VisionPairEvidence] = Field(
        ...,
        max_length=12,
    )
    grounding_terms_a: list[VisionGroundingTerm] = Field(
        ...,
        max_length=12,
    )
    grounding_terms_b: list[VisionGroundingTerm] = Field(
        ...,
        max_length=12,
    )

    @field_validator(
        "shared",
        "only_a",
        "only_b",
        "uncertainties",
        "grounding_terms_a",
        "grounding_terms_b",
    )
    @classmethod
    def _phrases_are_nonblank_and_unique(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        for value in values:
            key = " ".join(value.split()).casefold()
            if not key:
                raise ValueError("pair evidence phrases must not be blank")
            if key in seen:
                raise ValueError("pair evidence phrases must be unique within a field")
            seen.add(key)
        return values

    @model_validator(mode="after")
    def _contains_reviewable_evidence(self) -> "VisionPairProposal":
        if not any(
            (
                self.shared,
                self.only_a,
                self.only_b,
                self.differences,
                self.uncertainties,
                self.grounding_terms_a,
                self.grounding_terms_b,
            )
        ):
            raise ValueError(
                "pair proposals require evidence beyond the summary"
            )
        return self


class VisionPairCompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a_sample_id: int = Field(..., strict=True, ge=1, le=2**63 - 1)
    b_sample_id: int = Field(..., strict=True, ge=1, le=2**63 - 1)

    @model_validator(mode="after")
    def _samples_are_distinct(self) -> "VisionPairCompareRequest":
        if self.a_sample_id == self.b_sample_id:
            raise ValueError("pair comparison requires two distinct samples")
        return self


class VisionSource(BaseModel):
    """One source whose encoded bytes passed a forced local pixel decode."""

    model_config = ConfigDict(extra="forbid")

    sample_id: int
    filename: str
    split: str
    image_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    decode_status: Literal["decoded"] = "decoded"
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    mode: str = Field(..., min_length=1, max_length=32)
    byte_length: int = Field(..., ge=1)


class VisionPairSource(VisionSource):
    """Compatibility name for an ordered source in a pair proposal."""


class VisionPairCompareResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epistemic_status: Literal["model_proposal"] = "model_proposal"
    task: Literal["semantic_difference"] = "semantic_difference"
    image_a: VisionPairSource
    image_b: VisionPairSource
    model: str
    model_digest: str
    model_family: str | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None
    provider: Literal["ollama"] = "ollama"
    runtime: Literal["ollama"] = "ollama"
    runtime_version: str
    adapter_id: Literal["ollama_sequential_frames"]
    adapter_version: int
    protocol: Literal["sequential_frames_v1"]
    prompt_version: int
    schema_version: int
    request_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    proposal_id: str = Field(..., pattern=r"^vp_[0-9a-f]{32}$")
    latency_ms: int = Field(..., ge=0)
    proposal: VisionPairProposal
    note: str


VisionProposal = Annotated[
    Union[
        VisionSceneProposal,
        VisionRoadSceneProposal,
        VisionCaptionAuditProposal,
        VisionOcrProposal,
        VisionQuestionProposal,
    ],
    Field(discriminator="kind"),
]


class VisionInspectResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epistemic_status: Literal["model_proposal"] = "model_proposal"
    sample_id: int
    filename: str
    task: VisionTask
    # Present only for the focused-question task. Keeping the normalized input
    # beside the answer makes the exported proposal independently readable;
    # input_sha256 remains the integrity binding for image bytes + task input.
    question: str | None = Field(None, max_length=500)
    model: str
    model_digest: str
    model_family: str | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None
    prompt_version: int
    schema_version: int
    input_sha256: str
    latency_ms: int = Field(..., ge=0)
    source: VisionSource
    proposal: VisionProposal
    note: str

    @model_validator(mode="after")
    def _question_matches_task(self) -> VisionInspectResponse:
        if self.task == "question" and not self.question:
            raise ValueError("question-task responses must preserve the question")
        if self.task != "question" and self.question is not None:
            raise ValueError("question is only valid for the question task")
        return self


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
