import type { Block } from "./blocks";

/** The four difficulty axes, as 0-10 percentile buckets over this dataset.
 * Ranks, not measurements — a 7 means "harder than ~70% of this corpus". */
/** Server-side cap on a pasted id list (backend: deps.MAX_ID_LIST). Kept in
 * sync by hand; the backend rejects anything over it regardless. */
export const MAX_ID_LIST = 60000;

export const AXES = ["legibility", "rarity", "difficulty", "clutter"] as const;
export type Axis = (typeof AXES)[number];

export interface AxisScores {
  legibility?: number | null;
  rarity?: number | null;
  difficulty?: number | null;
  clutter?: number | null;
  /** Per axis: the raw component values, plus a "why" string naming the
   * components that are themselves in the hard tail. Templated server-side
   * from measured percentiles — never model-generated. */
  detail: Record<string, Record<string, number | string>>;
}

export interface MatchPath {
  path: string; // "keyword" | "semantic" | "boosted"
  rank: number; // 1-based rank within that path
}

export interface SampleCard {
  id: number;
  filename: string;
  split: string;
  width?: number | null;
  height?: number | null;
  thumb_url: string;
  caption?: string | null;
  score?: number | null;
  match_caption?: string | null;
  matched_terms?: string[] | null;
  match_paths?: MatchPath[] | null;
  axes?: AxisScores | null;
}

export interface SampleList {
  items: SampleCard[];
  total: number;
  page: number;
  per_page: number;
}

export interface CaptionOut {
  text: string;
  agreement?: number | null;
}

export interface SampleDetail {
  id: number;
  filename: string;
  split: string;
  width?: number | null;
  height?: number | null;
  filesize?: number | null;
  image_url: string;
  thumb_url: string;
  captions: CaptionOut[];
  tags: string[];
  vlm_tags: string[];
  attributes: Record<string, string>;
  cluster?: number | null;
  caption_consistency?: number | null;
  axes?: AxisScores | null;
}

export interface TermStat {
  term: string;
  images: number;
  fraction: number;
  common: boolean;
}

export interface SearchResponse {
  items: SampleCard[];
  mode_used: string;
  degraded: boolean;
  message?: string | null;
  score_basis?: string | null; // "cosine" | "cosine_adj" | "rrf" | "prism_ll" | null
  rrf_k?: number | null;
  term_stats: TermStat[];
  offset: number;
  has_more: boolean;
  sort?: string | null;
  ids_resolved?: number | null;
  depth_limit: number;
  depth_reached: boolean;
}

export interface StatsOverview {
  total_samples: number;
  total_captions: number;
  splits: Record<string, number>;
  avg_caption_length_words: number;
  image_size_buckets: Record<string, number>;
  embeddings_available: boolean;
  vlm_enriched: number;
  /** Model provenance. All optional: a running backend may predate these
   * fields, and the status card must degrade to the older flags above rather
   * than crash — on a status surface, a broken card is worse than a quiet one.
   * `embed_provider` is the ACTIVE provider and may differ from
   * `embed_preferred` (fallback); null means retrieval is keyword-only. */
  embed_preferred?: string | null;
  embed_provider?: string | null;
  embed_model?: string | null;
  embed_dim?: number | null;
  embed_index_ready?: boolean;
  embed_fallback_reason?: string | null;
  sim_floor?: number | null;
  vlm_model?: string | null;
  chat_model?: string | null;
}

/** One append-only workspace event from /api/activity. The endpoint returns a
 * bare array (not an envelope), newest first by id — ids are monotonic where
 * same-tick timestamps can collide. Payload is opaque JSON with two writers
 * (server album_* mutations, client search snapshots), so read its fields
 * defensively; nothing in it is guaranteed. */
export interface ActivityEvent {
  id: number;
  kind: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface CaptionStats {
  length_histogram: { bucket: string; count: number }[];
  top_words: { word: string; count: number }[];
}

export interface DuplicatePair {
  a: SampleCard;
  b: SampleCard;
  similarity: number;
}

export interface MapPoint {
  id: number;
  x: number;
  y: number;
  cluster: number;
  thumb_url: string;
  split: string;
  agreement?: number | null;
  legibility?: number | null;
  rarity?: number | null;
  difficulty?: number | null;
  clutter?: number | null;
}

export interface TagInfo {
  name: string;
  count: number;
}

export interface SuspectCaption {
  caption: string;
  agreement: number;
  sibling_mean?: number | null;
  sample: SampleCard;
}

export interface QASummary {
  available: boolean;
  scored_captions: number;
  mean_agreement?: number | null;
  histogram: { lo: number; hi: number; count: number }[];
  min_agreement?: number | null;
  max_agreement?: number | null;
}

/** Exact size of a review threshold's selection, in both units: the histogram
 * can only give a bin-rounded count, and images ≠ captions because one image
 * can hold several weak captions. */
export interface QASelection {
  max_agreement: number;
  captions: number;
  samples: number;
}

/** One attribute label, and how over- or under-represented it is in a selection.
 * `count` travels with `lift` deliberately: a 6x multiplier over three images and
 * over three hundred are different findings. */
export interface FacetLift {
  group: string;
  label: string;
  count: number;
  set_share: number;
  corpus_share: number;
  lift: number;
  z: number;
  drill: string;
}

export interface SetAxis {
  axis: Axis;
  set_mean: number;
  corpus_mean: number;
  delta: number;
}

/** What characterises a selection, relative to the whole dataset. */
export interface DescribeResponse {
  set_size: number;
  corpus_size: number;
  filtered: boolean;
  message?: string | null;
  over: FacetLift[];
  under: FacetLift[];
  axes: SetAxis[];
  splits: Record<string, number>;
  clusters: { cluster: number; count: number }[];
  mean_agreement?: number | null;
  corpus_mean_agreement?: number | null;
  min_facet_count: number;
  min_abs_z: number;
}

export interface LeakagePoint {
  threshold: number;
  pairs: number;
  cross_split: number;
  contaminated: number;
}

export interface LeakagePair {
  a_id: number; b_id: number; score: number;
  a_split: string; b_split: string;
  a_thumb: string; b_thumb: string;
  cross_split: boolean;
}

/** Held-out images with a training near-duplicate. Reported as a curve, because
 * "near-duplicate" is a threshold on a cosine and the answer moves with it. */
export interface LeakageReport {
  threshold: number;
  floor: number;
  pairs: number;
  cross_split_pairs: number;
  by_split_pair: Record<string, number>;
  contaminated: number;
  held_out_total: number;
  contaminated_fraction: number;
  curve: LeakagePoint[];
  examples: LeakagePair[];
  default_threshold: number;
  caveat: string;
}

export interface AttributeLabel {
  label: string;
  count: number;
  fraction: number;
}

export interface AttributeGroup {
  grp: string;
  labels: AttributeLabel[];
  labelled: number;
  abstained: number;
  mean_confidence?: number | null;
}

export interface EvalModeResult {
  mode: string;
  recall_at: Record<string, number>;
  mrr: number;
  median_rank?: number | null;
  mean_candidates: number;
  empty_query_rate: number;
  /** Rows measured on their own query set (the PRISM rows use test-split
   * queries only) say how many queries, and why, so the table can too. */
  queries?: number | null;
  note?: string | null;
}

export interface EvalResponse {
  available: boolean;
  message?: string | null;
  sample_size: number;
  pool_size: number;
  depth: number;
  mean_query_words: number;
  results: EvalModeResult[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatTraceStep {
  agent: string;
  tool: string;
  input: string;
  lane_seconds?: number | null;
}

export interface ChatResponse {
  reply: string;
  samples: SampleCard[];
  trace: ChatTraceStep[];
  /** Interactive visualizations the specialists produced, in the order produced. */
  blocks: Block[];
  /** Which specialists ran, and which died. A partial answer must look partial. */
  lanes: string[];
  lanes_failed: string[];
  elapsed_s?: number | null;
}

export interface ChatStatus {
  available: boolean;
  model: string;
  reason?: string;
  specialists?: string[];
}

export type SearchMode = "hybrid" | "semantic" | "keyword" | "boosted";

// ---- Albums ----------------------------------------------------------------

/** One album on the shelf. `origin` is provenance — "manual" | "tag" |
 * "agent" — so an agent-proposed album stays distinguishable from the user's
 * own curation when the review flow arrives. */
export interface AlbumSummary {
  id: number;
  name: string;
  summary?: string | null;
  category?: string | null;
  origin: string;
  item_count: number;
  cover: string | null;                // thumb of cover else first item; null when empty
  created_at: string;
}

export interface AlbumDetail extends AlbumSummary {
  notes?: string | null;
  cover_sample_id?: number | null;
  updated_at: string;
  items: SampleCard[];                 // in album order
}

/** One explainable scenario group: a k-means cluster over the top of a
 * ranking, labelled from measured attributes — templated, never generated. */
export interface ScenarioGroup {
  label: string;
  evidence: string;
  count: number;
  sample_ids: number[];
}
export interface ScenarioResponse {
  groups: ScenarioGroup[];
  basis: string;
}

/** Composed search body: text and reference images pull, negatives push. */
export interface ComposedQuery {
  text?: string;
  positive_ids?: number[];
  negative_ids?: number[];
  top_k?: number;
  offset?: number;
  split?: string;
  tag?: string;
  vlm_tag?: string;
  attr?: string[];
  album?: number;
  max_agreement?: number;
}
