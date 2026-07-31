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
  path: string; // "keyword" | "semantic"
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
  /** What `score` is (e.g. "cosine") on endpoints that return a bare list
   * with no envelope to carry the basis. */
  score_basis?: string | null;
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
  score_basis?: string | null; // "cosine" | "cosine_adj" | "rrf" | null
  rrf_k?: number | null;
  term_stats: TermStat[];
  offset: number;
  has_more: boolean;
  sort?: string | null;
  ids_resolved?: number | null;
  depth_limit: number;
  depth_reached: boolean;
}

// ---- Promptable segmentation ------------------------------------------------

/** Normalized image-space geometry. A point label follows the SAM convention:
 * 1 keeps the clicked area, 0 removes it from the current mask. */
export interface SegmentPoint {
  x: number;
  y: number;
  label: 0 | 1;
}

export interface SegmentBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface SegmentPrompt {
  points: SegmentPoint[];
  box?: SegmentBox;
}

export interface SegmentRequest extends SegmentPrompt {
  sample_id: number;
  label_name?: string;
  parent_name?: string;
}

export interface RegionSearchRequest extends SegmentBox {
  sample_id: number;
  role: "positive" | "negative";
  text?: string;
  top_k?: number;
  offset?: number;
}

export interface SegmentStatus {
  ready: boolean;
  reason?: string | null;
  model?: string | null;
  revision?: string | null;
}

/** A mask is returned either inline or as a same-origin URL. Keeping both
 * optional lets the service choose the cheaper transport without branching in
 * the editor. */
export interface SegmentResult {
  sample_id: number;
  /** Server-authenticated identity for these exact displayed mask bytes. */
  preview_token: string;
  source_sha256: string;
  mask_sha256: string;
  prompt: {
    points: SegmentPoint[];
    box?: SegmentBox | null;
  };
  mask_data_url: string;
  mask_url?: string | null;
  bbox: SegmentBox;
  area_fraction: number;
  predicted_iou: number;
  model: string;
  model_revision: string;
  mask_width: number;
  mask_height: number;
  label_name?: string | null;
  parent_name?: string | null;
  label_path?: string[];
}

export interface SegmentAnnotation {
  id: number;
  sample_id: number;
  kind: string;
  geometry: SegmentBox;
  label?: string | null;
  created_at: string;
  label_name?: string | null;
  parent_name?: string | null;
  label_path?: string[];
  points: SegmentPoint[];
  box?: SegmentBox | null;
  bbox?: SegmentBox | null;
  mask_data_url?: string | null;
  mask_url?: string | null;
  cutout_url?: string | null;
  artifact_package_url?: string | null;
  mask_width?: number | null;
  mask_height?: number | null;
  model_id?: string | null;
  model_revision?: string | null;
  prompt?: {
    points: SegmentPoint[];
    box?: SegmentBox | null;
  } | null;
  proposal_source?: DetectionProposalSource | null;
  predicted_iou?: number | null;
  provenance?: string | null;
}

export interface SegmentAnnotationCreate extends SegmentPrompt {
  label_name: string;
  parent_name?: string | null;
  preview_token: string;
  mask_data_url: string;
  proposal_token?: string | null;
}

export interface ObjectLabel {
  id: number;
  name: string;
  parent_id?: number | null;
  path: string[];
}

export interface DetectStatus {
  ready: boolean;
  reason?: string | null;
  model?: string | null;
  revision?: string | null;
}

export interface DetectBox extends SegmentBox {
  label: string;
  /** Optional taxonomy resolution supplied by the detector service. The client
   * never guesses a parent from the leaf text. */
  label_name?: string | null;
  parent_name?: string | null;
  label_path?: string[] | null;
  score: number;
  proposal_token: string;
}

export interface DetectResponse {
  sample_id: number;
  model: string;
  revision: string;
  queries: string;
  boxes: DetectBox[];
  note: string;
}

export interface DetectionProposalSource {
  kind: "detector";
  model_id: string;
  model_revision: string;
  queries: string;
  original_label: string;
  proposed_label: string;
  score: number;
  box: SegmentBox;
}

// ---- Local vision inspection -----------------------------------------------

export type VisionTask =
  | "scene"
  | "road_scene"
  | "caption_audit"
  | "ocr"
  | "question";

export interface VisionModelStatus {
  name: string;
  ready: boolean;
  reason?: string | null;
  digest?: string | null;
  family?: string | null;
  parameter_size?: string | null;
  quantization_level?: string | null;
  capabilities: string[];
}

export interface VisionPairCapabilityStatus {
  ready: boolean;
  reason?: string | null;
  provider: "ollama";
  model?: string | null;
  model_digest?: string | null;
  runtime: "ollama";
  runtime_version?: string | null;
  adapter_id: "ollama_sequential_frames";
  adapter_version: number;
  protocol?: "sequential_frames_v1" | null;
}

export interface VisionModelsResponse {
  default_model?: string | null;
  models: VisionModelStatus[];
  pair_comparison: VisionPairCapabilityStatus;
}

export interface VisionVisibleObject {
  name: string;
  attributes: string[];
  location: "foreground" | "midground" | "background" | "throughout" | "unknown";
}

export interface VisionSceneProposal {
  kind: "scene";
  summary: string;
  objects: VisionVisibleObject[];
  setting: "indoor" | "outdoor" | "mixed" | "unknown";
  lighting:
    | "daylight"
    | "low_light"
    | "artificial_light"
    | "backlit"
    | "mixed"
    | "unknown";
  surface_conditions: string[];
  visible_text: string[];
  uncertainties: string[];
  search_terms: string[];
}

export interface VisionRoadActor {
  category: "pedestrian" | "cyclist" | "motorcyclist" | "vehicle" | "animal" | "other";
  description: string;
  location: "road" | "road_edge" | "sidewalk" | "crossing" | "off_road" | "unknown";
}

export interface VisionRoadSceneProposal {
  kind: "road_scene";
  road_scene: boolean;
  summary: string;
  actors: VisionRoadActor[];
  traffic_controls: string[];
  surface_conditions: string[];
  visibility_limitations: string[];
  uncertainties: string[];
  search_terms: string[];
}

export interface VisionCaptionAssessment {
  caption_index: number;
  status: "supported" | "partly_supported" | "unsupported" | "uncertain";
  visible_evidence: string;
}

export interface VisionCaptionAuditProposal {
  kind: "caption_audit";
  assessments: VisionCaptionAssessment[];
  discrepancies: string[];
  uncertainties: string[];
  search_terms: string[];
}

export interface VisionTextRegion {
  text: string;
  location:
    | "top_left"
    | "top"
    | "top_right"
    | "left"
    | "center"
    | "right"
    | "bottom_left"
    | "bottom"
    | "bottom_right"
    | "unknown";
  legibility: "clear" | "partial" | "uncertain";
}

export interface VisionOcrProposal {
  kind: "ocr";
  regions: VisionTextRegion[];
  uncertainties: string[];
  search_terms: string[];
}

export interface VisionQuestionProposal {
  kind: "question";
  answer: string;
  visible_evidence: string[];
  uncertainties: string[];
}

export type VisionProposal =
  | VisionSceneProposal
  | VisionRoadSceneProposal
  | VisionCaptionAuditProposal
  | VisionOcrProposal
  | VisionQuestionProposal;

export interface VisionInspectRequest {
  sample_id: number;
  model: string;
  task: VisionTask;
  question?: string;
}

export interface VisionInspectResponse {
  epistemic_status: "model_proposal";
  sample_id: number;
  filename: string;
  task: VisionTask;
  question?: string | null;
  model: string;
  model_digest: string;
  model_family?: string | null;
  parameter_size?: string | null;
  quantization_level?: string | null;
  prompt_version: number;
  schema_version: number;
  input_sha256: string;
  latency_ms: number;
  source: VisionSource;
  proposal: VisionProposal;
  note: string;
}

export type VisionPairChange =
  | "presence"
  | "count"
  | "position"
  | "pose"
  | "appearance"
  | "background"
  | "text"
  | "other";

export interface VisionPairDifference {
  subject: string;
  change_type: VisionPairChange;
  image_a: string;
  image_b: string;
}

export interface VisionPairProposal {
  kind: "pair_comparison";
  summary: string;
  shared: string[];
  only_a: string[];
  only_b: string[];
  differences: VisionPairDifference[];
  uncertainties: string[];
  grounding_terms_a: string[];
  grounding_terms_b: string[];
}

export interface VisionPairCompareRequest {
  a_sample_id: number;
  b_sample_id: number;
}

export interface VisionSource {
  sample_id: number;
  filename: string;
  split: string;
  image_sha256: string;
  decode_status: "decoded";
  width: number;
  height: number;
  mode: string;
  byte_length: number;
}

export interface VisionPairSource extends VisionSource {}

export interface VisionPairCompareResponse {
  epistemic_status: "model_proposal";
  task: "semantic_difference";
  image_a: VisionPairSource;
  image_b: VisionPairSource;
  model: string;
  model_digest: string;
  model_family?: string | null;
  parameter_size?: string | null;
  quantization_level?: string | null;
  provider: "ollama";
  runtime: "ollama";
  runtime_version: string;
  adapter_id: "ollama_sequential_frames";
  adapter_version: number;
  protocol: "sequential_frames_v1";
  prompt_version: number;
  schema_version: number;
  request_sha256: string;
  proposal_id: string;
  latency_ms: number;
  proposal: VisionPairProposal;
  note: string;
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
  /** Whether the enrichment model is pulled in Ollama right now — distinct
   * from vlm_enriched, which is past work. */
  vlm_ready?: boolean;
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

/** GET /api/albums/{id}/analysis. Split by epistemic status on purpose:
 * `measured` is counted/computed from stored data and works with no model
 * running; `generated` names its model and only ever reports availability
 * here — the summary itself comes from an explicit POST. */
export interface AnalysisSignal {
  kind: "attribute" | "tag" | "vlm_tag";
  grp?: string;
  label: string;
  share: number;
}

export interface AlbumAnalysis {
  album_id: number;
  count: number;
  truncated: boolean;
  measured: {
    score_basis: string;
    coherence: number | null;
    common: AnalysisSignal[];
    different: { grp: string; top: { label: string; share: number }[] }[];
    outliers: { id: number; score: number }[];
    note: string | null;
  };
  generated: {
    available: boolean;
    model: string;
    summary: string | null;
    message: string | null;
  };
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

/** The contaminated ids themselves — the report says how many, this says which. */
export interface LeakageContamination {
  threshold: number;
  held_out_split: string | null;
  total: number;
  ids: number[];
  truncated: boolean;
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
  /** Rows measured on their own query set say how many queries, and why,
   * so the table can too. */
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

export type SearchMode = "hybrid" | "semantic" | "keyword";

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
