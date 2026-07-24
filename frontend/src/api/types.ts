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
}

export interface SearchResponse {
  items: SampleCard[];
  mode_used: string;
  degraded: boolean;
  message?: string | null;
}

export interface StatsOverview {
  total_samples: number;
  total_captions: number;
  splits: Record<string, number>;
  avg_caption_length_words: number;
  image_size_buckets: Record<string, number>;
  embeddings_available: boolean;
  vlm_enriched: number;
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
}

export interface AttributeLabel {
  label: string;
  count: number;
  fraction: number;
}

export interface AttributeGroup {
  grp: string;
  labels: AttributeLabel[];
}

export interface EvalModeResult {
  mode: string;
  recall_at: Record<string, number>;
}

export interface EvalResponse {
  available: boolean;
  message?: string | null;
  sample_size: number;
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
}

export interface ChatResponse {
  reply: string;
  samples: SampleCard[];
  trace: ChatTraceStep[];
}

export interface ChatStatus {
  available: boolean;
  model: string;
  reason?: string;
}

export type SearchMode = "hybrid" | "semantic" | "keyword";
