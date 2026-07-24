export interface SampleCard {
  id: number;
  filename: string;
  split: string;
  width?: number | null;
  height?: number | null;
  thumb_url: string;
  caption?: string | null;
  score?: number | null;
}

export interface SampleList {
  items: SampleCard[];
  total: number;
  page: number;
  per_page: number;
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
  captions: string[];
  tags: string[];
  vlm_tags: string[];
  cluster?: number | null;
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

export type SearchMode = "hybrid" | "semantic" | "keyword";
