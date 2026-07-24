import type {
  AttributeGroup, CaptionStats, ChatMessage, ChatResponse, ChatStatus,
  DuplicatePair, EvalResponse, MapPoint, QASummary, SampleCard, SampleDetail,
  SampleList, SearchMode, SearchResponse, StatsOverview, SuspectCaption, TagInfo,
} from "./types";

type Params = Record<string, string | number | undefined>;

async function get<T>(path: string, params?: Params, signal?: AbortSignal): Promise<T> {
  const url = new URL(`/api${path}`, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== "") url.searchParams.set(k, String(v));
    }
  }
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

async function send<T>(path: string, method: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  listSamples: (params: Params, signal?: AbortSignal) =>
    get<SampleList>("/samples", params, signal),
  getSample: (id: number | string) => get<SampleDetail>(`/samples/${id}`),
  similar: (id: number | string) => get<SampleCard[]>(`/samples/${id}/similar`),
  search: (q: string, mode: SearchMode, filters: Params, signal?: AbortSignal) =>
    get<SearchResponse>("/search", { q, mode, ...filters }, signal),

  overview: () => get<StatsOverview>("/stats/overview"),
  captionStats: () => get<CaptionStats>("/stats/captions"),
  duplicates: () => get<DuplicatePair[]>("/stats/duplicates"),
  map: () => get<MapPoint[]>("/map"),

  tags: () => get<TagInfo[]>("/tags"),
  vlmTags: () => get<TagInfo[]>("/vlm-tags"),
  addTag: (id: number, name: string) =>
    send<{ ok: boolean }>(`/samples/${id}/tags`, "POST", { name }),
  removeTag: (id: number, name: string) =>
    send<{ ok: boolean }>(`/samples/${id}/tags/${encodeURIComponent(name)}`, "DELETE"),
  bulkTag: (sample_ids: number[], name: string) =>
    send<{ ok: boolean; tag: string; tagged: number }>("/tags/bulk", "POST", { sample_ids, name }),

  qaSummary: () => get<QASummary>("/qa/summary"),
  suspectCaptions: (params?: Params) => get<SuspectCaption[]>("/qa/captions", params),
  inconsistentSamples: () => get<SuspectCaption[]>("/qa/consistency"),
  coverage: () => get<AttributeGroup[]>("/attributes/coverage"),
  evalRetrieval: (sampleSize = 1000) =>
    get<EvalResponse>("/eval/retrieval", { sample_size: sampleSize }),

  chat: (messages: ChatMessage[]) => send<ChatResponse>("/chat", "POST", { messages }),
  chatStatus: () => get<ChatStatus>("/chat/status"),
};
