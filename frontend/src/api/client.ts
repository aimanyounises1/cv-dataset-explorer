import type {
  ActivityEvent, AlbumAnalysis,
  AttributeGroup, CaptionStats, ChatMessage, ChatResponse, ChatStatus,
  AlbumDetail, AlbumSummary, ComposedQuery, ScenarioResponse,
  DescribeResponse, LeakageReport,
  DuplicatePair, EvalResponse, MapPoint, QASelection, QASummary, SampleCard,
  SampleDetail, SampleList, SearchMode, SearchResponse, StatsOverview,
  SuspectCaption, TagInfo,
} from "./types";

type Params = Record<string, string | number | string[] | undefined>;

async function get<T>(path: string, params?: Params, signal?: AbortSignal): Promise<T> {
  const url = new URL(`/api${path}`, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v === undefined || v === "") continue;
      // An array becomes repeated occurrences (`?attr=a&attr=b`), which is what
      // the API intersects. `String(v)` would have sent the single value "a,b".
      if (Array.isArray(v)) for (const one of v) url.searchParams.append(k, String(one));
      else url.searchParams.set(k, String(v));
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
  /** Composed retrieval: text pulls, reference images pull, negatives push.
   * Scores are only comparable within one response (basis "composed"). */
  composedSearch: (body: ComposedQuery, signal?: AbortSignal) =>
    fetch("/api/search/composed", {
      method: "POST", signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(async (r) => {
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      return r.json() as Promise<SearchResponse>;
    }),
  /** At most three explainable groups over the current ranking, on demand. */
  scenarioGroups: (body: ComposedQuery, signal?: AbortSignal) =>
    fetch("/api/search/scenarios", {
      method: "POST", signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(async (r) => {
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      return r.json() as Promise<ScenarioResponse>;
    }),

  /** Image-to-image retrieval. Raw bytes, not multipart: one file needs no
   * form envelope, and the server carries no form parser. */
  searchByImage: async (file: Blob, topK = 24): Promise<SampleCard[]> => {
    const res = await fetch(`/api/search/by-image?top_k=${topK}`, {
      method: "POST",
      headers: { "Content-Type": file.type || "application/octet-stream" },
      body: file,
    });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    return res.json() as Promise<SampleCard[]>;
  },

  overview: () => get<StatsOverview>("/stats/overview"),
  /** The workspace trail, newest first. A bare array by contract — see
   * ActivityEvent for what the payload does and does not promise. */
  listActivity: (limit = 100) => get<ActivityEvent[]>("/activity", { limit }),
  /** Client-witnessed events only — the server allowlists the kinds. */
  recordActivity: (kind: string, payload: Record<string, unknown>) =>
    send<ActivityEvent>("/activity", "POST", { kind, payload }),
  captionStats: () => get<CaptionStats>("/stats/captions"),
  duplicates: () => get<DuplicatePair[]>("/stats/duplicates"),
  map: () => get<MapPoint[]>("/map"),

  /** Albums are ordered, first-class collections — not tags. The shelf and
   * the tray both mutate through these and then announce it on the
   * cvde-albums-changed window event, so every album surface refetches. */
  listAlbums: () => get<AlbumSummary[]>("/albums"),
  createAlbum: (name: string) => send<AlbumDetail>("/albums", "POST", { name }),
  albumDetail: (id: number) => get<AlbumDetail>(`/albums/${id}`),
  addToAlbum: (id: number, sample_ids: number[]) =>
    send<{ ok: boolean; added: number }>(`/albums/${id}/items`, "POST", { sample_ids }),
  removeFromAlbum: (id: number, sampleId: number) =>
    send<{ ok: boolean }>(`/albums/${id}/items/${sampleId}`, "DELETE"),
  updateAlbum: (id: number, body: Partial<Pick<AlbumDetail,
      "name" | "summary" | "category" | "notes" | "cover_sample_id">>) =>
    send<AlbumDetail>(`/albums/${id}`, "PATCH", body),
  deleteAlbum: (id: number) => send<{ ok: boolean }>(`/albums/${id}`, "DELETE"),
  /** Measured signals always; the generated half only reports availability —
   * drafting a summary is a separate, explicit POST. */
  albumAnalysis: (id: number) => get<AlbumAnalysis>(`/albums/${id}/analysis`),
  generateAlbumSummary: (id: number) =>
    send<{ summary: string; model: string; based_on: string }>(
      `/albums/${id}/analysis/summary`, "POST", {}),

  tags: () => get<TagInfo[]>("/tags"),
  vlmTags: () => get<TagInfo[]>("/vlm-tags"),
  addTag: (id: number, name: string) =>
    send<{ ok: boolean }>(`/samples/${id}/tags`, "POST", { name }),
  removeTag: (id: number, name: string) =>
    send<{ ok: boolean }>(`/samples/${id}/tags/${encodeURIComponent(name)}`, "DELETE"),
  bulkTag: (sample_ids: number[], name: string) =>
    send<{ ok: boolean; tag: string; tagged: number }>("/tags/bulk", "POST", { sample_ids, name }),

  /** The inverse of every other call: given a selection, what is it made of. */
  describe: (params: Params, signal?: AbortSignal) =>
    get<DescribeResponse>("/describe", params, signal),

  /** Held-out images that have a near-duplicate in training. */
  leakage: (threshold: number, signal?: AbortSignal) =>
    get<LeakageReport>("/stats/leakage", { threshold }, signal),

  qaSummary: () => get<QASummary>("/qa/summary"),
  qaSelection: (maxAgreement: number, signal?: AbortSignal) =>
    get<QASelection>("/qa/selection", { max_agreement: maxAgreement }, signal),
  suspectCaptions: (params?: Params) => get<SuspectCaption[]>("/qa/captions", params),
  // (params carries limit / split / max_agreement — the review threshold)
  inconsistentSamples: () => get<SuspectCaption[]>("/qa/consistency"),
  coverage: () => get<AttributeGroup[]>("/attributes/coverage"),
  evalRetrieval: (sampleSize = 1000) =>
    get<EvalResponse>("/eval/retrieval", { sample_size: sampleSize }),

  chat: (messages: ChatMessage[]) => send<ChatResponse>("/chat", "POST", { messages }),
  chatStatus: () => get<ChatStatus>("/chat/status"),
};
