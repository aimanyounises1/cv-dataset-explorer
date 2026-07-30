import type {
  ActivityEvent, AlbumAnalysis,
  AttributeGroup, CaptionStats, ChatMessage, ChatResponse, ChatStatus,
  AlbumDetail, AlbumSummary, ComposedQuery, ScenarioResponse,
  DescribeResponse, LeakageContamination, LeakageReport,
  DuplicatePair, EvalResponse, MapPoint, QASelection, QASummary, SampleCard,
  DetectResponse, DetectStatus, ObjectLabel, RegionSearchRequest,
  SegmentAnnotation, SegmentAnnotationCreate, SegmentResult,
  SegmentRequest, SegmentStatus,
  SampleDetail, SampleList, SearchMode, SearchResponse, StatsOverview,
  SuspectCaption, TagInfo, VisionInspectRequest, VisionInspectResponse,
  VisionModelsResponse, VisionPairCompareRequest, VisionPairCompareResponse,
} from "./types";

type Params = Record<string, string | number | string[] | undefined>;

/** One entry of FastAPI's 422 envelope: `{"detail": [{loc, msg, ...}]}`. */
type ValidationItem = { loc?: unknown[]; msg?: string };

/** The sentence inside FastAPI's error envelope.
 *
 * `detail` is a string for a raised `HTTPException` ("Sample not found") and a
 * list of per-field objects for a request that failed validation. Both are
 * turned into one line; anything else falls back to the caller's status text,
 * because inventing a sentence for a shape we do not recognise would be a
 * claim the payload does not support.
 */
function detailSentence(detail: unknown): string | null {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const parts = (detail as ValidationItem[])
      .map((d) => {
        const msg = typeof d?.msg === "string" ? d.msg : null;
        if (!msg) return null;
        // `loc` is ["path", "sample_id"] or ["query", "mode"] — the last
        // element is the field the reader can actually change.
        const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : null;
        return field ? `${String(field)}: ${msg}` : msg;
      })
      .filter((s): s is string => s !== null);
    if (parts.length > 0) return parts.join("; ");
  }
  return null;
}

/** What went wrong, as something a reader can act on — with the payload kept.
 *
 * Every failed request used to reach the screen as `String(e)`, which on a
 * validation failure is the entire `{"detail":[{"type":"int_parsing",…}]}`
 * document printed into a red box. The status and the raw body are still the
 * useful clue when something is genuinely unexpected, so neither is thrown
 * away: `message` is the sentence, `status` is matchable without parsing text,
 * and `body` holds the payload for the surfaces that show it behind a
 * disclosure.
 *
 * `toString()` returns the sentence alone rather than `Error`'s
 * `"<name>: <message>"`, so the call sites that render `String(e)` print the
 * sentence and not a class name.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly body: string;
  readonly detail: unknown;

  constructor(status: number, body: string) {
    let detail: unknown;
    try {
      detail = (JSON.parse(body) as { detail?: unknown }).detail;
    } catch {
      /* not JSON — `body` is still the whole truth we have */
    }
    super(detailSentence(detail)
      ?? (body.trim() || `The request failed with status ${status}.`));
    this.name = "ApiError";
    this.status = status;
    this.body = body;
    this.detail = detail;
  }

  override toString(): string { return this.message; }
}

/** True when `e` is the API refusing with this status — asked by the surfaces
 * that treat "not found" or "already exists" as an outcome rather than a fault.
 * A helper rather than `e.message.startsWith("404")`: the message is now a
 * sentence, and a sentence is not a status. */
export const isStatus = (e: unknown, status: number): boolean =>
  e instanceof ApiError && e.status === status;

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
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.json() as Promise<T>;
}

async function send<T>(
  path: string, method: string, body?: unknown, signal?: AbortSignal,
): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method, signal,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.json() as Promise<T>;
}

export const api = {
  listSamples: (params: Params, signal?: AbortSignal) =>
    get<SampleList>("/samples", params, signal),
  getSample: (id: number | string) => get<SampleDetail>(`/samples/${id}`),
  similar: (id: number | string, topK?: number) =>
    get<SampleCard[]>(`/samples/${id}/similar`, { top_k: topK }),
  search: (q: string, mode: SearchMode, filters: Params, signal?: AbortSignal) =>
    get<SearchResponse>("/search", { q, mode, ...filters }, signal),
  /** Composed retrieval: text pulls, reference images pull, negatives push.
   * Scores are only comparable within one response (basis "composed"). */
  composedSearch: (body: ComposedQuery, signal?: AbortSignal) =>
    send<SearchResponse>("/search/composed", "POST", body, signal),
  /** At most three explainable groups over the current ranking, on demand. */
  scenarioGroups: (body: ComposedQuery, signal?: AbortSignal) =>
    send<ScenarioResponse>("/search/scenarios", "POST", body, signal),

  /** Image-to-image retrieval. Raw bytes, not multipart: one file needs no
   * form envelope, and the server carries no form parser. */
  searchByImage: async (file: Blob, topK = 24): Promise<SampleCard[]> => {
    const res = await fetch(`/api/search/by-image?top_k=${topK}`, {
      method: "POST",
      headers: { "Content-Type": file.type || "application/octet-stream" },
      body: file,
    });
    if (!res.ok) throw new ApiError(res.status, await res.text());
    return res.json() as Promise<SampleCard[]>;
  },

  segmentStatus: (signal?: AbortSignal) =>
    get<SegmentStatus>("/segment/status", undefined, signal),
  segment: (body: SegmentRequest, signal?: AbortSignal) =>
    send<SegmentResult>("/segment", "POST", body, signal),
  listSegmentAnnotations: (sampleId: number, signal?: AbortSignal) =>
    get<SegmentAnnotation[]>(
      `/samples/${sampleId}/segment-annotations`, undefined, signal),
  saveSegmentAnnotation: (
    sampleId: number, body: SegmentAnnotationCreate, signal?: AbortSignal,
  ) => send<SegmentAnnotation>(
    `/samples/${sampleId}/segment-annotations`, "POST", body, signal),
  deleteSegmentAnnotation: (annotationId: number, signal?: AbortSignal) =>
    send<{ ok: boolean }>(
      `/segment-annotations/${annotationId}`, "DELETE", undefined, signal),
  searchByAnnotation: (
    annotationId: number, topK = 24, signal?: AbortSignal,
  ) => send<SearchResponse>(
    "/search/by-annotation", "POST",
    { annotation_id: annotationId, top_k: topK }, signal),
  searchByRegion: (body: RegionSearchRequest, signal?: AbortSignal) =>
    send<SearchResponse>("/search/by-region", "POST", body, signal),
  detectStatus: (signal?: AbortSignal) =>
    get<DetectStatus>("/detect/status", undefined, signal),
  detectRegions: (
    sampleId: number, queries: string, signal?: AbortSignal,
  ) => send<DetectResponse>(
    "/detect", "POST", { sample_id: sampleId, queries }, signal),
  objectLabels: (signal?: AbortSignal) =>
    get<ObjectLabel[]>("/object-labels", undefined, signal),

  /** Read-only local VLM proposals. The endpoint validates Ollama's live
   * vision capability and returns exact model-digest provenance. */
  visionModels: (signal?: AbortSignal) =>
    get<VisionModelsResponse>("/vision/models", undefined, signal),
  inspectVision: (body: VisionInspectRequest, signal?: AbortSignal) =>
    send<VisionInspectResponse>("/vision/inspect", "POST", body, signal),
  compareVision: (body: VisionPairCompareRequest, signal?: AbortSignal) =>
    send<VisionPairCompareResponse>("/vision/compare", "POST", body, signal),

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
  leakageContaminated: (threshold: number, signal?: AbortSignal) =>
    get<LeakageContamination>("/stats/leakage/contaminated", { threshold }, signal),

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
