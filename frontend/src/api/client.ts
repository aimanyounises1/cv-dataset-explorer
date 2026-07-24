import type {
  CaptionStats, DuplicatePair, MapPoint, SampleCard, SampleDetail, SampleList,
  SearchMode, SearchResponse, StatsOverview, TagInfo,
} from "./types";

type Params = Record<string, string | number | undefined>;

async function get<T>(path: string, params?: Params): Promise<T> {
  const url = new URL(`/api${path}`, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== "") url.searchParams.set(k, String(v));
    }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  listSamples: (params: Params) => get<SampleList>("/samples", params),
  getSample: (id: number | string) => get<SampleDetail>(`/samples/${id}`),
  similar: (id: number | string) => get<SampleCard[]>(`/samples/${id}/similar`),
  search: (q: string, mode: SearchMode, filters: Params) =>
    get<SearchResponse>("/search", { q, mode, ...filters }),
  overview: () => get<StatsOverview>("/stats/overview"),
  captionStats: () => get<CaptionStats>("/stats/captions"),
  duplicates: () => get<DuplicatePair[]>("/stats/duplicates"),
  map: () => get<MapPoint[]>("/map"),
  tags: () => get<TagInfo[]>("/tags"),
  vlmTags: () => get<TagInfo[]>("/vlm-tags"),

  addTag: async (id: number, name: string): Promise<void> => {
    const res = await fetch(`/api/samples/${id}/tags`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!res.ok) throw new Error(await res.text());
  },
  removeTag: async (id: number, name: string): Promise<void> => {
    const res = await fetch(`/api/samples/${id}/tags/${encodeURIComponent(name)}`, {
      method: "DELETE",
    });
    if (!res.ok) throw new Error(await res.text());
  },
};
