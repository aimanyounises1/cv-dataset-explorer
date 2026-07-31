import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { AXES, Axis } from "../api/types";
import { AXIS_META } from "../components/AxisFilters";
import type { ActiveFilterChip } from "../components/ActiveFilters";

/**
 * The current selection, derived from the URL.
 *
 * This app's thesis is that every view produces a *set*: a lasso on the map, a
 * brush on the quality histogram, a pasted id list, a search, a saved view. The
 * set already lives in the query string, which is what makes all of those the
 * same object — shareable, bookmarkable, and survivable across a page reload.
 *
 * So this hook is a **read**, not a store. There is deliberately no context, no
 * reducer and no client-side cache *for the selection*: the moment a set lives
 * in two places the interface can show one thing while the URL says another, and
 * every hand-off in the product stops being trustworthy.
 *
 * The rule is about reproducibility, not about banning browser storage. State a
 * colleague must be able to reproduce from a pasted link lives in the URL;
 * ephemeral per-session preference deliberately does not, and sits in
 * session/localStorage instead — the scan order behind `useResultOrder`, the
 * gallery's card density, the map's colour mode, the assistant transcript. None
 * of those change which samples you are looking at, and putting them in the URL
 * would make every shared link carry one person's view settings.
 *
 * It exists because the selection now has a permanent home in the right rail,
 * which renders on every route — so the chip-building and export-link logic that
 * used to be private to GalleryPage has to be reachable from anywhere.
 */

/** Everything that narrows the corpus. `q`/`mode` are query, not membership.
 * Exported so other export paths (the command palette) serialize from this
 * one vocabulary — a hand-copied key list over there is how `cluster` once
 * went missing from "Download current view" and 8,000 rows shipped as 552. */
export const SCALARS = ["split", "tag", "vlm_tag", "ids", "max_agreement",
                        "cluster", "album"] as const;

/** Parameters that may appear more than once, intersected server-side.
 *
 * `attr` left this list for a long time and the cost was silent: the URL could
 * hold `?attr=time_of_day:night&attr=environment:indoors`, `URLSearchParams.get`
 * returned only the first, and the gallery filtered by one facet while the
 * address bar promised two. Nothing errored — the user simply got 392 images
 * where they had asked for 99. */
export const REPEATABLE = ["attr"] as const;

/** The ranked-slice depth every export path has always requested. */
const EXPORT_RANKED_TOP_K = 500;

/** Serialize the selection vocabulary found in a raw query string into
 * `/api/export` parameters. The ONE builder behind every export path — the
 * rail's pills and the palette's "Download current view" — so the two can
 * never drift: a hand-copied list is how `cluster` once went missing and
 * 8,000 rows shipped labelled as a slice. Reads the raw string rather than
 * hook state so it serves any route without the hook mounted; only
 * recognised keys are copied, so a route's own params (`view`, `a`/`b` on
 * compare) are ignored, not leaked. */
export function buildExportParams(
  search: string, format: "csv" | "jsonl" | "json",
): URLSearchParams {
  const params = new URLSearchParams(search);
  const out = new URLSearchParams();
  const q = params.get("q");
  if (q) {
    out.set("q", q);
    out.set("mode", params.get("mode") || "hybrid");
    out.set("top_k", String(EXPORT_RANKED_TOP_K));
  }
  for (const k of SCALARS) {
    const v = params.get(k);
    if (v) out.set(k, v);
  }
  // `getAll`: an intersection of repeatable facets must survive into the
  // export, or the file is a wider set than the screen it was pressed on.
  for (const k of REPEATABLE) {
    for (const v of params.getAll(k)) if (v) out.append(k, v);
  }
  for (const axis of AXES) {
    for (const suffix of ["_min", "_max"] as const) {
      const v = params.get(`${axis}${suffix}`);
      if (v) out.set(`${axis}${suffix}`, v);
    }
  }
  out.set("format", format);
  return out;
}

export interface Selection {
  /** Query parameters describing the set, ready to spread into an API call. */
  params: Record<string, string | number | string[] | undefined>;
  /** One removable chip per active constraint. */
  chips: ActiveFilterChip[];
  /** Any membership constraint. Drives "what is in this selection?", which
   * compares a filtered set against the corpus and has no meaning for a query. */
  active: boolean;
  /** Anything that produces a result set worth taking away — a query counts,
   * even though it adds no chip. Without this a plain search had no export. */
  exportable: boolean;
  /** Where this selection appears to have come from, for the rail's header. */
  origin: string | null;
  /** The free-text query, which narrows results but is not a filter. */
  query: string;
  mode: string;
  exportHref: (format: "csv" | "jsonl" | "json") => string;
  remove: (key: string) => void;
  clearAll: () => void;
  /** Write filter parameters. Empty string deletes. Any control that narrows the
   * corpus goes through here, from whichever pane it happens to live in.
   * Pushes a history entry so Back undoes the change; pass `{replace: true}`
   * from controls that emit a stream of values rather than one decision. */
  set: (updates: Record<string, string | string[]>,
        options?: { replace?: boolean }) => void;
  /** The four axis ranges, in the shape `AxisFilters` expects. */
  axisRanges: Partial<Record<Axis, [number | null, number | null]>>;
}

export function useSelection(): Selection {
  const [searchParams, setSearchParams] = useSearchParams();

  const get = useCallback(
    (k: string) => searchParams.get(k) ?? "", [searchParams]);

  const query = get("q");
  const mode = get("mode") || "hybrid";

  const axisRanges = useMemo(() => {
    const out: Partial<Record<Axis, [number | null, number | null]>> = {};
    for (const a of AXES) {
      const lo = searchParams.get(`${a}_min`);
      const hi = searchParams.get(`${a}_max`);
      if (lo != null || hi != null) {
        out[a] = [lo == null ? null : Number(lo), hi == null ? null : Number(hi)];
      }
    }
    return out;
  }, [searchParams]);

  const params = useMemo(() => {
    const p: Record<string, string | number | string[] | undefined> = {};
    for (const k of SCALARS) {
      const v = searchParams.get(k);
      if (v) p[k] = v;
    }
    for (const k of REPEATABLE) {
      const all = searchParams.getAll(k).filter(Boolean);
      if (all.length) p[k] = all;
    }
    for (const [axis, range] of Object.entries(axisRanges)) {
      const [lo, hi] = range as [number | null, number | null];
      if (lo != null) p[`${axis}_min`] = lo;
      if (hi != null) p[`${axis}_max`] = hi;
    }
    return p;
  }, [searchParams, axisRanges]);

  const chips = useMemo(() => {
    const out: ActiveFilterChip[] = [];
    const push = (key: string, label: string, value: string) =>
      out.push({ key, label, value });
    if (get("split")) push("split", "Split", get("split"));
    if (get("tag")) push("tag", "Tag", get("tag"));
    if (get("vlm_tag")) push("vlm_tag", "VLM tag", get("vlm_tag"));
    for (const v of searchParams.getAll("attr").filter(Boolean)) {
      // `attr:<value>` rather than `attr`, so removing one facet of an
      // intersection leaves the others standing.
      out.push({ key: `attr:${v}`, label: "Attribute", value: v });
    }
    for (const a of AXES) {
      const r = axisRanges[a];
      if (r) push(a, AXIS_META[a].label, `${r[0] ?? 0}–${r[1] ?? 10}`);
    }
    if (get("ids")) {
      const n = new Set(get("ids").replace(/,/g, "\n").split("\n")
        .map((s) => s.trim()).filter(Boolean)).size;
      push("ids", "Id list", String(n));
    }
    if (get("max_agreement")) {
      push("max_agreement", "Caption agreement",
           `≤ ${Number(get("max_agreement")).toFixed(3)}`);
    }
    if (get("cluster")) push("cluster", "Cluster", get("cluster"));
    // The chip shows the id; the shelf alongside shows which album that is,
    // by name and highlight — a sync name lookup here would mean a fetch
    // inside a memo.
    if (get("album")) push("album", "Album", `#${get("album")}`);
    return out;
  }, [get, axisRanges, searchParams]);

  /** Which view most likely produced this set. Used only as a label — it is
   * inferred from the parameters rather than tracked, because tracking it would
   * mean writing state the URL cannot carry, and a shared link would lose it. */
  const origin = useMemo(() => {
    if (get("album")) return "Album";
    if (get("max_agreement")) return "Caption quality";
    if (get("ids")) return "Id list or map lasso";
    if (get("cluster")) return "Embedding map";
    if (Object.keys(axisRanges).length > 0) return "Difficulty axes";
    if (searchParams.getAll("attr").length) return "Attribute slice";
    if (chips.length > 0) return "Filters";
    return null;
  }, [get, axisRanges, chips.length, searchParams]);

  /** Write filter parameters.
   *
   * Discrete choices PUSH a history entry, so Back undoes the filter a person
   * just applied. Every write used to replace, which meant two filter clicks
   * left `history.length` at 2 and the first Back press left the application
   * entirely — measured: `/` → `?split=train` → `?split=validation` → Back →
   * about:blank. For a product whose thesis is that the URL is the state, a
   * back button that cannot walk that state was the sharpest contradiction in
   * it.
   *
   * `replace` is for controls that emit a stream rather than a decision — a
   * slider dragged through twenty values, a debounced query still being typed,
   * a "load more" that deepens the same view. Those must not leave twenty
   * entries between the reader and where they came from.
   */
  const setParams = useCallback((
    updates: Record<string, string | string[]>,
    options?: { replace?: boolean },
  ) => {
    setSearchParams((prev) => {
      const p = new URLSearchParams(prev);
      for (const [k, v] of Object.entries(updates)) {
        // An array replaces every occurrence of that key at once, which is how a
        // repeatable facet is edited: writing them one at a time through `set`
        // would drop the siblings it is meant to preserve.
        if (Array.isArray(v)) {
          p.delete(k);
          for (const one of v.filter(Boolean)) p.append(k, one);
        } else if (v) p.set(k, v);
        else p.delete(k);
      }
      return p;
    }, { replace: options?.replace ?? false });
  }, [setSearchParams]);

  const remove = useCallback((key: string) => {
    const colon = key.indexOf(":");
    if (colon > 0 && (REPEATABLE as readonly string[]).includes(key.slice(0, colon))) {
      const [k, value] = [key.slice(0, colon), key.slice(colon + 1)];
      setSearchParams((prev) => {
        const p = new URLSearchParams(prev);
        const keep = p.getAll(k).filter((v) => v !== value);
        p.delete(k);
        for (const v of keep) p.append(k, v);
        p.delete("page");
        return p;
        // Pushes, like the other two branches below: dropping one facet of an
        // intersection is a decision, and Back should put it back.
      }, { replace: false });
    } else if ((AXES as readonly string[]).includes(key)) {
      setParams({ [`${key}_min`]: "", [`${key}_max`]: "", page: "" });
    } else {
      setParams({ [key]: "", page: "" });
    }
  }, [setParams, setSearchParams]);

  const clearAll = useCallback(() => {
    const updates: Record<string, string> = { page: "" };
    for (const k of [...SCALARS, ...REPEATABLE]) updates[k] = "";
    for (const a of AXES) { updates[`${a}_min`] = ""; updates[`${a}_max`] = ""; }
    setParams(updates);   // the query and mode survive: those are not filters
  }, [setParams]);

  const exportHref = useCallback(
    (format: "csv" | "jsonl" | "json") =>
      `/api/export?${buildExportParams(searchParams.toString(), format).toString()}`,
    [searchParams]);

  return {
    params, chips, active: chips.length > 0,
    exportable: chips.length > 0 || Boolean(query), origin, query, mode,
    exportHref, remove, clearAll, set: setParams, axisRanges,
  };
}
