import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, api } from "../api/client";
import { AXES } from "../api/types";
import type { MapPoint } from "../api/types";
import MapWorkingSet, { ISOLATED_N } from "../components/MapWorkingSet";
import ScatterPlot from "../components/ScatterPlot";
import { Listbox } from "../components/controls";
import { useSelection } from "../hooks/useSelection";
import {
  COLOR_MODES, ColorMode, NO_DATA, Ramp, buildColorScale, formatDomainValue,
} from "../lib/mapColor";
import { SURFACE, rampStops, sequential } from "../lib/viz";
import "../styles/map.css";

const COLOR_KEY = "cvde-map-color";

/** Where the near-duplicate shortcut cuts. Matches the leakage report's own
 * default so the two surfaces are talking about the same pairs. */
const DUP_THRESHOLD = 0.9;

/** Points outside the working set stay on the canvas as context, in the same
 * grey the panel borders use — the region a lasso caught should read as "lit",
 * not as "the only thing that exists". */
const CONTEXT_GREY = SURFACE.borderStrong;

/** The map payload plus the neighbourhood measurement `/api/map` carries. The
 * shared `MapPoint` type is what the endpoint returns; `isolation` is optional
 * there too, so this alias exists only to name what the page reads. */
type Point = MapPoint & { isolation?: number | null };

/** Colour by measured neighbourhood density, alongside the shared modes.
 *
 * It is defined here rather than in `lib/mapColor` because it is the one
 * dimension only this page has: the value comes from the 768-D embedding
 * space, and the map is the only view that can show where in the corpus it
 * runs thin. */
const NEIGHBOUR_MODE = "neighbour_sim";
type MapColorMode = ColorMode | typeof NEIGHBOUR_MODE;

/** Mean cosine to an image's 10 nearest neighbours in SigLIP space.
 *
 * Stored as a distance (`isolation`), reported here as the similarity, because
 * every other cosine in this product reads "higher is closer". */
function neighbourSim(p: Point): number | null {
  return typeof p.isolation === "number" ? 1 - p.isolation : null;
}

/** The sentence a reader needs out of a failed request. `ApiError` has already
 * read it out of FastAPI's envelope — so a 503 that names the command which
 * enables a capability reads as that sentence, and this only has to cover the
 * non-API rejections (an aborted fetch, a network failure) that reach the same
 * handlers. */
function apiMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function quantile(sortedAsc: number[], q: number): number {
  if (sortedAsc.length === 0) return 0;
  const i = Math.min(sortedAsc.length - 1,
                     Math.max(0, Math.round(q * (sortedAsc.length - 1))));
  return sortedAsc[i];
}

/** Filters the map can answer from the payload it already has: every one of
 * these is an equality or a range over a column that travels with the point.
 * Everything else — tag and album membership, an attribute intersection, the
 * "any caption at or below" agreement cut — is resolved by the server through
 * the same export route the gallery hands its sets to, so this view can never
 * disagree with that one about what a filter means. */
const SERVER_RESOLVED = ["tag", "vlm_tag", "album", "attr", "max_agreement"] as const;

interface Scope {
  status: "loading" | "ready" | "error";
  ids: Set<number>;
  message: string | null;
}

/** Where a working set came from. Carried with the ids because "212 images"
 * means nothing without it, and the three sources answer different questions. */
type SetSource = "lasso" | "isolated" | "duplicates";

interface WorkingSet {
  ids: number[];
  source: SetSource;
  /** One line naming exactly what was measured to produce these ids. */
  label: string;
}

export default function MapPage() {
  const [points, setPoints] = useState<Point[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [workingSet, setWorkingSet] = useState<WorkingSet | null>(null);
  const [dropped, setDropped] = useState<Set<number>>(new Set());
  const [marked, setMarked] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [lastTag, setLastTag] = useState<string | null>(null);
  const [dupNote, setDupNote] = useState<string | null>(null);
  const [scope, setScope] = useState<Scope | null>(null);
  const [colorMode, setColorMode] = useState<MapColorMode>(
    () => (localStorage.getItem(COLOR_KEY) as MapColorMode) || "difficulty");

  const selection = useSelection();
  const navigate = useNavigate();

  useEffect(() => {
    try { localStorage.setItem(COLOR_KEY, colorMode); } catch { /* non-essential */ }
  }, [colorMode]);

  useEffect(() => {
    api.map().then(setPoints).catch((e) => setError(String(e)));
  }, []);

  // ------------------------------------------------------ the active slice --

  const { params, query, exportHref } = selection;
  const needsServer = useMemo(
    () => Boolean(query) || SERVER_RESOLVED.some((k) => k in params),
    [params, query]);

  // A membership filter the payload cannot express is resolved once, by the
  // export route, which is `run_search` plus the same WHERE builder the gallery
  // uses. `scores=false` because only the ids are wanted here.
  useEffect(() => {
    if (!needsServer) { setScope(null); return; }
    const ctrl = new AbortController();
    setScope({ status: "loading", ids: new Set(), message: null });
    fetch(`${exportHref("json")}&scores=false`, { signal: ctrl.signal })
      .then(async (r) => {
        if (!r.ok) throw new ApiError(r.status, await r.text());
        return r.json() as Promise<{ samples: { id: number }[] }>;
      })
      .then((data) => setScope({
        status: "ready",
        ids: new Set(data.samples.map((s) => s.id)),
        message: null,
      }))
      .catch((e: unknown) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        // A selection this page cannot resolve must say so: silently drawing
        // all 8,000 points under an active filter is the map lying about
        // which corpus it is showing.
        setScope({ status: "error", ids: new Set(), message: apiMessage(e) });
      });
    return () => ctrl.abort();
  }, [needsServer, exportHref]);

  const visible = useMemo<Point[]>(() => {
    const all = points ?? [];
    if (needsServer) return scope?.status === "ready"
      ? all.filter((p) => scope.ids.has(p.id))
      : all;                              // while it resolves, show the corpus
    const split = params.split ? String(params.split) : null;
    const cluster = params.cluster != null ? Number(params.cluster) : null;
    const idList = params.ids
      ? new Set(String(params.ids).replace(/,/g, "\n").split("\n")
          .map((s) => Number(s.trim())).filter((n) => Number.isFinite(n)))
      : null;
    const ranges = AXES
      .map((a) => ({
        axis: a,
        lo: params[`${a}_min`] != null ? Number(params[`${a}_min`]) : null,
        hi: params[`${a}_max`] != null ? Number(params[`${a}_max`]) : null,
      }))
      .filter((r) => r.lo != null || r.hi != null);
    if (!split && cluster == null && !idList && ranges.length === 0) return all;
    return all.filter((p) => {
      if (split && p.split !== split) return false;
      if (cluster != null && p.cluster !== cluster) return false;
      if (idList && !idList.has(p.id)) return false;
      for (const r of ranges) {
        const v = p[r.axis];
        // A sample with no score on that axis is out, exactly as a SQL range
        // predicate drops a NULL.
        if (typeof v !== "number") return false;
        if (r.lo != null && v < r.lo) return false;
        if (r.hi != null && v > r.hi) return false;
      }
      return true;
    });
  }, [points, needsServer, scope, params]);

  const visibleIds = useMemo(() => new Set(visible.map((p) => p.id)), [visible]);
  const byId = useMemo(
    () => new Map((points ?? []).map((p) => [p.id, p])), [points]);
  /** Whether the URL narrows the corpus at all — asked of the selection, not of
   * the point count, so a filter that happens to match everything still says
   * which filter is in force. */
  const scoped = selection.active || Boolean(query);

  // Counted from the data, not configured: k is the backend's choice and this
  // page should report what arrived, not what was asked for.
  const clusterCount = useMemo(
    () => new Set(visible.map((p) => p.cluster)).size, [visible]);

  // ------------------------------------------------- measured neighbourhood --

  /** The neighbour-similarity distribution of what is on screen. Recomputed per
   * slice on purpose: "the sparse tail of *this* selection" is the question a
   * filtered map is being asked. */
  const nnStats = useMemo(() => {
    const values = visible
      .map(neighbourSim)
      .filter((v): v is number => v != null)
      .sort((a, b) => a - b);
    if (values.length === 0) return null;
    return {
      n: values.length,
      p5: quantile(values, 0.05),
      median: quantile(values, 0.5),
      min: values[0],
    };
  }, [visible]);

  /** A colour mode is only offered while the values behind it exist. Without
   * the measurement the neighbour ramp would paint a flat grey map under a
   * legend reading 0 to 0, which looks like a broken chart rather than like a
   * missing capability — the panel below says which command supplies it. */
  const nnAvailable = nnStats != null;
  const activeMode: MapColorMode =
    colorMode === NEIGHBOUR_MODE && !nnAvailable ? "difficulty" : colorMode;

  /* The same rule as a list of options: the neighbour mode joins only once its
     measurement is loaded, so the control never offers a dimension the map
     cannot paint. */
  const colorModeOptions = useMemo(
    () => [
      ...COLOR_MODES.map((m) => ({ value: m.value as string, label: m.label })),
      ...(nnAvailable
        ? [{ value: NEIGHBOUR_MODE, label: "Neighbour similarity (10-NN cosine)" }]
        : []),
    ],
    [nnAvailable]);

  // Rebuilt only when the data or the dimension changes: the canvas redraws on
  // every pan and zoom, so the scale must not be reconstructed per frame.
  const { colorOf: dimensionColor, ramp } = useMemo<{
    colorOf: (p: Point) => string; ramp: Ramp;
  }>(() => {
    if (activeMode !== NEIGHBOUR_MODE) {
      return buildColorScale(visible, activeMode as ColorMode);
    }
    const values = visible.map(neighbourSim).filter((v): v is number => v != null);
    if (values.length === 0) {
      return { colorOf: () => NO_DATA, ramp: { kind: "quantity", domain: [0, 0] } };
    }
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const span = hi - lo || 1;
    return {
      colorOf: (p: Point) => {
        const v = neighbourSim(p);
        return v == null ? NO_DATA : sequential((v - lo) / span);
      },
      ramp: { kind: "quantity", domain: [lo, hi] },
    };
  }, [visible, activeMode]);

  // ------------------------------------------------------- the working set --

  const kept = useMemo(
    () => (workingSet?.ids ?? []).filter((id) => !dropped.has(id)), [workingSet, dropped]);
  const keptSet = useMemo(() => new Set(kept), [kept]);
  const ringIds = useMemo(
    () => (marked == null ? undefined : new Set([marked])), [marked]);

  /** Two states, and they are different pictures: with no working set the map
   * is the corpus coloured by one dimension; with one, everything outside it
   * drops back to context grey so the set is visible as a *shape*. */
  const colorOf = useCallback((p: MapPoint) => (
    keptSet.size === 0 || keptSet.has(p.id) ? dimensionColor(p) : CONTEXT_GREY
  ), [keptSet, dimensionColor]);

  const adopt = useCallback((ids: number[], source: SetSource, label: string) => {
    setToast(null);
    setDropped(new Set());
    setMarked(null);
    if (ids.length === 0) {
      setWorkingSet(null);
      setToast(source === "lasso"
        ? "That lasso caught no points — drag over a denser region."
        : "Nothing matched.");
      return;
    }
    setWorkingSet({ ids, source, label });
  }, []);

  const selectIsolated = () => {
    const ranked = visible
      .filter((p) => neighbourSim(p) != null)
      .sort((a, b) => (neighbourSim(a) as number) - (neighbourSim(b) as number))
      .slice(0, ISOLATED_N);
    if (ranked.length === 0) {
      setToast("Neighbour similarity is not available for these images — run "
               + "`python -m app.ingest` to compute embeddings and the axis pass.");
      return;
    }
    const worst = neighbourSim(ranked[ranked.length - 1]) as number;
    adopt(ranked.map((p) => p.id), "isolated",
          `the ${ranked.length} most isolated images ${scoped ? "in this slice" : "in the corpus"}`
          + ` — mean cosine to their 10 nearest neighbours at or below ${worst.toFixed(3)}`);
  };

  const loadDuplicates = async () => {
    setBusy(true);
    try {
      const rep = await api.leakage(DUP_THRESHOLD);
      const ids = [...new Set(rep.examples.flatMap((e) => [e.a_id, e.b_id]))]
        .filter((id) => visibleIds.has(id));
      setDupNote(
        `${rep.pairs.toLocaleString()} pairs at cosine ≥ ${rep.threshold} across the `
        + `corpus, ${rep.cross_split_pairs.toLocaleString()} of them cross-split; `
        + `${rep.contaminated.toLocaleString()} of ${rep.held_out_total.toLocaleString()} `
        + `held-out images (${(rep.contaminated_fraction * 100).toFixed(2)}%) have a `
        + "train near-duplicate.");
      adopt(ids, "duplicates",
            `the ${rep.examples.length} strongest near-duplicate pairs at cosine ≥ `
            + `${rep.threshold} (${ids.length} images${scoped ? " in this slice" : ""}) `
            + `of ${rep.pairs.toLocaleString()} such pairs`);
    } catch (e) {
      // The report needs embeddings; its 503 names the command that builds them.
      setDupNote(null);
      setToast(apiMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const toggleDrop = useCallback((id: number) => {
    setDropped((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  // The two ends of the linked view: the grid marks an id, the canvas rings it.
  // Leaving clears the mark only when it is still the id that thumbnail set, so
  // a fast move between two cards cannot un-ring the one now under the cursor.
  const highlight = useCallback((id: number) => setMarked(id), []);
  const unhighlight = useCallback(
    (id: number) => setMarked((m) => (m === id ? null : m)), []);

  const restoreDropped = useCallback(() => setDropped(new Set()), []);
  const clearSet = useCallback(() => {
    setWorkingSet(null);
    setDropped(new Set());
  }, []);

  /** Tags everything still in the set. Resolves true when the tag stuck, which
   * is when — and only when — the panel clears its input. */
  const applyTag = async (name: string): Promise<boolean> => {
    if (kept.length === 0) return false;
    setBusy(true);
    try {
      const res = await api.bulkTag(kept, name);
      setToast(`Tagged ${kept.length} samples as “${res.tag}”`);
      setLastTag(res.tag);
      setWorkingSet(null);
      setDropped(new Set());
      return true;
    } catch (err) {
      setToast(apiMessage(err) || "Bulk tag failed");
      setLastTag(null);
      return false;
    } finally {
      setBusy(false);
    }
  };

  if (error) return <div className="error">{error}</div>;
  if (points === null) return <div className="loading">Loading embedding map…</div>;
  if (points.length === 0) {
    return (
      <div className="empty">
        No embedding projection available yet — run <code>python -m app.ingest</code> to
        compute SigLIP embeddings and the UMAP layout.
      </div>
    );
  }

  return (
    <div className={`map-page${workingSet ? " has-set" : ""}`}>
      <h1 className="section-title" style={{ marginTop: 0 }}>Embedding map</h1>
      {/* Colour is the map's only free channel — position is fixed by UMAP — so
          which dimension it carries is the most consequential control here. */}
      <div className="map-toolbar">
        <span className="density-label">Colour by</span>
        {/* The rail's own listbox, not a native <select>: this was the last
            browser-voiced control in the app, and it sits on the one setting
            that decides what the map is saying. */}
        <Listbox
          inline
          label="Colour points by"
          trigger={colorModeOptions.find((o) => o.value === activeMode)?.label ?? "Colour by"}
          selected={activeMode}
          options={colorModeOptions}
          onPick={(v) => setColorMode(v as MapColorMode)}
        />

        <div className="map-legend">
          {ramp.kind === "quantity" && ramp.domain ? (
            <div>
              <div className="legend-ramp" aria-hidden="true">
                {rampStops(sequential, 12).map((c, i) => (
                  <span key={i} style={{ background: c }} />
                ))}
              </div>
              <div className="legend-ends">
                <span>{formatDomainValue(ramp.domain[0])}</span>
                <span>{formatDomainValue(ramp.domain[1])}</span>
              </div>
            </div>
          ) : (
            <div className="legend-swatches">
              {/* A nominal legend names groups the URL can already select on —
                  `cluster` and `split` are both in useSelection's SCALARS and
                  this page filters by both. Until now they were labels only:
                  the map could paint twelve clusters and count them, and the
                  only way to look at one was to type ?cluster=N in the address
                  bar. Each swatch is the control for its own group, and the
                  one in force says so and turns itself off. */}
              {(ramp.categories ?? []).slice(0, 12).map((c) => {
                const facet = COLOR_MODES.find((m) => m.value === activeMode)?.facet;
                if (!facet) {
                  return (
                    <span className="legend-swatch" key={c.label}>
                      <i className="legend-dot" style={{ background: c.color }} />
                      {c.label}
                    </span>
                  );
                }
                const value = facet === "cluster" ? c.label.replace(/^#/, "") : c.label;
                const on = String(params[facet] ?? "") === value;
                return (
                  <button type="button" key={c.label} aria-pressed={on}
                          className={`legend-swatch legend-pick${on ? " on" : ""}`}
                          title={on
                            ? `Showing only ${c.label} — click to show every group again`
                            : `Show only ${c.label}`}
                          onClick={() => selection.set(
                            { [facet]: on ? "" : value, page: "" })}>
                    <i className="legend-dot" style={{ background: c.color }} />
                    {c.label}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* How to read and use the picture, before the picture: one line per
          mark. The full honesty panel stays below the canvas — this strip is
          the part a first-time reader needs before their first click. */}
      <div className="map-key" aria-label="How to read the map">
        <span className="map-key-item">
          <i className="legend-dot" style={{ background: "var(--accent)" }} /> one
          image — click opens its sample page
        </span>
        <span className="map-key-item">
          <span className="map-key-kbd">shift+drag</span> lasso a region into the
          working set below
        </span>
        <span className="map-key-item">
          the working set stays lit; everything else drops to grey — hover a
          thumbnail to ring its point
        </span>
      </div>

      <div className="controls" style={{ marginBottom: 10 }}>
        <div className="meta-line" style={{ marginBottom: 0 }}>
          {scoped ? (
            <>
              {visible.length.toLocaleString()} of {points.length.toLocaleString()} images
              {" "}shown — {selection.chips.map((c) => `${c.label} ${c.value}`).join(", ")
                || `search “${query}”`}
            </>
          ) : (
            <>{points.length.toLocaleString()} images in embedding space</>
          )}
          {clusterCount > 0 && (
            <> · {clusterCount} clusters (k-means in the original 768-D space)</>
          )}
          {query && (
            <> · a search is a ranking; the map can only show it as a set, so the
            order is gone here</>
          )}
        </div>
        {scope?.status === "loading" && (
          <span className="map-actions-hint">Resolving the active selection…</span>
        )}
        {scoped && scope?.status !== "loading" && visible.length === 0 && (
          <span className="map-actions-hint">
            This selection puts nothing on the map: either it matches no images, or
            none of the ones it matches have a projection. Empty on purpose, not broken.
          </span>
        )}
        {scope?.status === "error" && (
          <span className="pill warn-pill" role="status">
            Showing all {points.length.toLocaleString()} points: the active selection
            could not be resolved — {scope.message}
          </span>
        )}
      </div>

      <ScatterPlot
        points={visible}
        onSelect={(id) => navigate(`/samples/${id}`)}
        onSelectBox={(ids) => adopt(ids, "lasso",
          `${ids.length} images from one lasso on the projection`)}
        selectedIds={ringIds}
        ringIds={ringIds}
        colorOf={colorOf}
      />

      {/* The lasso's product is a set of images, so the images are on the page:
          a region you cannot see is a region you cannot judge. Selecting here
          and selecting on the canvas are the same selection — the ring above and
          the mark below are one `marked` id, which is why it lives on this page
          and the panel only reports hover back to it. */}
      <MapWorkingSet
        workingSet={workingSet}
        byId={byId}
        kept={kept}
        dropped={dropped}
        marked={marked}
        onHighlight={highlight}
        onUnhighlight={unhighlight}
        onToggleDrop={toggleDrop}
        onRestoreDropped={restoreDropped}
        onClear={clearSet}
        onScopeMap={() => selection.set({ ids: kept.join(","), page: "" })}
        nnStats={nnStats}
        dupNote={dupNote}
        onSelectIsolated={selectIsolated}
        onSelectDuplicates={() => void loadDuplicates()}
        busy={busy}
        onTag={applyTag}
        toast={toast}
        lastTag={lastTag}
      />

      {/* A 2-D projection of a 768-D space is a way to get around the dataset,
          not a measurement of it: most of a point's true nearest neighbours are
          not its neighbours here, and neither cluster size nor the gap between
          clusters carries meaning. Saying so in-product is the difference
          between a navigation aid and a misleading chart. */}
      {/* Open by default: a caveat you have to discover does not prevent the
          misreading it exists to prevent. It sits after the canvas because the
          map is the page's subject — the key strip above carries the one-line
          version a reader needs before their first click. */}
      <details className="caveat" open>
        <summary>
          Reading this map: it is a navigation surface, not an analysis surface
        </summary>
        <ul>
          <li>
            {/* An earlier version added "published measurements find that well
                over half of a point's true neighbours are missing here" — no
                citation, no author, and nothing in this repo measures it. On a
                panel where every other figure is computed from this corpus, an
                appeal to unnamed literature was the one unverifiable line. It
                is gone rather than dressed up: the neighbour statistics below
                are measured, and they make the point without borrowing. */}
            <strong>Distances are not similarities.</strong> UMAP preserves local
            neighbourhoods approximately and distorts distance by orders of
            magnitude. Two points sitting together here are a hypothesis to
            check against the measured neighbours on a sample's own page, never
            a result.
          </li>
          <li>
            <strong>Blob size and spacing mean nothing.</strong> The algorithm
            expands dense regions and contracts sparse ones, and the gap between two
            blobs is not a measure of how different they are.
          </li>
          <li>
            <strong>Colour is the only honest channel here.</strong> Position comes
            from UMAP and is distorted; colour comes straight from a per-image value
            and is not. So a colour <em>gradient</em> across a region is a real signal
            about the data, while the region's shape and placement are largely an
            artefact of the projection. When colouring by cluster, note those are
            k-means groups computed in the original 768-dimensional space and merely
            drawn here — not clusters discovered in 2-D.
          </li>
          <li>
            <strong>Isolation is measured, not read off the picture.</strong> The
            neighbour-similarity colouring and the “most isolated” shortcut both use
            the mean cosine to an image's 10 nearest neighbours in the full
            768-dimensional space, computed by the axis pass and stored per sample.
            Two dots far apart on this screen may be neighbours in that space; the
            number is the evidence, the gap is not.
          </li>
          <li>
            <strong>For an honest neighbourhood, open a sample.</strong> The
            “Similar images” list on the detail page is exact nearest-neighbour
            search in the full 768-dimensional embedding space; trust it over the
            picture whenever the two disagree.
          </li>
        </ul>
        <p>
          Use the map to find regions worth looking at, then lasso them
          (shift+drag) and inspect the actual images in the working set.
        </p>
      </details>
    </div>
  );
}
