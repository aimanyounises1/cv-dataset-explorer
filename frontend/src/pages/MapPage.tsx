import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { MapPoint } from "../api/types";
import ScatterPlot from "../components/ScatterPlot";
import {
  COLOR_MODES, ColorMode, buildColorScale, formatDomainValue,
} from "../lib/mapColor";
import { rampStops, sequential } from "../lib/viz";

const COLOR_KEY = "cvde-map-color";

/** Ids a lasso may carry into the gallery through the URL.
 *
 * The gallery's id filter accepts far more than this, but only as a POST body —
 * a link cannot. At ~5 characters per id this stays near 4 kB, comfortably
 * inside every browser's address-bar limit. Past it the hand-off would silently
 * truncate, so it says so instead. */
const HANDOFF_LIMIT = 800;

export default function MapPage() {
  const [points, setPoints] = useState<MapPoint[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number[]>([]);
  const [tagName, setTagName] = useState("");
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [lastTag, setLastTag] = useState<string | null>(null);
  const [colorMode, setColorMode] = useState<ColorMode>(
    () => (localStorage.getItem(COLOR_KEY) as ColorMode) || "difficulty");

  useEffect(() => {
    try { localStorage.setItem(COLOR_KEY, colorMode); } catch { /* non-essential */ }
  }, [colorMode]);
  const navigate = useNavigate();

  useEffect(() => {
    api.map().then(setPoints).catch((e) => setError(String(e)));
  }, []);

  const selectedSet = useMemo(() => new Set(selected), [selected]);

  // Counted from the data, not configured: k is the backend's choice and this
  // page should report what arrived, not what was asked for.
  const clusterCount = useMemo(
    () => new Set((points ?? []).map((p) => p.cluster)).size, [points]);

  // Rebuilt only when the data or the dimension changes: the canvas redraws on
  // every pan and zoom, so the scale must not be reconstructed per frame.
  const { colorOf, ramp } = useMemo(
    () => buildColorScale(points ?? [], colorMode), [points, colorMode]);

  const bulkTag = async (e: FormEvent) => {
    e.preventDefault();
    const name = tagName.trim().toLowerCase();
    if (!name || selected.length === 0) return;
    setBusy(true);
    try {
      const res = await api.bulkTag(selected, name);
      setToast(`Tagged ${selected.length} samples as “${res.tag}”`);
      setLastTag(res.tag);
      setTagName("");
      setSelected([]);
    } catch (err) {
      setToast(err instanceof Error ? err.message : "Bulk tag failed");
      setLastTag(null);
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
    <div>
      <h1 className="section-title" style={{ marginTop: 0 }}>Embedding map</h1>
      {/* Colour is the map's only free channel — position is fixed by UMAP — so
          which dimension it carries is the most consequential control here. */}
      <div className="map-toolbar">
        <span className="density-label">Colour by</span>
        <select value={colorMode} aria-label="Colour points by"
                onChange={(e) => setColorMode(e.target.value as ColorMode)}>
          {COLOR_MODES.map((m) => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>

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
              {(ramp.categories ?? []).slice(0, 12).map((c) => (
                <span className="legend-swatch" key={c.label}>
                  <i className="legend-dot" style={{ background: c.color }} />
                  {c.label}
                </span>
              ))}
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
          <span className="map-key-kbd">shift+drag</span> lasso a region into a
          working set
        </span>
        <span className="map-key-item">
          isolated dots are coverage-gap candidates — confirm with the sample
          page's exact neighbours
        </span>
      </div>

      <div className="controls" style={{ marginBottom: 10 }}>
        <div className="meta-line" style={{ marginBottom: 0 }}>
          {points.length.toLocaleString()} images in embedding space
          {clusterCount > 0 && (
            <> · {clusterCount} clusters (k-means in the original 768-D space)</>
          )}
        </div>
        {/* The action row keeps its place in both states: what a lasso will do
            before one exists, what this lasso can do once it does. */}
        {selected.length === 0 && (
          <span className="map-actions-hint">
            Lasso a region to act on it — open the slice in the gallery, or tag
            it in bulk.
          </span>
        )}
        {selected.length > 0 && (
          <form className="selection-bar" onSubmit={(e) => void bulkTag(e)}>
            <span className="pill accent">{selected.length} selected</span>
            {/* The lasso's real product is a set. Sending it straight to the
                gallery is the short path; tagging first is for a set you want
                to keep, not one you want to look at. */}
            {selected.length <= HANDOFF_LIMIT ? (
              <Link className="primary button-link"
                    to={`/?ids=${selected.join(",")}`}
                    title="Open exactly these images in the gallery, where they can be searched, sorted and exported">
                Inspect {selected.length} in gallery
              </Link>
            ) : (
              <span className="meta-line" style={{ marginBottom: 0 }}>
                Too many for a link ({selected.length} &gt; {HANDOFF_LIMIT}) — tag
                the selection and open the tag instead.
              </span>
            )}
            <input
              aria-label="Tag name for selection"
              placeholder="tag name (e.g. night-scenes)"
              value={tagName}
              onChange={(e) => setTagName(e.target.value)}
            />
            <button className="primary" type="submit" disabled={busy || !tagName.trim()}>
              Tag selection
            </button>
            <button className="ghost" type="button" onClick={() => setSelected([])}>
              Clear
            </button>
          </form>
        )}
        {toast && (
          <span className="pill score" role="status">
            {toast}{" "}
            {lastTag && (
              <Link className="attr-link" to={`/?tag=${encodeURIComponent(lastTag)}`}>
                review slice in gallery
              </Link>
            )}
          </span>
        )}
      </div>
      <ScatterPlot
        points={points}
        onSelect={(id) => navigate(`/samples/${id}`)}
        onSelectBox={(ids) => { setToast(null); setSelected(ids); }}
        selectedIds={selectedSet}
        colorOf={colorOf}
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
            <strong>Distances are not similarities.</strong> UMAP preserves local
            neighbourhoods approximately and distorts distance by orders of magnitude.
            At this scale, published measurements find that well over half of a point's
            true high-dimensional neighbours are missing from its 2-D neighbourhood.
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
            <strong>For an honest neighbourhood, open a sample.</strong> The
            “Similar images” list on the detail page is exact nearest-neighbour
            search in the full 768-dimensional embedding space; trust it over the
            picture whenever the two disagree.
          </li>
        </ul>
        <p>
          Use the map to find regions worth looking at, then lasso them
          (shift+drag) and inspect the actual images.
        </p>
      </details>
    </div>
  );
}
