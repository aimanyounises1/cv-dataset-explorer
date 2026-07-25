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

      <div className="controls" style={{ marginBottom: 10 }}>
        <div className="meta-line" style={{ marginBottom: 0 }}>
          {points.length.toLocaleString()} images in embedding space
        </div>
        {selected.length > 0 && (
          <form className="selection-bar" onSubmit={(e) => void bulkTag(e)}>
            <span className="pill accent">{selected.length} selected</span>
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
      {/* A 2-D projection of a 768-D space is a way to get around the dataset,
          not a measurement of it: most of a point's true nearest neighbours are
          not its neighbours here, and neither cluster size nor the gap between
          clusters carries meaning. Saying so in-product is the difference
          between a navigation aid and a misleading chart. */}
      {/* Open by default: a caveat you have to discover does not prevent the
          misreading it exists to prevent. */}
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

      <ScatterPlot
        points={points}
        onSelect={(id) => navigate(`/samples/${id}`)}
        onSelectBox={(ids) => { setToast(null); setSelected(ids); }}
        selectedIds={selectedSet}
        colorOf={colorOf}
      />
    </div>
  );
}
