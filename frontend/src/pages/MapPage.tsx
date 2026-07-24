import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { MapPoint } from "../api/types";
import ScatterPlot from "../components/ScatterPlot";

export default function MapPage() {
  const [points, setPoints] = useState<MapPoint[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number[]>([]);
  const [tagName, setTagName] = useState("");
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [lastTag, setLastTag] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    api.map().then(setPoints).catch((e) => setError(String(e)));
  }, []);

  const selectedSet = useMemo(() => new Set(selected), [selected]);

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
            <strong>Cluster size and spacing mean nothing.</strong> The algorithm
            expands dense regions and contracts sparse ones, and the gap between two
            blobs is not a measure of how different they are. Colours are k-means
            groups in the original space, drawn here — not clusters discovered in 2-D.
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
      />
    </div>
  );
}
