import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { ImagesBlock } from "../../api/blocks";
import { api } from "../../api/client";
import type { SampleCard } from "../../api/types";
import AxisLegend from "../AxisLegend";
import ImageCard from "../ImageCard";
import { formatCount } from "../../lib/viz";

export default function ImagesBlockView({ block }: { block: ImagesBlock }) {
  const navigate = useNavigate();
  const [items, setItems] = useState<SampleCard[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const ids = block.sample_ids;
  // Keyed on the ids themselves, not the array identity: a transcript re-render
  // hands over a fresh array with the same contents, and refetching on every
  // keystroke in the composer would be one request per character typed.
  const idKey = ids.join(",");

  useEffect(() => {
    if (ids.length === 0) {
      setItems([]);
      return;
    }
    const ctrl = new AbortController();
    setItems(null);
    setError(null);
    // The block carries ids, not cards — the agent chose a set, and the gallery
    // endpoint is what turns a set into thumbnails, captions and difficulty
    // axes. Fetching them here means an agent-selected set renders identically
    // to a hand-filtered one.
    api.listSamples({ ids: ids.join(","), per_page: 200 }, ctrl.signal)
      .then((list) => {
        // SQL returned them by id; the tool's order is the meaningful one
        // (ranked, or hardest first), so restore it.
        const byId = new Map(list.items.map((s) => [s.id, s]));
        setItems(ids.map((id) => byId.get(id)).filter((s): s is SampleCard => s !== undefined));
      })
      .catch((e: unknown) => {
        if (ctrl.signal.aborted) return;
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => ctrl.abort();
  }, [idKey]);

  const total = block.total ?? ids.length;
  const drill = block.drill ?? null;

  return (
    <div className="vblock-body">
      {error !== null && (
        <div className="error">Could not load these images: {error}</div>
      )}
      {items === null && error === null && (
        <div className="vblock-loading">Loading {ids.length} images…</div>
      )}
      {items !== null && items.length === 0 && (
        <div className="vblock-empty">No images in this set.</div>
      )}
      {items !== null && items.length > 0 && (
        <>
          {items.some((s) => s.axes) && (
            <div className="vblock-images-legend"><AxisLegend /></div>
          )}
          <div className="grid vblock-images">
            {items.map((s) => <ImageCard key={s.id} sample={s} />)}
          </div>
        </>
      )}

      {(total > ids.length || drill !== null) && (
        <div className="vblock-images-more">
          {total > ids.length && (
            <span className="vblock-hint">
              Showing {formatCount(ids.length)} of {formatCount(total)}.
            </span>
          )}
          {drill !== null && (
            <button
              type="button" className="vblock-more-link"
              title={`Open this slice in the gallery (?${drill})`}
              onClick={() => navigate(`/?${drill}`)}
            >
              Open {total > ids.length ? `all ${formatCount(total)}` : "this set"} in the gallery →
            </button>
          )}
        </div>
      )}
    </div>
  );
}
