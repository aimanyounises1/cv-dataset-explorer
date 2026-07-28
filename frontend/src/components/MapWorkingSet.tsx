import { FormEvent, useState } from "react";
import { Link } from "react-router-dom";
import type { MapPoint } from "../api/types";

/** Ids a lasso may carry into the gallery through the URL.
 *
 * The gallery's id filter accepts far more than this, but only as a POST body —
 * a link cannot. At ~5 characters per id this stays near 4 kB, comfortably
 * inside every browser's address-bar limit. Past it the hand-off would silently
 * truncate, so it says so instead. */
const HANDOFF_LIMIT = 800;

/** Thumbnails drawn for a working set. The set itself is not capped — every
 * action below applies to all of it — but a lasso can hold three thousand
 * images and rendering three thousand <img> tags is how a linked view becomes
 * an unusable one. The panel says which number is which. */
const GRID_LIMIT = 180;

/** How many images the isolation shortcut selects: a page of work, not a
 * threshold on a scale nobody has calibrated. Exported because the button here
 * promises the number and the map's ranking produces it — one constant, so the
 * label and the action cannot drift apart. */
export const ISOLATED_N = 100;

/** The neighbour-similarity distribution of whatever the map is showing.
 * Absent when the measurement has not been computed. */
export interface NeighbourStats {
  n: number;
  p5: number;
  median: number;
  min: number;
}

/** A set as adopted, before any drops: the ids, and one line naming exactly
 * what was measured to produce them. */
export interface WorkingSetView {
  ids: number[];
  label: string;
}

interface Props {
  /** The adopted set, or null when nothing is selected. */
  workingSet: WorkingSetView | null;
  /** Thumbnails by id — the panel draws images, not point records. */
  byId: ReadonlyMap<number, MapPoint>;
  /** Ids still in the set: every action here applies to exactly these. */
  kept: number[];
  /** Ids dropped by hand — still drawn, faded, because a set you cannot
   *  un-edit is a set you cannot trust. */
  dropped: ReadonlySet<number>;
  /** The id ringed on the canvas; the grid marks the same one, which is the
   *  whole of the link between the two halves. */
  marked: number | null;
  onHighlight: (id: number) => void;
  onUnhighlight: (id: number) => void;
  onToggleDrop: (id: number) => void;
  onRestoreDropped: () => void;
  onClear: () => void;
  /** Put the kept ids into the URL as the map's own selection. Until this
   *  existed a lasso lived in component state: it could be handed to the
   *  gallery through a one-way link, but it could not be shared, reloaded,
   *  saved as a view, exported from the rail or added to an album — the very
   *  things the rest of the app does with a selection. */
  onScopeMap: () => void;
  /** Measured entry points into a set, and what the last one reported. */
  nnStats: NeighbourStats | null;
  dupNote: string | null;
  onSelectIsolated: () => void;
  onSelectDuplicates: () => void;
  /** True while a request one of these actions started is in flight. */
  busy: boolean;
  /** Applies the tag to `kept`; resolves true when it stuck, which is when —
   *  and the only time — the input clears. */
  onTag: (name: string) => Promise<boolean>;
  toast: string | null;
  lastTag: string | null;
}

/** The lasso's product is a set of images, so the images are on the page: a
 * region you cannot see is a region you cannot judge. Selecting here and
 * selecting on the canvas are the same selection — this panel owns only the
 * tag field; the set, the drops and the ring live with the map. */
export default function MapWorkingSet({
  workingSet, byId, kept, dropped, marked,
  onHighlight, onUnhighlight, onToggleDrop, onRestoreDropped, onClear, onScopeMap,
  nnStats, dupNote, onSelectIsolated, onSelectDuplicates,
  busy, onTag, toast, lastTag,
}: Props) {
  const [tagName, setTagName] = useState("");

  const submitTag = async (e: FormEvent) => {
    e.preventDefault();
    const name = tagName.trim().toLowerCase();
    if (!name || kept.length === 0) return;
    if (await onTag(name)) setTagName("");
  };

  const shown = workingSet ? workingSet.ids.slice(0, GRID_LIMIT) : [];

  return (
    <section className="map-workbench" aria-label="Working set">
      <div className="mw-head">
        <span className="mw-title">Working set</span>
        {workingSet ? (
          <span className="mw-source">{workingSet.label}</span>
        ) : (
          <span className="mw-source">
            nothing selected — <span className="map-key-kbd">shift+drag</span> the
            map, or start from a measured signal below
          </span>
        )}
      </div>

      {/* Two entry points that do not need a hand: both are measurements over
          the SigLIP embeddings, and both say what they measured. */}
      <div className="mw-tools">
        <button className="ghost" type="button" onClick={onSelectIsolated}
                disabled={!nnStats}
                title="Select the images with the lowest mean cosine to their 10 nearest neighbours">
          Most isolated {ISOLATED_N}
        </button>
        <button className="ghost" type="button" onClick={onSelectDuplicates}
                disabled={busy}
                title="Select the images in the strongest near-duplicate pairs">
          Near-duplicate pairs
        </button>
        {nnStats ? (
          <span className="mw-measure">
            Neighbour similarity over the {nnStats.n.toLocaleString()} images shown:
            median <b>{nnStats.median.toFixed(3)}</b>, 5th percentile{" "}
            <b>{nnStats.p5.toFixed(3)}</b>, lowest <b>{nnStats.min.toFixed(3)}</b> —
            mean cosine to each image's 10 nearest neighbours in the 768-D SigLIP
            space, not a distance on this screen.
          </span>
        ) : (
          <span className="mw-measure">
            Neighbour similarity is unavailable — run <code>python -m app.ingest</code> to
            compute the embeddings and the axis pass that measures it.
          </span>
        )}
        {dupNote && (
          <span className="mw-measure">
            {dupNote} A near-duplicate here is only “SigLIP cosine above the
            threshold”, which is not a calibrated notion of the same image — look
            at the pairs before believing the count.{" "}
            <Link className="attr-link" to="/stats">Full leakage report</Link>
          </span>
        )}
      </div>

      {workingSet && (
        <>
          {/* The action row: what this set can do, and what it costs to move.
              Tagging keeps a set; the gallery hand-off looks at one. */}
          <form className="selection-bar" onSubmit={(e) => void submitTag(e)}>
            <span className="pill accent">{kept.length} selected</span>
            {dropped.size > 0 && (
              <button className="ghost" type="button" onClick={onRestoreDropped}>
                Restore {dropped.size} dropped
              </button>
            )}
            {kept.length > 0 && kept.length <= HANDOFF_LIMIT && (
              <button className="ghost" type="button" onClick={onScopeMap}
                      title="Narrow the map to exactly these images and put them in the address bar, so the set can be shared, reloaded, saved as a view or added to an album">
                Keep {kept.length} on the map
              </button>
            )}
            {kept.length > 0 && kept.length <= HANDOFF_LIMIT ? (
              <Link className="primary button-link"
                    to={`/?ids=${kept.join(",")}`}
                    title="Open exactly these images in the gallery, where they can be searched, sorted and exported">
                Inspect {kept.length} in gallery
              </Link>
            ) : (
              <span className="meta-line" style={{ marginBottom: 0 }}>
                {kept.length === 0
                  ? "Every image was dropped — restore some, or lasso again."
                  : `Too many for a link (${kept.length} > ${HANDOFF_LIMIT}) — tag the `
                    + "selection and open the tag instead."}
              </span>
            )}
            <input
              aria-label="Tag name for selection"
              placeholder="tag name (e.g. night-scenes)"
              value={tagName}
              onChange={(e) => setTagName(e.target.value)}
            />
            <button className="primary" type="submit"
                    disabled={busy || !tagName.trim() || kept.length === 0}>
              Tag selection
            </button>
            <button className="ghost" type="button" onClick={onClear}>
              Clear
            </button>
          </form>

          <ul className="mw-grid">
            {shown.map((id) => {
              const p = byId.get(id);
              const isDropped = dropped.has(id);
              return (
                <li key={id}
                    className={`mw-card${isDropped ? " dropped" : ""}${
                      marked === id ? " marked" : ""}`}>
                  <button
                    type="button"
                    className="mw-thumb"
                    aria-pressed={!isDropped}
                    aria-label={isDropped
                      ? `Sample ${id} — dropped from the set, click to put it back`
                      : `Sample ${id} — in the set, click to drop it`}
                    onMouseEnter={() => onHighlight(id)}
                    onMouseLeave={() => onUnhighlight(id)}
                    onFocus={() => onHighlight(id)}
                    onBlur={() => onUnhighlight(id)}
                    onClick={() => onToggleDrop(id)}
                  >
                    {p ? <img src={p.thumb_url} alt="" loading="lazy" /> : <span />}
                  </button>
                  <Link className="mw-open" to={`/samples/${id}`}
                        title={`Open sample ${id}`}>#{id}</Link>
                </li>
              );
            })}
          </ul>
          <div className="meta-line" style={{ marginTop: 8, marginBottom: 0 }}>
            {workingSet.ids.length > GRID_LIMIT
              ? `Showing the first ${GRID_LIMIT} of ${workingSet.ids.length.toLocaleString()} images `
                + "— the actions above apply to the whole set."
              : `${workingSet.ids.length.toLocaleString()} image${workingSet.ids.length === 1 ? "" : "s"} `
                + "— click one to drop it from the set."}
          </div>
        </>
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
    </section>
  );
}
