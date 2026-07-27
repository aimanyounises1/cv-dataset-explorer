import { useSearchParams } from "react-router-dom";
import ActiveFilters from "../ActiveFilters";
import SavedViews from "../SavedViews";
import SetSummary from "../SetSummary";
import { useSelection } from "../../hooks/useSelection";

/**
 * The current selection, given a permanent home.
 *
 * The application's organising idea is that every view produces a set — the map
 * lasso, the quality brush, a pasted id list, a chart facet. Until now each view
 * grew its own bespoke hand-off strip (`.selection-bar` on the map,
 * `.dist-actions` on Quality, three stacked accordions on the gallery), so the
 * same object looked different, lived somewhere different, and offered different
 * actions depending on which page happened to have produced it.
 *
 * Here it is one object in one place: what the set is built from, what it is
 * made of, and what can leave the tool with it. The rail is a read of
 * `useSelection`, which is itself a read of the URL, so it is correct on every
 * route without any view having to tell it anything.
 *
 * It shows nothing when nothing is selected. Browsing all 8,000 images is the
 * case where you want the grid columns back, and a rail full of empty states
 * would be furniture pretending to be information.
 */
export default function SelectionRail() {
  const sel = useSelection();
  const [searchParams, setSearchParams] = useSearchParams();

  // Shown for a search too: its results are a set you can export, even
  // though a query adds no chip.
  if (!sel.exportable) return null;

  return (
    <aside className="rail-r" aria-label="Current selection">
      <div className="rail-r-head">
        <div className="eyebrow">The set</div>
        {sel.origin
          ? <div className="rail-origin">from {sel.origin}</div>
          : sel.query && <div className="rail-origin">search “{sel.query}”</div>}
      </div>

      {/* No heading of its own: ActiveFilters already labels itself
          "filtering by", and two stacked headings for one list read as a
          nesting that isn't there. */}
      {sel.chips.length > 0 && (
      <section className="rail-section">
        <ActiveFilters chips={sel.chips} onRemove={sel.remove}
                       onClearAll={sel.clearAll} />
      </section>
      )}

      {/* Always open here. On the gallery this was a collapsed accordion that
          most people never opened, which meant the one panel that answers
          "what did I just select?" was the one thing hidden by default. */}
      <section className="rail-section">
        <SetSummary params={sel.params} active={sel.active} />
      </section>

      <section className="rail-section">
        <div className="eyebrow rail-section-title">Take it with you</div>
        <div className="rail-export">
          {(["csv", "jsonl", "json"] as const).map((f) => (
            <a key={f} className="pill export-pill" href={sel.exportHref(f)}
               title={`Download this selection as ${f.toUpperCase()}, with the `
                      + `query recorded in the manifest and every computed score `
                      + `on each row`
                      + (sel.query
                         ? ". A search export is the ranked results to the "
                           + "fusion depth, not every match — the manifest "
                           + "records the depth and a truncated flag"
                         : "")}>
              {f}
            </a>
          ))}
        </div>
        <SavedViews
          current={searchParams.toString()}
          onRestore={(qs) => setSearchParams(new URLSearchParams(qs),
                                             { replace: false })}
        />
      </section>
    </aside>
  );
}
