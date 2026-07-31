import { useLocation } from "react-router-dom";
import { AXES, Axis } from "../../api/types";
import AxisFilters from "../AxisFilters";
import FilterBar from "../FilterBar";
import IdListFilter from "../IdListFilter";
import { useSelection } from "../../hooks/useSelection";

/** Routes whose content is a filtered view of the corpus. Elsewhere — the
 * benchmark, the assistant — a filter panel would be a control with nothing to
 * control, so the rail simply does not draw one. The map is in that category
 * too: it always draws all 8,000 points (api.map takes no parameters), so a
 * panel there was exactly the control with nothing to control — it wrote
 * chips, shrank the rail's set, and changed nothing on the canvas, while a
 * lasso happily selected points "outside" the pretend filter. */
const FILTERABLE = ["/"];

/**
 * Every input that narrows the corpus, in one place on the left.
 *
 * These four surfaces used to sit above the results in the centre pane, three of
 * them behind collapsed `<details>` that auto-opened when active — so applying a
 * filter pushed the images further down. Precision cost you visibility, which is
 * exactly backwards.
 *
 * The difficulty axes in particular are no longer behind a disclosure. They are
 * the reason this tool exists — "show me the hardest 300 samples in validation"
 * is the question it was built to answer — and hiding them behind a summary made
 * the app's best feature its least discoverable one.
 */
export default function FilterPanel() {
  const { pathname } = useLocation();
  const sel = useSelection();

  if (!FILTERABLE.includes(pathname)) return null;

  return (
    <div className="rail-filters">
      <div className="eyebrow rail-group-title">Filters</div>

      <FilterBar
        filters={{
          split: String(sel.params.split ?? ""),
          tag: String(sel.params.tag ?? ""),
          vlm_tag: String(sel.params.vlm_tag ?? ""),
          attrs: (sel.params.attr as string[] | undefined) ?? [],
        }}
        onChange={(f) => sel.set({
          split: f.split, tag: f.tag, vlm_tag: f.vlm_tag, attr: f.attrs, page: "",
        })}
      />

      <AxisFilters
        value={sel.axisRanges}
        onChange={(next) => {
          const updates: Record<string, string> = { page: "" };
          for (const a of AXES) {
            const r = next[a as Axis];
            updates[`${a}_min`] = r && r[0] != null ? String(r[0]) : "";
            updates[`${a}_max`] = r && r[1] != null ? String(r[1]) : "";
          }
          // Replaces: the dual-range emits on `onInput`, so a single drag
          // through a range would otherwise bury the previous view under one
          // history entry per pixel travelled.
          sel.set(updates, { replace: true });
        }}
      />

      {/* Still a disclosure: pasting ids is a deliberate, occasional act, unlike
          the selects and sliders which are the everyday path. */}
      <IdListFilter
        value={String(sel.params.ids ?? "")}
        onChange={(v) => sel.set({ ids: v, page: "" })}
      />
    </div>
  );
}
