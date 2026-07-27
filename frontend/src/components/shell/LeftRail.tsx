import { useEffect, useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { api } from "../../api/client";
import FilterPanel from "./FilterPanel";
import SavedViews from "../SavedViews";

/**
 * Navigation grouped by the job you came to do, not by the table behind it.
 *
 * The seven routes were a flat list of peers, which made a first-time reader
 * ask "is Quality a kind of Statistics?" — a question the old top bar had no way
 * to answer. Grouping says what each destination is *for*.
 *
 * **Every path is unchanged.** Only labels and grouping move, so saved views
 * (which store bare query strings), bookmarks and shared links all keep working.
 * `/samples/:id` is deliberately absent: it is a child of Browse, reached by
 * clicking a frame, not a destination you navigate to cold.
 */

interface Item { to: string; label: string; end?: boolean; hint: string }
interface Group { title: string; items: Item[] }

const GROUPS: Group[] = [
  {
    title: "Find",
    items: [
      { to: "/", label: "Browse", end: true,
        hint: "Search and filter the corpus" },
      { to: "/map", label: "Embedding map",
        hint: "Lasso a region of embedding space" },
    ],
  },
  {
    title: "Audit",
    items: [
      { to: "/quality", label: "Caption quality",
        hint: "Rank captions least supported by their image" },
    ],
  },
  {
    title: "Trust",
    items: [
      { to: "/stats", label: "Dataset profile",
        hint: "Composition, coverage, duplicates and leakage" },
      { to: "/eval", label: "Retrieval benchmark",
        hint: "The tool measuring its own search accuracy" },
    ],
  },
  {
    title: "Ask",
    items: [
      { to: "/chat", label: "Assistant",
        hint: "Multi-agent answers rendered as live charts" },
    ],
  },
];

/** Fetched once for the whole session; the rail is on every route and this must
 * not become a request per navigation. Failure is silent — a missing corpus
 * count is not worth an error state in the furniture. */
let overviewPromise: Promise<{ total_samples: number } | null> | null = null;
function corpusSize() {
  if (!overviewPromise) {
    overviewPromise = api.overview().catch(() => null);
  }
  return overviewPromise;
}

const RAIL_KEY = "cvde-rail";

export default function LeftRail() {
  const [total, setTotal] = useState<number | null>(null);
  const location = useLocation();
  const navigate = useNavigate();
  /* Collapsed is workspace geometry, not shareable state — localStorage like
   * the gallery density, never the URL. The state is mirrored onto the root
   * element so the app grid's column width follows without prop-drilling
   * through App. */
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(RAIL_KEY) === "min");
  useEffect(() => {
    try { localStorage.setItem(RAIL_KEY, collapsed ? "min" : "full"); }
    catch { /* non-essential */ }
    if (collapsed) document.documentElement.dataset.rail = "min";
    else delete document.documentElement.dataset.rail;
  }, [collapsed]);

  useEffect(() => {
    let live = true;
    corpusSize().then((o) => { if (live && o) setTotal(o.total_samples); });
    return () => { live = false; };
  }, []);

  return (
    <nav className={`rail-l${collapsed ? " collapsed" : ""}`} aria-label="Sections">
      <div className="rail-brand">
        <span className="brand-mark" aria-hidden="true">◈</span>
        <span className="rail-brand-name">CV Dataset Explorer</span>
        <button className="rail-toggle" aria-expanded={!collapsed}
                title={collapsed ? "Expand the library rail" : "Collapse the library rail"}
                onClick={() => setCollapsed(!collapsed)}>
          {collapsed ? "»" : "«"}
        </button>
      </div>

      <div className="rail-groups">
        {GROUPS.map((g) => (
          <div className="rail-group" key={g.title}>
            <div className="eyebrow rail-group-title">{g.title}</div>
            {g.items.map((it) => (
              <NavLink key={it.to} to={it.to} end={it.end}
                       title={collapsed ? `${it.label} — ${it.hint}` : it.hint}
                       className={({ isActive }) =>
                         `rail-link${isActive ? " active" : ""}`}>
                {/* Both spans always render; CSS swaps them, so collapsing
                    never remounts the nav or loses keyboard focus. */}
                <span className="rail-link-label">{it.label}</span>
                <span className="rail-link-min" aria-hidden="true">{it.label[0]}</span>
              </NavLink>
            ))}
          </div>
        ))}

        {/* The library lives with navigation, not with the selection: a saved
            view is how you COME BACK, so it must be reachable when nothing is
            selected yet — which is exactly when the selection rail is absent.
            Saving captures the gallery's query string (the only route whose
            search params are a filter set); restoring navigates there. */}
        <div className="rail-group rail-library">
          <div className="eyebrow rail-group-title">Library</div>
          <SavedViews
            current={location.pathname === "/" ? location.search : ""}
            onRestore={(qs) => navigate(qs ? `/?${qs}` : "/")}
          />
        </div>

        <FilterPanel />
      </div>

      <div className="rail-foot">
        <div className="rail-corpus">
          {total != null ? `${total.toLocaleString()} images` : "Flickr8k"}
        </div>
        {/* The palette is the fast path for people who know what they want; the
            rail is the map for people who do not. Neither replaces the other, so
            the hint lives here rather than the palette growing a nav. */}
        <div className="rail-kbd"><kbd>⌘K</kbd> search anything</div>
      </div>
    </nav>
  );
}
