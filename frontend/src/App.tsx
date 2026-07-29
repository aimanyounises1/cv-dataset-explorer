import { Suspense, lazy, useEffect, useRef } from "react";
import type { RefObject } from "react";
import { Link, Route, Routes, useLocation } from "react-router-dom";
import CommandPalette from "./components/CommandPalette";
import Toasts from "./components/Toast";
import LeftRail from "./components/shell/LeftRail";
import SelectionRail from "./components/shell/SelectionRail";
import { useSelection } from "./hooks/useSelection";
import type { Selection } from "./hooks/useSelection";
import GalleryPage from "./pages/GalleryPage";

// All non-default routes load on demand, so the gallery ships without
// recharts, the canvas map, or the chat view in its initial bundle.
const SamplePage = lazy(() => import("./pages/SamplePage"));
const MapPage = lazy(() => import("./pages/MapPage"));
const StatsPage = lazy(() => import("./pages/StatsPage"));
const QualityPage = lazy(() => import("./pages/QualityPage"));
const EvalPage = lazy(() => import("./pages/EvalPage"));
const ChatPage = lazy(() => import("./pages/ChatPage"));
const ComparePage = lazy(() => import("./pages/ComparePage"));

/** Between the page's name and what distinguishes this tab from another tab of
 * the same page; and between both and the product name. */
const TITLE_SLICE_SEP = " · ";
const TITLE_APP_SEP = " — ";

/** The product name exactly as the document was served with it (index.html's
 * <title>), so it is stated in one place rather than copied here. Composed
 * titles always end with it, so taking the last segment recovers the name even
 * if a hot reload re-evaluates this module after a title has been written. */
const APP_NAME = document.title.split(TITLE_APP_SEP).pop()?.trim()
  || document.title;

/** The only sub-view parameter in the app: `/stats?view=<id>` addresses one of
 * the dataset profile's five views. The raw id is used rather than the view's
 * display label because the id is what the bookmark and the shared link carry,
 * and because the labels live in StatsPage and LeftRail — a third copy here is
 * a copy that drifts. */
const VIEW_PARAM = "view";

/** What tells two tabs of the *same* page apart: the query you searched for,
 * the profile view you opened, or the filters you set. The filters come from
 * `useSelection`, so the tab title, the rail's chips and the URL cannot
 * disagree about what is currently selected. */
function sliceLabel(search: string, chips: Selection["chips"],
                    query: string): string {
  if (query) return `“${query}”`;
  const view = new URLSearchParams(search).get(VIEW_PARAM);
  if (view) return view;
  if (chips.length === 0) return "";
  const [first, ...rest] = chips;
  // One chip spelled out, the remainder counted: a tab strip truncates, so the
  // first characters have to carry the difference.
  return `${first.label} ${first.value}`
    + (rest.length > 0 ? ` +${rest.length}` : "");
}

/**
 * The tab, the history entry and the bookmark say which page and which slice.
 *
 * Every route used to serve the served title, "CV Dataset Explorer", so the
 * eight destinations were indistinguishable in the tab strip and in Back —
 * exactly the places an audit uses when it keeps `?split=train` and
 * `?split=test` open side by side, and exactly what a bookmark keeps.
 *
 * The page's name is read from the heading the page already renders rather
 * than declared a second time here: the tab and the `<h1>` are then the same
 * string by construction. A heading arrives late — a lazy route is still
 * loading, or the profile is still waiting on its first fetch — and it can
 * change in place when the sample stepper moves to the next id, so the pane is
 * observed rather than read once.
 */
function RouteTitle({ paneRef }: { paneRef: RefObject<HTMLElement | null> }) {
  const location = useLocation();
  const { chips, query } = useSelection();

  useEffect(() => {
    const pane = paneRef.current;
    if (!pane) return undefined;
    const slice = sliceLabel(location.search, chips, query);
    const apply = () => {
      const heading = pane.querySelector("h1")?.textContent?.trim();
      const next = [[heading, slice].filter(Boolean).join(TITLE_SLICE_SEP),
                    APP_NAME].filter(Boolean).join(TITLE_APP_SEP);
      // Assign only on a real change: an unchanged title is a write the
      // browser does not need on every keystroke of a streaming answer.
      if (document.title !== next) document.title = next;
    };
    apply();
    const observer = new MutationObserver(apply);
    observer.observe(pane,
                     { childList: true, subtree: true, characterData: true });
    return () => observer.disconnect();
  }, [paneRef, location.pathname, location.search, chips, query]);

  return null;
}

/**
 * Three columns: inputs on the left, the artifact in the middle, the set on the
 * right.
 *
 * The old shell stacked all three in one vertical column, so on a filtered
 * gallery you scrolled past 481px of controls before seeing an image — the more
 * precisely you specified a set, the less of it you could see. Splitting them
 * onto separate axes means the centre pane starts at the top and stays there
 * however many filters are active.
 *
 * The right rail renders only when something is selected, so browsing the whole
 * corpus gets the grid columns back.
 */
export default function App() {
  const paneRef = useRef<HTMLElement>(null);
  return (
    <div className="app">
      <LeftRail />
      <main className="pane" ref={paneRef}>
        <Suspense fallback={<div className="loading">Loading…</div>}>
          <Routes>
            <Route path="/" element={<GalleryPage />} />
            <Route path="/samples/:id" element={<SamplePage />} />
            <Route path="/map" element={<MapPage />} />
            <Route path="/stats" element={<StatsPage />} />
            <Route path="/quality" element={<QualityPage />} />
            <Route path="/eval" element={<EvalPage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/compare" element={<ComparePage />} />
            {/* An unmatched path must say so — an empty <Routes> is
                indistinguishable from a crashed build. It says so with a
                heading, like every other route: this is the one page a reader
                arrives at already unsure where they are, and it was the one
                page whose document outline started at nothing. */}
            <Route path="*" element={
              <div className="empty">
                <h1 className="section-title" style={{ marginTop: 0 }}>
                  Nothing at this address
                </h1>
                <p>
                  No view answers this path — <Link to="/">back to Browse</Link>.
                </p>
              </div>
            } />
          </Routes>
        </Suspense>
      </main>
      <SelectionRail />
      {/* Renders nothing; it only names the tab. Kept out of App's own body so
          that subscribing to the URL's selection re-renders this and not the
          whole shell. */}
      <RouteTitle paneRef={paneRef} />
      {/* Outside the panes and outside Suspense: available on every route,
          including while a lazy route is still loading. */}
      <CommandPalette />
      {/* Also outside the panes: a drop's confirmation and its Undo must
          survive whatever navigation the drop caused. */}
      <Toasts />
    </div>
  );
}
