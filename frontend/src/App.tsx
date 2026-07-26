import { Suspense, lazy } from "react";
import { Route, Routes } from "react-router-dom";
import CommandPalette from "./components/CommandPalette";
import LeftRail from "./components/shell/LeftRail";
import SelectionRail from "./components/shell/SelectionRail";
import GalleryPage from "./pages/GalleryPage";

// All non-default routes load on demand, so the gallery ships without
// recharts, the canvas map, or the chat view in its initial bundle.
const SamplePage = lazy(() => import("./pages/SamplePage"));
const MapPage = lazy(() => import("./pages/MapPage"));
const StatsPage = lazy(() => import("./pages/StatsPage"));
const QualityPage = lazy(() => import("./pages/QualityPage"));
const EvalPage = lazy(() => import("./pages/EvalPage"));
const ChatPage = lazy(() => import("./pages/ChatPage"));

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
  return (
    <div className="app">
      <LeftRail />
      <main className="pane">
        <Suspense fallback={<div className="loading">Loading…</div>}>
          <Routes>
            <Route path="/" element={<GalleryPage />} />
            <Route path="/samples/:id" element={<SamplePage />} />
            <Route path="/map" element={<MapPage />} />
            <Route path="/stats" element={<StatsPage />} />
            <Route path="/quality" element={<QualityPage />} />
            <Route path="/eval" element={<EvalPage />} />
            <Route path="/chat" element={<ChatPage />} />
          </Routes>
        </Suspense>
      </main>
      <SelectionRail />
      {/* Outside the panes and outside Suspense: available on every route,
          including while a lazy route is still loading. */}
      <CommandPalette />
    </div>
  );
}
