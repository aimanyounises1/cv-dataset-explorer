import { Suspense, lazy } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import GalleryPage from "./pages/GalleryPage";

// All non-default routes load on demand, so the gallery ships without
// recharts, the canvas map, or the chat view in its initial bundle.
const SamplePage = lazy(() => import("./pages/SamplePage"));
const MapPage = lazy(() => import("./pages/MapPage"));
const StatsPage = lazy(() => import("./pages/StatsPage"));
const QualityPage = lazy(() => import("./pages/QualityPage"));
const EvalPage = lazy(() => import("./pages/EvalPage"));
const ChatPage = lazy(() => import("./pages/ChatPage"));

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">◈</span> CV Dataset Explorer
          <span className="brand-sub">Flickr8k</span>
        </div>
        <nav>
          <NavLink to="/" end>Gallery</NavLink>
          <NavLink to="/map">Map</NavLink>
          <NavLink to="/stats">Statistics</NavLink>
          <NavLink to="/quality">Quality</NavLink>
          <NavLink to="/eval">Benchmark</NavLink>
          <NavLink to="/chat">Assistant</NavLink>
        </nav>
      </header>
      <main>
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
    </div>
  );
}
