import { NavLink, Route, Routes } from "react-router-dom";
import ChatPage from "./pages/ChatPage";
import EvalPage from "./pages/EvalPage";
import GalleryPage from "./pages/GalleryPage";
import MapPage from "./pages/MapPage";
import QualityPage from "./pages/QualityPage";
import SamplePage from "./pages/SamplePage";
import StatsPage from "./pages/StatsPage";

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
        <Routes>
          <Route path="/" element={<GalleryPage />} />
          <Route path="/samples/:id" element={<SamplePage />} />
          <Route path="/map" element={<MapPage />} />
          <Route path="/stats" element={<StatsPage />} />
          <Route path="/quality" element={<QualityPage />} />
          <Route path="/eval" element={<EvalPage />} />
          <Route path="/chat" element={<ChatPage />} />
        </Routes>
      </main>
    </div>
  );
}
