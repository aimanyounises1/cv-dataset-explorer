import { NavLink, Route, Routes } from "react-router-dom";
import GalleryPage from "./pages/GalleryPage";
import MapPage from "./pages/MapPage";
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
          <NavLink to="/map">Embedding Map</NavLink>
          <NavLink to="/stats">Statistics</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<GalleryPage />} />
          <Route path="/samples/:id" element={<SamplePage />} />
          <Route path="/map" element={<MapPage />} />
          <Route path="/stats" element={<StatsPage />} />
        </Routes>
      </main>
    </div>
  );
}
