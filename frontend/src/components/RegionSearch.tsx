import { useEffect, useRef, useState } from "react";
import type { SampleCard } from "../api/types";
import ImageCard from "./ImageCard";
import "../styles/region.css";

/**
 * Region evidence on a sample's own image: mark a rectangle (or accept a
 * detector suggestion) and use it as a positive or negative query through
 * POST /api/search/by-region. The server crops the original from normalized
 * geometry, so the search reproduces from the request alone.
 *
 * Detection is an optional layer: the "Suggest regions" control renders only
 * after /api/detect/status says ready, and its absence explains itself in the
 * title of nothing — no fake buttons.
 */
interface Rect { x: number; y: number; w: number; h: number }
interface DetBox extends Rect { label: string; score: number }

interface RegionResults {
  items: SampleCard[];
  message: string | null;
  role: "positive" | "negative";
}

let detectStatusPromise: Promise<{ ready: boolean; reason: string | null }> | null = null;
const detectStatus = () => {
  detectStatusPromise ??= fetch("/api/detect/status")
    .then((r) => r.json())
    .catch(() => ({ ready: false, reason: "status unavailable" }));
  return detectStatusPromise;
};

export default function RegionSearch({ sampleId, imageUrl }:
    { sampleId: number; imageUrl: string }) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [marking, setMarking] = useState(false);
  const [rect, setRect] = useState<Rect | null>(null);
  const [drag, setDrag] = useState<{ x: number; y: number } | null>(null);
  const [boxes, setBoxes] = useState<DetBox[] | null>(null);
  const [detReady, setDetReady] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [steer, setSteer] = useState("");
  const [results, setResults] = useState<RegionResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { detectStatus().then((s) => setDetReady(s.ready)); }, []);
  // A new sample invalidates every piece of region state at once.
  useEffect(() => {
    setRect(null); setBoxes(null); setResults(null); setError(null);
    setMarking(false);
  }, [sampleId]);

  const norm = (e: React.MouseEvent): { x: number; y: number } | null => {
    const el = imgRef.current;
    if (!el) return null;
    const b = el.getBoundingClientRect();
    return {
      x: Math.min(Math.max((e.clientX - b.left) / b.width, 0), 1),
      y: Math.min(Math.max((e.clientY - b.top) / b.height, 0), 1),
    };
  };

  const search = async (role: "positive" | "negative") => {
    if (!rect) return;
    setBusy(role);
    setError(null);
    try {
      const r = await fetch("/api/search/by-region", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_id: sampleId, ...rect, role,
          text: role === "positive" && steer.trim() ? steer.trim() : undefined,
          top_k: 18,
        }),
      });
      const body = await r.json();
      if (!r.ok) {
        const detail = Array.isArray(body.detail)
          ? body.detail[0]?.msg : body.detail;
        throw new Error(String(detail ?? r.status).replace(/^Value error, /, ""));
      }
      setResults({ items: body.items, message: body.message, role });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const suggest = async () => {
    setBusy("detect");
    setError(null);
    try {
      const r = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_id: sampleId }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(String(body.detail ?? r.status));
      setBoxes(body.boxes);
      if (!body.boxes.length) setError("The detector proposed no regions here.");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="region-search" aria-label="Region search">
      <div className="rs-actions">
        <button className={marking ? "primary" : "ghost"}
                onClick={() => { setMarking((m) => !m); setBoxes(null); }}
                title="Drag on the image to mark a rectangle — drawn by you, no segmentation model involved">
          {marking ? "Marking — drag on the image" : "Mark a region"}
        </button>
        {detReady && (
          <button className="ghost" onClick={suggest} disabled={busy === "detect"}
                  title="Zero-shot box proposals (Grounding DINO tiny, local) — click a box to use it as evidence">
            {busy === "detect" ? "Detecting…" : "Suggest regions"}
          </button>
        )}
        {rect && (
          <>
            <input className="rs-steer" value={steer} placeholder="steer with text (optional)"
                   onChange={(e) => setSteer(e.target.value)} maxLength={200} />
            <button className="primary" disabled={busy !== null}
                    onClick={() => search("positive")}>
              {busy === "positive" ? "Searching…" : "Find similar to region"}
            </button>
            <button className="ghost" disabled={busy !== null}
                    onClick={() => search("negative")}>
              {busy === "negative" ? "Searching…" : "Search away from it"}
            </button>
            <button className="ghost" onClick={() => { setRect(null); setResults(null); }}>
              Clear
            </button>
          </>
        )}
      </div>

      <div className={`rs-stage${marking ? " marking" : ""}`}
           onMouseDown={(e) => {
             if (!marking) return;
             const p = norm(e);
             if (p) { setDrag(p); setRect({ x: p.x, y: p.y, w: 0, h: 0 }); }
             e.preventDefault();
           }}
           onMouseMove={(e) => {
             if (!marking || !drag) return;
             const p = norm(e);
             if (!p) return;
             setRect({ x: Math.min(drag.x, p.x), y: Math.min(drag.y, p.y),
                       w: Math.abs(p.x - drag.x), h: Math.abs(p.y - drag.y) });
           }}
           onMouseUp={() => { if (drag) { setDrag(null); setMarking(false); } }}>
        <img ref={imgRef} src={imageUrl} alt="" draggable={false} />
        {rect && rect.w > 0.005 && (
          <div className="rs-rect" style={{
            left: `${rect.x * 100}%`, top: `${rect.y * 100}%`,
            width: `${rect.w * 100}%`, height: `${rect.h * 100}%` }} />
        )}
        {(boxes ?? []).map((b, i) => (
          <button key={i} className="rs-box"
                  style={{ left: `${b.x * 100}%`, top: `${b.y * 100}%`,
                           width: `${b.w * 100}%`, height: `${b.h * 100}%` }}
                  title={`${b.label} — detector confidence ${b.score}; click to use as evidence`}
                  onClick={() => { setRect({ x: b.x, y: b.y, w: b.w, h: b.h });
                                   setBoxes(null); }}>
            <span>{b.label} {Math.round(b.score * 100)}%</span>
          </button>
        ))}
      </div>

      {error && <p className="rs-error">{error}</p>}
      {results && (
        <div className="rs-results">
          <p className="rs-note">
            {results.role === "positive"
              ? "Ranked by similarity to the marked region (basis: composed)."
              : results.message ?? "Ranked by distance from the marked region."}
          </p>
          <div className="grid rs-grid">
            {results.items.map((s) => (
              <ImageCard key={s.id} sample={s} scoreBasis="composed" />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
