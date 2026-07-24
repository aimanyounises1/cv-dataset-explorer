import { useCallback, useEffect, useRef, useState } from "react";
import type { MapPoint } from "../api/types";

const PALETTE = [
  "#4f9cff", "#3ecf8e", "#ff7a7a", "#ffbe50", "#c792ea", "#4dd0e1",
  "#f48fb1", "#aed581", "#ffb74d", "#90a4ae", "#7986cb", "#e57373",
];

interface Props {
  points: MapPoint[];
  onSelect: (id: number) => void;
  onSelectBox?: (ids: number[]) => void;
  selectedIds?: Set<number>;
}

interface View { scale: number; tx: number; ty: number; }
interface Box { x0: number; y0: number; x1: number; y1: number; }

/** Canvas scatter of the embedding space. Scroll = zoom, drag = pan,
 * click = open sample, Shift+drag = box-select for bulk actions. */
export default function ScatterPlot({ points, onSelect, onSelectBox, selectedIds }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const viewRef = useRef<View>({ scale: 1, tx: 0, ty: 0 });
  const dragRef = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const boxRef = useRef<Box | null>(null);
  const [hover, setHover] = useState<{ point: MapPoint; sx: number; sy: number } | null>(null);

  const toScreen = useCallback((p: { x: number; y: number }, w: number, h: number) => {
    const { scale, tx, ty } = viewRef.current;
    const margin = 24;
    const bx = margin + p.x * (w - 2 * margin);
    const by = margin + (1 - p.y) * (h - 2 * margin);
    return { sx: bx * scale + tx, sy: by * scale + ty };
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const r = Math.max(1.6, Math.min(3.2, 2.2 * viewRef.current.scale ** 0.5));
    for (const p of points) {
      const { sx, sy } = toScreen(p, w, h);
      if (sx < -5 || sy < -5 || sx > w + 5 || sy > h + 5) continue;
      const selected = selectedIds?.has(p.id);
      ctx.fillStyle = selected ? "#ffffff" : PALETTE[p.cluster % PALETTE.length];
      ctx.globalAlpha = selected ? 1 : 0.85;
      ctx.beginPath();
      ctx.arc(sx, sy, selected ? r + 1 : r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (boxRef.current) {
      const b = boxRef.current;
      ctx.strokeStyle = "#4f9cff";
      ctx.fillStyle = "rgba(79,156,255,0.12)";
      ctx.lineWidth = 1;
      const x = Math.min(b.x0, b.x1), y = Math.min(b.y0, b.y1);
      const bw = Math.abs(b.x1 - b.x0), bh = Math.abs(b.y1 - b.y0);
      ctx.fillRect(x, y, bw, bh);
      ctx.strokeRect(x, y, bw, bh);
    }
  }, [points, toScreen, selectedIds]);

  // Native non-passive wheel listener: zoom without scrolling the page.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const v = viewRef.current;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const newScale = Math.min(30, Math.max(0.5, v.scale * factor));
      v.tx = mx - ((mx - v.tx) / v.scale) * newScale;
      v.ty = my - ((my - v.ty) / v.scale) * newScale;
      v.scale = newScale;
      setHover(null);
      draw();
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [draw]);

  useEffect(() => {
    draw();
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [draw]);

  const findNearest = (mx: number, my: number): MapPoint | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    let best: MapPoint | null = null;
    let bestDist = 12 * 12;
    for (const p of points) {
      const { sx, sy } = toScreen(p, w, h);
      const d = (sx - mx) ** 2 + (sy - my) ** 2;
      if (d < bestDist) {
        bestDist = d;
        best = p;
      }
    }
    return best;
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (e.shiftKey && onSelectBox) {
      boxRef.current = {
        x0: e.clientX - rect.left, y0: e.clientY - rect.top,
        x1: e.clientX - rect.left, y1: e.clientY - rect.top,
      };
    } else {
      dragRef.current = { x: e.clientX, y: e.clientY, moved: false };
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    if (boxRef.current && e.buttons === 1) {
      boxRef.current.x1 = mx;
      boxRef.current.y1 = my;
      setHover(null);
      draw();
      return;
    }
    if (dragRef.current && e.buttons === 1) {
      const dx = e.clientX - dragRef.current.x;
      const dy = e.clientY - dragRef.current.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) dragRef.current.moved = true;
      viewRef.current.tx += dx;
      viewRef.current.ty += dy;
      dragRef.current.x = e.clientX;
      dragRef.current.y = e.clientY;
      setHover(null);
      draw();
      return;
    }
    const p = findNearest(mx, my);
    if (p) {
      const canvas = canvasRef.current!;
      const s = toScreen(p, canvas.clientWidth, canvas.clientHeight);
      setHover({ point: p, sx: s.sx, sy: s.sy });
    } else {
      setHover(null);
    }
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (boxRef.current && onSelectBox) {
      const b = boxRef.current;
      boxRef.current = null;
      const canvas = canvasRef.current!;
      const [x0, x1] = [Math.min(b.x0, b.x1), Math.max(b.x0, b.x1)];
      const [y0, y1] = [Math.min(b.y0, b.y1), Math.max(b.y0, b.y1)];
      if (x1 - x0 > 4 && y1 - y0 > 4) {
        const ids = points.filter((p) => {
          const { sx, sy } = toScreen(p, canvas.clientWidth, canvas.clientHeight);
          return sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1;
        }).map((p) => p.id);
        onSelectBox(ids);
      }
      draw();
      return;
    }
    const wasDrag = dragRef.current?.moved;
    dragRef.current = null;
    if (wasDrag) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const p = findNearest(e.clientX - rect.left, e.clientY - rect.top);
    if (p) onSelect(p.id);
  };

  return (
    <div className="map-wrap">
      <canvas
        ref={canvasRef}
        className="map-canvas"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => { dragRef.current = null; boxRef.current = null; setHover(null); draw(); }}
      />
      {hover && (
        <div className="map-tooltip" style={{ left: hover.sx + 14, top: hover.sy + 14 }}>
          <img src={hover.point.thumb_url} alt="" />
        </div>
      )}
      <div className="map-hint">
        Each dot is one image, positioned by UMAP over SigLIP embeddings and colored by cluster.
        Scroll to zoom, drag to pan, click a point to open the sample,
        <strong> Shift+drag to select a region</strong> for bulk tagging.
      </div>
    </div>
  );
}
