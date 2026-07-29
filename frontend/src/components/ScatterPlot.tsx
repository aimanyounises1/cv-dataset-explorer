import { useCallback, useEffect, useRef, useState } from "react";
import type { MapPoint } from "../api/types";

interface Props {
  points: MapPoint[];
  onSelect: (id: number) => void;
  onSelectBox?: (ids: number[], aperture: MapAperture) => void;
  interactionMode?: MapInteractionMode;
  /** The last committed region, stored in projection coordinates rather than
   * screen pixels so it stays attached to the same points while the reader
   * pans, zooms, or resizes the window. */
  aperture?: MapAperture | null;
  /** The working-set count can be lower than the geometric count after a
   * reviewer drops images. The label reports the set actions will use. */
  apertureCount?: number;
  selectedIds?: Set<number>;
  /** Points to ring in ink, over whatever colour they already have — the map
   * side of a link with something outside the canvas (the working-set grid
   * points at one image; this is where it is). A ring rather than a fill
   * because on the paper canvas a light fill has nothing to contrast with.
   * Paint only: it reads the same projection every other dot does. */
  ringIds?: Set<number>;
  /** Colour per point, from lib/mapColor. Passed in rather than computed here so
   * the same scale drives the canvas and the legend beside it. */
  colorOf: (p: MapPoint) => string;
}

export type MapInteractionMode = "pan" | "select";

/** A rectangle in the map payload's normalized 0..1 projection space. */
export interface MapAperture {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

interface View { scale: number; tx: number; ty: number; }

type NavigationDirection = "up" | "down" | "left" | "right";

type PointerGesture =
  | {
      kind: "pan";
      pointerId: number;
      lastX: number;
      lastY: number;
      originX: number;
      originY: number;
      moved: boolean;
    }
  | { kind: "select"; pointerId: number };

const MAP_MARGIN = 24;
const MIN_APERTURE_PX = 4;
const POINTER_DRAG_THRESHOLD_PX = 3;
const KEYBOARD_PAN_PX = 36;
const KEYBOARD_ZOOM_FACTOR = 1.15;
const KEYBOARD_EDGE_PADDING_PX = 32;
const MIN_SCALE = 0.5;
const MAX_SCALE = 30;

function ordered(box: MapAperture): MapAperture {
  return {
    x0: Math.min(box.x0, box.x1),
    y0: Math.min(box.y0, box.y1),
    x1: Math.max(box.x0, box.x1),
    y1: Math.max(box.y0, box.y1),
  };
}

function includesPoint(box: MapAperture, point: MapPoint): boolean {
  const b = ordered(box);
  return point.x >= b.x0 && point.x <= b.x1
    && point.y >= b.y0 && point.y <= b.y1;
}

function imageCountLabel(count: number): string {
  return `${count.toLocaleString()} ${count === 1 ? "image" : "images"}`;
}

/** Canvas scatter of the embedding space. Scroll = zoom, drag = pan,
 * click = open sample. Select mode makes a drag create a persistent aperture;
 * Shift+drag remains the power-user path from Pan mode. */
export default function ScatterPlot({
  points, onSelect, onSelectBox, interactionMode = "pan",
  aperture = null, apertureCount, selectedIds, ringIds, colorOf,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const viewRef = useRef<View>({ scale: 1, tx: 0, ty: 0 });
  const gestureRef = useRef<PointerGesture | null>(null);
  const draftRef = useRef<MapAperture | null>(null);
  const [hover, setHover] = useState<{ point: MapPoint; sx: number; sy: number } | null>(null);
  const [keyboardFocusId, setKeyboardFocusId] = useState<number | null>(null);
  const [keyboardAnchorId, setKeyboardAnchorId] = useState<number | null>(null);
  const [canvasFocused, setCanvasFocused] = useState(false);
  const [liveAnnouncement, setLiveAnnouncement] = useState({
    sequence: 0,
    message: "",
  });

  const announce = useCallback((message: string) => {
    setLiveAnnouncement((current) => ({
      sequence: current.sequence + 1,
      message,
    }));
  }, []);

  const toScreen = useCallback((p: { x: number; y: number }, w: number, h: number) => {
    const { scale, tx, ty } = viewRef.current;
    const bx = MAP_MARGIN + p.x * (w - 2 * MAP_MARGIN);
    const by = MAP_MARGIN + (1 - p.y) * (h - 2 * MAP_MARGIN);
    return { sx: bx * scale + tx, sy: by * scale + ty };
  }, []);

  const fromScreen = useCallback((sx: number, sy: number, w: number, h: number) => {
    const { scale, tx, ty } = viewRef.current;
    const bx = (sx - tx) / scale;
    const by = (sy - ty) / scale;
    const innerW = Math.max(1, w - 2 * MAP_MARGIN);
    const innerH = Math.max(1, h - 2 * MAP_MARGIN);
    return {
      x: Math.min(1, Math.max(0, (bx - MAP_MARGIN) / innerW)),
      y: Math.min(1, Math.max(0, 1 - (by - MAP_MARGIN) / innerH)),
    };
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
    const styles = getComputedStyle(canvas);
    const stageInk = styles.getPropertyValue("--stage-text").trim() || "#edf3f0";
    const stageBackground = styles.getPropertyValue("--stage-bg").trim() || "#101915";
    const accent = styles.getPropertyValue("--accent").trim() || "#216d50";
    const r = Math.max(1.6, Math.min(3.2, 2.2 * viewRef.current.scale ** 0.5));
    for (const p of points) {
      const { sx, sy } = toScreen(p, w, h);
      if (sx < -5 || sy < -5 || sx > w + 5 || sy > h + 5) continue;
      const selected = selectedIds?.has(p.id);
      ctx.fillStyle = selected ? "#ffffff" : colorOf(p);
      ctx.globalAlpha = selected ? 1 : 0.85;
      ctx.beginPath();
      ctx.arc(sx, sy, selected ? r + 1 : r, 0, Math.PI * 2);
      ctx.fill();
      if (ringIds?.has(p.id)) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = stageInk;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(sx, sy, r + 4, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    /** The draft temporarily takes the committed aperture's place. An invalid
     * click therefore snaps back to the prior selection; a valid drag replaces
     * it on commit. */
    const keyboardAnchor = keyboardAnchorId === null
      ? null
      : points.find((point) => point.id === keyboardAnchorId) ?? null;
    const keyboardFocus = keyboardFocusId === null
      ? null
      : points.find((point) => point.id === keyboardFocusId) ?? null;
    const keyboardDraft = keyboardAnchor && keyboardFocus
      ? {
          x0: keyboardAnchor.x,
          y0: keyboardAnchor.y,
          x1: keyboardFocus.x,
          y1: keyboardFocus.y,
        }
      : null;
    const isDraft = draftRef.current !== null || keyboardDraft !== null;
    const shownAperture = draftRef.current ?? keyboardDraft ?? aperture;
    if (shownAperture) {
      const b = ordered(shownAperture);
      const a = toScreen({ x: b.x0, y: b.y0 }, w, h);
      const z = toScreen({ x: b.x1, y: b.y1 }, w, h);
      const x = Math.min(a.sx, z.sx);
      const y = Math.min(a.sy, z.sy);
      const bw = Math.abs(z.sx - a.sx);
      const bh = Math.abs(z.sy - a.sy);

      const surface = styles.getPropertyValue("--bg-raised").trim() || "#ffffff";
      const ink = styles.getPropertyValue("--text").trim() || "#17231e";
      const count = isDraft
        ? points.filter((point) => includesPoint(b, point)).length
        : apertureCount ?? points.filter((point) => includesPoint(b, point)).length;

      ctx.save();
      ctx.strokeStyle = accent;
      ctx.fillStyle = "rgba(33,109,80,0.12)";
      ctx.fillStyle = `color-mix(in srgb, ${accent} 12%, transparent)`;
      // An engine that cannot resolve color-mix keeps the fallback assignment
      // above, as specified for an invalid canvas colour value.
      ctx.lineWidth = isDraft ? 1.5 : 2;
      ctx.setLineDash(isDraft ? [5, 4] : []);
      ctx.fillRect(x, y, bw, bh);
      ctx.strokeRect(x, y, bw, bh);
      ctx.setLineDash([]);

      // Keep the label inside the visible canvas even when the aperture itself
      // is partly off-screen after a pan.
      if (x + bw >= 0 && y + bh >= 0 && x <= w && y <= h) {
        const label = `${count.toLocaleString()} selected`;
        ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
        const labelW = Math.ceil(ctx.measureText(label).width) + 12;
        const labelH = 22;
        const labelX = Math.min(
          Math.max(4, x),
          Math.max(4, w - labelW - 4),
        );
        const labelY = y >= labelH + 6
          ? y - labelH - 4
          : Math.min(Math.max(4, y + 4), Math.max(4, h - labelH - 4));
        ctx.fillStyle = surface;
        ctx.strokeStyle = accent;
        ctx.lineWidth = 1;
        ctx.fillRect(labelX, labelY, labelW, labelH);
        ctx.strokeRect(labelX, labelY, labelW, labelH);
        ctx.fillStyle = ink;
        ctx.textBaseline = "middle";
        ctx.fillText(label, labelX + 6, labelY + labelH / 2);
      }
      ctx.restore();
    }

    if (keyboardAnchor) {
      const { sx, sy } = toScreen(keyboardAnchor, w, h);
      ctx.save();
      ctx.translate(sx, sy);
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = accent;
      ctx.strokeStyle = stageInk;
      ctx.lineWidth = 2;
      ctx.fillRect(-5, -5, 10, 10);
      ctx.strokeRect(-5, -5, 10, 10);
      ctx.restore();
    }

    if (canvasFocused && keyboardFocus) {
      const { sx, sy } = toScreen(keyboardFocus, w, h);
      ctx.save();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = stageBackground;
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.arc(sx, sy, r + 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = stageInk;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(sx, sy, r + 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx - r - 12, sy);
      ctx.lineTo(sx + r + 12, sy);
      ctx.moveTo(sx, sy - r - 12);
      ctx.lineTo(sx, sy + r + 12);
      ctx.stroke();
      ctx.restore();
    }
  }, [
    points, toScreen, selectedIds, ringIds, colorOf, aperture, apertureCount,
    keyboardAnchorId, keyboardFocusId, canvasFocused,
  ]);

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
      const factor = e.deltaY < 0
        ? KEYBOARD_ZOOM_FACTOR
        : 1 / KEYBOARD_ZOOM_FACTOR;
      const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, v.scale * factor));
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

  useEffect(() => {
    setKeyboardFocusId((current) => (
      current !== null && points.some((point) => point.id === current)
        ? current
        : null
    ));
    setKeyboardAnchorId((current) => (
      interactionMode === "select"
      && current !== null
      && points.some((point) => point.id === current)
        ? current
        : null
    ));
  }, [points, interactionMode]);

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

  const describePoint = (point: MapPoint): string => (
    `image ${point.id}, cluster ${point.cluster}, ${point.split} split`
  );

  const findPointNearViewportCenter = (): MapPoint | null => {
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0) return null;
    const centerX = canvas.clientWidth / 2;
    const centerY = canvas.clientHeight / 2;
    let best = points[0];
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const point of points) {
      const { sx, sy } = toScreen(point, canvas.clientWidth, canvas.clientHeight);
      const distance = (sx - centerX) ** 2 + (sy - centerY) ** 2;
      if (distance < bestDistance) {
        best = point;
        bestDistance = distance;
      }
    }
    return best;
  };

  const resolveKeyboardPoint = (): MapPoint | null => (
    keyboardFocusId === null
      ? findPointNearViewportCenter()
      : points.find((point) => point.id === keyboardFocusId)
        ?? findPointNearViewportCenter()
  );

  const keepPointVisible = (point: MapPoint) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { sx, sy } = toScreen(point, canvas.clientWidth, canvas.clientHeight);
    const minX = Math.min(KEYBOARD_EDGE_PADDING_PX, canvas.clientWidth / 2);
    const minY = Math.min(KEYBOARD_EDGE_PADDING_PX, canvas.clientHeight / 2);
    const maxX = Math.max(minX, canvas.clientWidth - minX);
    const maxY = Math.max(minY, canvas.clientHeight - minY);
    if (sx < minX) viewRef.current.tx += minX - sx;
    if (sx > maxX) viewRef.current.tx -= sx - maxX;
    if (sy < minY) viewRef.current.ty += minY - sy;
    if (sy > maxY) viewRef.current.ty -= sy - maxY;
  };

  const focusKeyboardPoint = (point: MapPoint) => {
    keepPointVisible(point);
    setKeyboardFocusId(point.id);
    setHover(null);
    announce(`Focused ${describePoint(point)}.`);
  };

  const findDirectionalNeighbor = (
    current: MapPoint,
    direction: NavigationDirection,
  ): MapPoint | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const origin = toScreen(current, canvas.clientWidth, canvas.clientHeight);
    let best: MapPoint | null = null;
    let bestScore = Number.POSITIVE_INFINITY;
    for (const candidate of points) {
      if (candidate.id === current.id) continue;
      const target = toScreen(candidate, canvas.clientWidth, canvas.clientHeight);
      const dx = target.sx - origin.sx;
      const dy = target.sy - origin.sy;
      const primary = direction === "right" ? dx
        : direction === "left" ? -dx
          : direction === "down" ? dy
            : -dy;
      if (primary <= 0) continue;
      const perpendicular = direction === "left" || direction === "right"
        ? Math.abs(dy)
        : Math.abs(dx);
      const distance = Math.hypot(dx, dy);
      const score = distance * (1 + (2 * perpendicular) / primary);
      if (
        score < bestScore
        || (score === bestScore && best !== null && candidate.id < best.id)
      ) {
        best = candidate;
        bestScore = score;
      }
    }
    return best;
  };

  const zoomAtKeyboardFocus = (factor: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const focusedPoint = resolveKeyboardPoint();
    const center = focusedPoint
      ? toScreen(focusedPoint, canvas.clientWidth, canvas.clientHeight)
      : { sx: canvas.clientWidth / 2, sy: canvas.clientHeight / 2 };
    const view = viewRef.current;
    const nextScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, view.scale * factor));
    view.tx = center.sx - ((center.sx - view.tx) / view.scale) * nextScale;
    view.ty = center.sy - ((center.sy - view.ty) / view.scale) * nextScale;
    view.scale = nextScale;
    setHover(null);
    draw();
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!event.isPrimary || event.button !== 0) return;
    event.preventDefault();
    const canvas = event.currentTarget;
    canvas.focus({ preventScroll: true });
    setCanvasFocused(false);
    canvas.setPointerCapture(event.pointerId);
    if (keyboardAnchorId !== null) setKeyboardAnchorId(null);

    const rect = canvas.getBoundingClientRect();
    const sx = event.clientX - rect.left;
    const sy = event.clientY - rect.top;
    if (onSelectBox && (interactionMode === "select" || event.shiftKey)) {
      const point = fromScreen(sx, sy, canvas.clientWidth, canvas.clientHeight);
      draftRef.current = {
        x0: point.x,
        y0: point.y,
        x1: point.x,
        y1: point.y,
      };
      gestureRef.current = { kind: "select", pointerId: event.pointerId };
    } else {
      gestureRef.current = {
        kind: "pan",
        pointerId: event.pointerId,
        lastX: event.clientX,
        lastY: event.clientY,
        originX: event.clientX,
        originY: event.clientY,
        moved: false,
      };
    }
    setHover(null);
    draw();
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = event.currentTarget;
    const gesture = gestureRef.current;
    const rect = canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;

    if (gesture?.pointerId === event.pointerId) {
      event.preventDefault();
      if (gesture.kind === "select" && draftRef.current) {
        const point = fromScreen(mx, my, canvas.clientWidth, canvas.clientHeight);
        draftRef.current.x1 = point.x;
        draftRef.current.y1 = point.y;
      } else if (gesture.kind === "pan") {
        const dx = event.clientX - gesture.lastX;
        const dy = event.clientY - gesture.lastY;
        gesture.moved = gesture.moved
          || Math.hypot(
            event.clientX - gesture.originX,
            event.clientY - gesture.originY,
          ) > POINTER_DRAG_THRESHOLD_PX;
        viewRef.current.tx += dx;
        viewRef.current.ty += dy;
        gesture.lastX = event.clientX;
        gesture.lastY = event.clientY;
      }
      setHover(null);
      draw();
      return;
    }

    if (event.pointerType === "touch") return;
    const point = findNearest(mx, my);
    if (!point) {
      setHover(null);
      return;
    }
    const screen = toScreen(point, canvas.clientWidth, canvas.clientHeight);
    setHover({ point, sx: screen.sx, sy: screen.sy });
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = event.currentTarget;
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    event.preventDefault();

    if (gesture.kind === "select" && draftRef.current && onSelectBox) {
      const rect = canvas.getBoundingClientRect();
      const endpoint = fromScreen(
        event.clientX - rect.left,
        event.clientY - rect.top,
        canvas.clientWidth,
        canvas.clientHeight,
      );
      draftRef.current.x1 = endpoint.x;
      draftRef.current.y1 = endpoint.y;
      const b = ordered(draftRef.current);
      draftRef.current = null;
      gestureRef.current = null;
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      const a = toScreen({ x: b.x0, y: b.y0 }, canvas.clientWidth, canvas.clientHeight);
      const z = toScreen({ x: b.x1, y: b.y1 }, canvas.clientWidth, canvas.clientHeight);
      if (
        Math.abs(z.sx - a.sx) > MIN_APERTURE_PX
        && Math.abs(z.sy - a.sy) > MIN_APERTURE_PX
      ) {
        const ids = points
          .filter((point) => includesPoint(b, point))
          .map((point) => point.id);
        onSelectBox(ids, b);
      } else {
        // Restore the last committed aperture after an accidental click.
        draw();
      }
      return;
    }

    const wasDrag = gesture.kind === "pan" && gesture.moved;
    gestureRef.current = null;
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    if (wasDrag) return;
    const rect = canvas.getBoundingClientRect();
    const point = findNearest(event.clientX - rect.left, event.clientY - rect.top);
    if (point) {
      setKeyboardFocusId(point.id);
      onSelect(point.id);
    }
  };

  const cancelPointerGesture = (pointerId: number) => {
    if (gestureRef.current?.pointerId !== pointerId) return;
    gestureRef.current = null;
    draftRef.current = null;
    setHover(null);
    draw();
  };

  const handleCanvasFocus = (event: React.FocusEvent<HTMLCanvasElement>) => {
    setCanvasFocused(event.currentTarget.matches(":focus-visible"));
    const point = resolveKeyboardPoint();
    if (point) {
      setKeyboardFocusId(point.id);
      announce(`Focused ${describePoint(point)}.`);
    } else {
      announce("The embedding map has no points to focus.");
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLCanvasElement>) => {
    setCanvasFocused(true);
    const direction: NavigationDirection | null =
      event.key === "ArrowUp" ? "up"
        : event.key === "ArrowDown" ? "down"
          : event.key === "ArrowLeft" ? "left"
            : event.key === "ArrowRight" ? "right"
              : null;

    if (event.key === "Escape" && keyboardAnchorId !== null) {
      event.preventDefault();
      event.stopPropagation();
      setKeyboardAnchorId(null);
      announce("Selection anchor cancelled.");
      return;
    }

    if (direction && event.shiftKey) {
      event.preventDefault();
      if (direction === "left") viewRef.current.tx -= KEYBOARD_PAN_PX;
      if (direction === "right") viewRef.current.tx += KEYBOARD_PAN_PX;
      if (direction === "up") viewRef.current.ty -= KEYBOARD_PAN_PX;
      if (direction === "down") viewRef.current.ty += KEYBOARD_PAN_PX;
      setHover(null);
      draw();
      return;
    }

    if (direction) {
      event.preventDefault();
      const current = resolveKeyboardPoint();
      if (!current) {
        announce("The embedding map has no points to focus.");
        return;
      }
      const next = findDirectionalNeighbor(current, direction);
      if (next) {
        focusKeyboardPoint(next);
      } else {
        announce(`No point lies farther ${direction} of ${describePoint(current)}.`);
      }
      return;
    }

    if (
      event.key === "+"
      || event.key === "="
      || event.key === "Add"
    ) {
      event.preventDefault();
      zoomAtKeyboardFocus(KEYBOARD_ZOOM_FACTOR);
      return;
    }
    if (
      event.key === "-"
      || event.key === "_"
      || event.key === "Subtract"
    ) {
      event.preventDefault();
      zoomAtKeyboardFocus(1 / KEYBOARD_ZOOM_FACTOR);
      return;
    }
    if (event.key === "Home" || event.key === "0") {
      event.preventDefault();
      viewRef.current = { scale: 1, tx: 0, ty: 0 };
      setHover(null);
      draw();
      return;
    }
    if (event.key !== "Enter") return;

    event.preventDefault();
    const focusedPoint = resolveKeyboardPoint();
    if (!focusedPoint) {
      announce("The embedding map has no points to activate.");
      return;
    }
    setKeyboardFocusId(focusedPoint.id);

    if (interactionMode === "pan") {
      announce(`Opening ${describePoint(focusedPoint)}.`);
      onSelect(focusedPoint.id);
      return;
    }
    if (!onSelectBox) {
      announce("Region selection is unavailable.");
      return;
    }
    if (keyboardAnchorId === null) {
      setKeyboardAnchorId(focusedPoint.id);
      announce(`Selection anchor set at ${describePoint(focusedPoint)}.`);
      return;
    }

    const anchor = points.find((point) => point.id === keyboardAnchorId);
    if (!anchor) {
      setKeyboardAnchorId(focusedPoint.id);
      announce(`Selection anchor set at ${describePoint(focusedPoint)}.`);
      return;
    }
    const nextAperture = ordered({
      x0: anchor.x,
      y0: anchor.y,
      x1: focusedPoint.x,
      y1: focusedPoint.y,
    });
    const ids = points
      .filter((point) => includesPoint(nextAperture, point))
      .map((point) => point.id);
    setKeyboardAnchorId(null);
    onSelectBox(ids, nextAperture);
    announce(
      `Aperture committed from image ${anchor.id} to image ${focusedPoint.id}; `
      + `${imageCountLabel(ids.length)} selected.`,
    );
  };

  return (
    <div className="map-wrap">
      <canvas
        ref={canvasRef}
        className="map-canvas"
        data-mode={interactionMode}
        role="application"
        tabIndex={0}
        aria-label={
          interactionMode === "select"
            ? "Embedding map, Select mode. Drag to select. Arrow keys focus points; Enter anchors a corner, and a second Enter selects. Shift plus arrows pan; plus or minus zoom; Home or 0 resets."
            : "Embedding map, Pan mode. Drag to pan or tap a point. Arrow keys focus points; Enter opens one. Shift plus arrows pan; plus or minus zoom; Home or 0 resets."
        }
        aria-describedby="map-interaction-hint"
        style={{ cursor: interactionMode === "select" ? "crosshair" : undefined }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={(event) => cancelPointerGesture(event.pointerId)}
        onLostPointerCapture={(event) => cancelPointerGesture(event.pointerId)}
        onPointerLeave={() => {
          if (gestureRef.current === null) setHover(null);
        }}
        onFocus={handleCanvasFocus}
        onBlur={() => {
          setCanvasFocused(false);
          setHover(null);
        }}
        onKeyDown={handleKeyDown}
      >
        Embedding map of {points.length.toLocaleString()} images. Use the filters
        and working-set controls for a structured alternative.
      </canvas>
      {hover && (
        <div className="map-tooltip" style={{ left: hover.sx + 14, top: hover.sy + 14 }}>
          <img src={hover.point.thumb_url} alt="" />
        </div>
      )}
      <span className="sr-only" aria-live="polite">
        {aperture
          ? `${imageCountLabel(apertureCount ?? 0)} selected in the map aperture.`
          : "No map aperture selected."}
      </span>
      <span className="sr-only" aria-live="polite" aria-atomic="true">
        <span key={liveAnnouncement.sequence}>{liveAnnouncement.message}</span>
      </span>
      {/* Deliberately does not say what the colours mean — the legend above the
          canvas does, and it changes with the selected dimension. */}
      <div className="map-hint" id="map-interaction-hint">
        Each dot is one image, positioned by UMAP over SigLIP embeddings.
        {interactionMode === "select" ? (
          <> Drag to draw an aperture. Keyboard: arrows move between points;
            Enter anchors, then commits; Shift+arrows pan. Escape cancels an
            anchor, then returns to Pan.</>
        ) : (
          <> Scroll to zoom, drag to pan, tap a point to open it, or
            <strong> Shift+drag to select</strong>. Keyboard: arrows move between
            points; Enter opens; Shift+arrows pan; +/- zoom; Home or 0 resets.</>
        )}
      </div>
    </div>
  );
}
