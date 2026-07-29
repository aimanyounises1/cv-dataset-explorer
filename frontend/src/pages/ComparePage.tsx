import {
  useCallback, useEffect, useRef, useState,
} from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import {
  AXES, LeakagePair, SampleCard, SegmentAnnotation, SegmentBox,
} from "../api/types";
import ImageCard from "../components/ImageCard";
import { useActiveProviderName, useCorpusTotal } from "../lib/activeProvider";
import "../styles/compare.css";

/**
 * Compare & focus: two frames on one bench, under one loupe.
 *
 * The pair lives in the URL (`?a=&b=`) because a comparison is a claim someone
 * will want to hand to someone else. The zoom/pan transform deliberately does
 * NOT: it is the reader's hands on the loupe, meaningful only to whoever is
 * holding it right now — the same reason image-query results stay in memory.
 * One transform drives both panes, so "look at the third dog from the left"
 * means the same crop on each side.
 *
 * Two things this page deliberately does NOT own. Retrieval: a region drawn
 * here is searched through the same `/search/by-region` the sample page calls,
 * so one rectangle has one ranking whichever screen drew it. Annotation: the
 * saved outlines are read-only here — they are made, labelled and deleted on
 * /samples/:id, which has the taxonomy, the masks and the delete.
 */

// Must match AlbumShelf's DRAG_IDS. Declared locally rather than imported so
// this lazy chunk does not pull the shelf along for one string.
const DRAG_IDS = "application/x-cvde-ids";


const MIN_SCALE = 0.5;
const MAX_SCALE = 12;

/** How many neighbours a region query returns. One strip's worth: the point of
 * searching a crop here is "is this thing rare?", which the first screenful
 * answers — the gallery is where a full ranking gets paged. */
const REGION_TOP_K = 24;

/** Where the pair queue cuts, and how many it offers.
 *
 * 0.90 is the split-integrity panel's own default threshold, so the queue and
 * the count a reader just saw there describe the same pairs — a different cut
 * here would quietly be a different population. Ten is one screenful: this is a
 * way in, not a worklist, and the integrity view owns the full set. */
const LEAKAGE_THRESHOLD = 0.9;
const PAIR_QUEUE_N = 10;

const STAGE_TITLE =
  "Wheel zooms toward the cursor, drag pans, double-click resets — both panes "
  + "move together. Keyboard: + / - zoom, arrow keys pan, 0 resets. The "
  + "magnification in each pane header is a button: it snaps to 1:1, one "
  + "source pixel per screen pixel.";

interface SampleDetail {
  id: number;
  filename: string;
  split: string;
  width: number;
  height: number;
  image_url: string;
  thumb_url: string;
  vlm_tags: string[];
  attributes: Record<string, string>;
  cluster: number | null;
  axes: Record<string, number | object | null>;
}
interface View { s: number; tx: number; ty: number }

/** What one region query produced. `scoreBasis` travels with the items because
 * a score is uninterpretable without it — the same rule the gallery follows. */
interface RegionResult {
  fromId: number;
  role: "positive" | "negative";
  items: SampleCard[];
  scoreBasis: string | null;
  message: string | null;
}

const REST_VIEW: View = { s: 1, tx: 0, ty: 0 };

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} on ${path}`);
  return r.json() as Promise<T>;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Zoom about a point given in the layer's own (untransformed) coordinates:
 * with transform-origin 0 0, keeping p fixed on screen reduces to
 * t' = t + (s - s')·p. */
function zoomedView(v: View, factor: number, px: number, py: number): View {
  const s = clamp(v.s * factor, MIN_SCALE, MAX_SCALE);
  return { s, tx: v.tx + (v.s - s) * px, ty: v.ty + (v.s - s) * py };
}

function axisValue(sample: SampleDetail, axis: string): number | null {
  const v = sample.axes[axis];
  return typeof v === "number" ? v : null;
}

/** The view that shows actual pixels: one source pixel on one CSS pixel, with
 * the layer's centre back where it sits at rest. With transform-origin 0 0 a
 * layer point p lands at t + s·p, so the centre holds when
 * t = (1 − s)·(layer size / 2) — the same algebra as `zoomedView`, pivoting on
 * the middle. `layerW` is the layout width, which is viewport-dependent: that
 * is exactly why the scale that means 1:1 cannot be a constant. */
function actualPixelView(layerW: number, layerH: number, sourceW: number): View {
  const s = clamp(sourceW / layerW, MIN_SCALE, MAX_SCALE);
  return { s, tx: (1 - s) * layerW / 2, ty: (1 - s) * layerH / 2 };
}

function useSample(id: number | null) {
  const [state, setState] = useState<{ data: SampleDetail | null; error: string | null }>(
    { data: null, error: null });
  useEffect(() => {
    if (id == null) { setState({ data: null, error: null }); return; }
    let live = true;
    setState({ data: null, error: null });
    getJSON<SampleDetail>(`/api/samples/${id}`)
      .then((d) => { if (live) setState({ data: d, error: null }); })
      .catch((e: unknown) => { if (live) setState({ data: null, error: String(e) }); });
    return () => { live = false; };
  }, [id]);
  return state;
}

function idFromParam(raw: string | null): number | null {
  if (!raw) return null;
  const n = Number(raw);
  return Number.isInteger(n) && n > 0 ? n : null;
}

// ------------------------------------------------------------------- the page

export default function ComparePage() {
  const [params, setParams] = useSearchParams();
  const aId = idFromParam(params.get("a"));
  const bId = idFromParam(params.get("b"));
  const a = useSample(aId);
  const b = useSample(bId);

  const [view, setView] = useState<View>(REST_VIEW);
  useEffect(() => { setView(REST_VIEW); }, [aId, bId]);

  // Cosine between the pair, read off a's neighbour list rather than computed
  // fresh: same embeddings, same index, same number the rest of the app shows.
  // null = unknown (loading, or the fetch failed) — then the line is omitted;
  // "absent" = b is genuinely not among a's top-60, which is itself a finding.
  const [sim, setSim] = useState<{ score: number; rank: number } | "absent" | null>(null);
  useEffect(() => {
    if (aId == null || bId == null) { setSim(null); return; }
    let live = true;
    setSim(null);
    getJSON<SampleCard[]>(`/api/samples/${aId}/similar?top_k=60`)
      .then((cards) => {
        if (!live) return;
        const i = cards.findIndex((c) => c.id === bId);
        const hit = i >= 0 ? cards[i] : undefined;
        setSim(hit && hit.score != null ? { score: hit.score, rank: i + 1 } : "absent");
      })
      .catch(() => { /* unknown stays unknown — no line beats a fake number */ });
    return () => { live = false; };
  }, [aId, bId]);

  const [regionHits, setRegionHits] = useState<RegionResult | null>(null);
  const [searching, setSearching] = useState(false);
  const [regionErr, setRegionErr] = useState<string | null>(null);

  /** One region, one ranking implementation. The geometry goes to the server,
   * which crops the ORIGINAL file — the same `/search/by-region` the sample
   * page uses. Cropping in a canvas here instead would re-encode the pixels
   * before embedding them, and the same rectangle would score differently
   * depending on which screen it was drawn on. */
  const searchRegion = useCallback(async (
    sample: SampleDetail, rect: SegmentBox,
    role: "positive" | "negative", text: string,
  ) => {
    setSearching(true);
    setRegionErr(null);
    try {
      const resp = await api.searchByRegion({
        sample_id: sample.id, ...rect, role, top_k: REGION_TOP_K,
        // Steering text blends into the crop's vector on the positive path
        // only; sending it with a negative role would imply an effect the
        // ranking does not have.
        text: role === "positive" && text.trim() ? text.trim() : undefined,
      });
      setRegionHits({
        fromId: sample.id,
        role,
        items: resp.items,
        scoreBasis: resp.score_basis ?? null,
        message: resp.message ?? null,
      });
    } catch (e: unknown) {
      setRegionHits(null);
      setRegionErr(String(e));
    } finally {
      setSearching(false);
    }
  }, []);

  const fill = useCallback((slot: "a" | "b", ids: number[]) => {
    if (ids.length === 0) return;
    setParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set(slot, String(ids[0]));
      // A card dragged mid-selection carries the whole picked set; if the
      // other seat is free, the second id takes it rather than being dropped.
      const other = slot === "a" ? "b" : "a";
      if (ids.length > 1 && !next.get(other)) next.set(other, String(ids[1]));
      return next;
    }, { replace: true });
  }, [setParams]);

  // Keyboard drives the same shared transform. Zoom pivots on the first
  // pane's centre — with no cursor there is no better anchor.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
      const zoom = (factor: number) => setView((v) => {
        const layer = document.querySelector<HTMLElement>(".compare-img-layer");
        if (!layer) return v;
        const r = layer.getBoundingClientRect();
        return zoomedView(v, factor, r.width / 2 / v.s, r.height / 2 / v.s);
      });
      const pan = (dx: number, dy: number) =>
        setView((v) => ({ ...v, tx: v.tx + dx, ty: v.ty + dy }));
      switch (e.key) {
        case "+": case "=": zoom(1.25); break;
        case "-": case "_": zoom(0.8); break;
        case "0": setView(REST_VIEW); break;
        case "ArrowLeft": pan(40, 0); break;
        case "ArrowRight": pan(-40, 0); break;
        case "ArrowUp": pan(0, 40); break;
        case "ArrowDown": pan(0, -40); break;
        default: return;
      }
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="compare-page">
      <div className="eyebrow">Compare &amp; focus</div>
      <h1 className="section-title compare-title">Two frames, one loupe</h1>

      {/* With nothing loaded the bench is two empty rectangles asking the reader
          to already know two ids. The queue leads in that state and the panes
          shrink to a drop target beneath it; once a pair is open the panes are
          the whole point again and the queue is gone. */}
      {aId == null && bId == null && <PairQueue />}

      <div className={`compare-panes${aId == null && bId == null ? " idle" : ""}`}>
        <Pane slot="a" sample={a.data} error={a.error} view={view} setView={setView}
          onFill={fill} searching={searching} onSearchRegion={searchRegion} />
        <Pane slot="b" sample={b.data} error={b.error} view={view} setView={setView}
          onFill={fill} searching={searching} onSearchRegion={searchRegion} />
      </div>

      {regionErr && <div className="error">{regionErr}</div>}
      {(searching || regionHits) && (
        <section className="panel region-results">
          <h3>
            Region search
            {regionHits && !searching && (regionHits.role === "positive"
              ? ` — nearest ${regionHits.items.length} to a region of #${regionHits.fromId},`
                + " cropped from the original"
              : ` — the ${regionHits.items.length} frames farthest from a region of `
                + `#${regionHits.fromId}`)}
          </h3>
          {searching
            ? <div className="loading">Embedding the crop and ranking the corpus…</div>
            : regionHits && (
              <>
                {/* The server's own sentence about what this ranking is,
                    verbatim — the same line the sample page shows. */}
                {regionHits.message && <p className="compare-sim">{regionHits.message}</p>}
                <div className="grid">
                  {regionHits.items.map((c) => (
                    <ImageCard key={c.id} sample={c} scoreBasis={regionHits.scoreBasis} />
                  ))}
                </div>
              </>
            )}
        </section>
      )}

      {a.data && b.data && <DiffPanel a={a.data} b={b.data} sim={sim} />}
    </div>
  );
}

// ------------------------------------------------------------------ the panes

interface PaneProps {
  slot: "a" | "b";
  sample: SampleDetail | null;
  error: string | null;
  view: View;
  setView: React.Dispatch<React.SetStateAction<View>>;
  onFill: (slot: "a" | "b", ids: number[]) => void;
  searching: boolean;
  onSearchRegion: (
    sample: SampleDetail, rect: SegmentBox,
    role: "positive" | "negative", text: string,
  ) => void;
}

function Pane({
  slot, sample, error, view, setView, onFill, searching, onSearchRegion,
}: PaneProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const layerRef = useRef<HTMLDivElement | null>(null);
  const gestureRef = useRef<
    | { mode: "pan"; lastX: number; lastY: number }
    | { mode: "draw"; ox: number; oy: number }
    | null
  >(null);

  const [over, setOver] = useState(false);
  const [drawMode, setDrawMode] = useState(false);
  const [draft, setDraft] = useState<SegmentBox | null>(null);
  const [steer, setSteer] = useState("");
  const [saved, setSaved] = useState<SegmentAnnotation[]>([]);

  // The layer's layout width and height at scale 1 — the denominator of the
  // magnification readout and what 1:1 solves against. Observed, not assumed:
  // the same 500px-wide photograph lays out at 565 CSS px on a 1440 desktop
  // and 356 on a 390 phone, so `view.s === 1` is not the same magnification
  // twice.
  const [layerBox, setLayerBox] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  const sampleId = sample ? sample.id : null;

  // A new occupant clears the previous one's marks.
  useEffect(() => {
    setDraft(null); setSteer(""); setDrawMode(false);
  }, [sampleId]);

  useEffect(() => {
    const el = layerRef.current;
    if (!el) { setLayerBox({ w: 0, h: 0 }); return; }
    const read = () => setLayerBox({ w: el.offsetWidth, h: el.offsetHeight });
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [sampleId]);

  // The marks the sample page owns, shown here read-only: accepted segment
  // annotations, outlined by their bounding boxes. Compare does not annotate —
  // the label taxonomy, the mask, the delete and search-by-annotation all live
  // on /samples/:id, and a second, weaker annotator here would only write rows
  // that screen cannot see, edit or delete.
  useEffect(() => {
    setSaved([]);
    if (sampleId == null) return;
    const ctrl = new AbortController();
    api.listSegmentAnnotations(sampleId, ctrl.signal)
      .then((list) => { setSaved(list.filter((x) => x.geometry != null)); })
      .catch(() => { /* an aborted or unavailable read draws no marks, which
                        is what "this image has none" already looks like */ });
    return () => ctrl.abort();
  }, [sampleId]);

  // Wheel must be a native non-passive listener: React delegates wheel as
  // passive, so preventDefault (keeping the page still under the loupe) would
  // be ignored on the synthetic path.
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const layer = layerRef.current;
      if (!layer) return;
      const r = layer.getBoundingClientRect();
      setView((v) => zoomedView(
        v,
        Math.exp(-e.deltaY * 0.0015),
        (e.clientX - r.left) / v.s,
        (e.clientY - r.top) / v.s,
      ));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [sampleId, setView]);

  // Pointer position in image-normalized 0..1 coordinates. The layer's border
  // box IS the image content box (same aspect ratio, see the stylesheet), and
  // getBoundingClientRect already includes the shared transform.
  const normPoint = (e: React.PointerEvent): { x: number; y: number } | null => {
    const layer = layerRef.current;
    if (!layer) return null;
    const r = layer.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return null;
    return {
      x: clamp((e.clientX - r.left) / r.width, 0, 1),
      y: clamp((e.clientY - r.top) / r.height, 0, 1),
    };
  };

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    if (drawMode) {
      const p = normPoint(e);
      if (!p) return;
      gestureRef.current = { mode: "draw", ox: p.x, oy: p.y };
      setDraft({ x: p.x, y: p.y, w: 0, h: 0 });
    } else {
      gestureRef.current = { mode: "pan", lastX: e.clientX, lastY: e.clientY };
    }
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const g = gestureRef.current;
    if (!g) return;
    if (g.mode === "pan") {
      const dx = e.clientX - g.lastX;
      const dy = e.clientY - g.lastY;
      g.lastX = e.clientX;
      g.lastY = e.clientY;
      setView((v) => ({ ...v, tx: v.tx + dx, ty: v.ty + dy }));
    } else {
      const p = normPoint(e);
      if (!p) return;
      setDraft({
        x: Math.min(g.ox, p.x),
        y: Math.min(g.oy, p.y),
        w: Math.abs(p.x - g.ox),
        h: Math.abs(p.y - g.oy),
      });
    }
  };

  const endGesture = () => {
    const g = gestureRef.current;
    gestureRef.current = null;
    if (g && g.mode === "draw") {
      // A click is not a region; below ~1% of the frame there is nothing to crop.
      setDraft((d) => (d && d.w > 0.01 && d.h > 0.01 ? d : null));
    }
  };

  const acceptDrag = (e: React.DragEvent) => {
    if (Array.from(e.dataTransfer.types).includes(DRAG_IDS)) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
      setOver(true);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    setOver(false);
    const raw = e.dataTransfer.getData(DRAG_IDS);
    if (!raw) return;
    e.preventDefault();
    try {
      const ids = JSON.parse(raw) as unknown;
      if (Array.isArray(ids)) {
        onFill(slot, ids.filter((n): n is number => typeof n === "number"));
      }
    } catch { /* some other drag's payload — not ours to interpret */ }
  };

  // Displayed pixels per source pixel — the number a loupe exists to make
  // answerable ("is that blocking eight source pixels wide, or two?").
  // `view.s` cannot answer it: at rest it is 1 whatever the layout did.
  const mag = sample && layerBox.w > 0 ? (layerBox.w * view.s) / sample.width : null;

  const pct = (r: SegmentBox): React.CSSProperties => ({
    left: `${r.x * 100}%`,
    top: `${r.y * 100}%`,
    width: `${r.w * 100}%`,
    height: `${r.h * 100}%`,
  });

  return (
    <section
      className={`compare-pane${sample ? "" : " empty"}${over ? " drop" : ""}`}
      data-slot={slot}
      onDragOver={acceptDrag}
      onDragEnter={acceptDrag}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
    >
      {!sample && !error && (
        <p className="compare-empty-msg">
          Drop an image here, or pick two in the gallery and press Compare.
        </p>
      )}
      {!sample && error && <div className="error">{error}</div>}
      {sample && (
        <>
          <header className="pane-head">
            <span className="pane-id">{slot} · #{sample.id}</span>
            <Link to={`/samples/${sample.id}`} title="Open in the inspector">
              {sample.filename}
            </Link>
            <span className="pill">{sample.split}</span>
            {/* The magnification is the readout and the control at once: it
                states the ratio at rest, and clicking it snaps to 1:1. It
                borrows `.legend-pick`, which exists for exactly this — a
                fact first, a control second, its border arriving on hover. */}
            {mag != null && (
              <button
                type="button"
                className="pane-id legend-pick"
                title={`${mag.toFixed(2)} screen pixels per source pixel of the `
                  + `${sample.width}×${sample.height} original. Click for 1:1 — `
                  + "actual pixels, recentred. Both panes share one transform, "
                  + "so the other lands at its own ratio."}
                onClick={() => setView(
                  actualPixelView(layerBox.w, layerBox.h, sample.width))}
              >
                {mag.toFixed(2)}×
              </button>
            )}
            <button
              type="button"
              className={`ghost draw-toggle${drawMode ? " select-on" : ""}`}
              aria-pressed={drawMode}
              title={"Drag on the image to mark a rectangle — drawn by you, no "
                + "segmentation model involved — and search the corpus with it. "
                + "Sage outlines already on the image are saved segment annotations, "
                + "which are made and edited on the sample page."}
              onClick={() => setDrawMode((d) => !d)}
            >
              Draw region
            </button>
          </header>
          <div
            ref={stageRef}
            className={`pane-stage${drawMode ? " drawing" : ""}`}
            title={STAGE_TITLE}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endGesture}
            onPointerCancel={endGesture}
            onDoubleClick={() => { if (!drawMode) setView(REST_VIEW); }}
          >
            <div
              ref={layerRef}
              className="compare-img-layer"
              style={{
                aspectRatio: `${sample.width} / ${sample.height}`,
                transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.s})`,
              }}
            >
              <img src={sample.image_url} alt={sample.filename} draggable={false} />
              {saved.map((ann) => (
                <div key={ann.id} className="region-rect saved" style={pct(ann.geometry)}>
                  {(ann.label_name ?? ann.label)
                    && <span className="region-label">{ann.label_name ?? ann.label}</span>}
                </div>
              ))}
              {draft && <div className="region-rect draft" style={pct(draft)} />}
            </div>
          </div>
          {draft && (
            <div className="region-actions">
              {/* The same steering the composed query takes everywhere else:
                  the crop's vector averaged with this text. Empty means the
                  crop alone. */}
              <input
                value={steer}
                onChange={(e) => setSteer(e.target.value)}
                placeholder="steer with text (optional)"
                aria-label={`Steering text for the region on sample ${sample.id}`}
              />
              <button type="button" className="primary" disabled={searching}
                title={"Rank the corpus by closeness to this region — the server crops "
                  + "the original, so the same rectangle gives the same answer here and "
                  + "on the sample page"}
                onClick={() => onSearchRegion(sample, draft, "positive", steer)}>
                Search this region
              </button>
              <button type="button" className="ghost" disabled={searching}
                title={"Rank by distance from this region — what the corpus has least "
                  + "of. Steering text does not apply to this ranking."}
                onClick={() => onSearchRegion(sample, draft, "negative", steer)}>
                Away
              </button>
              <button type="button" className="ghost region-clear"
                title="Discard this rectangle"
                onClick={() => { setDraft(null); setSteer(""); }}>
                ✕
              </button>
            </div>
          )}
        </>
      )}
    </section>
  );
}

// ------------------------------------------------- shared / different readout

interface DiffPanelProps {
  a: SampleDetail;
  b: SampleDetail;
  sim: { score: number; rank: number } | "absent" | null;
}

function DiffPanel({ a, b, sim }: DiffPanelProps) {
  const providerName = useActiveProviderName();
  const corpusTotal = useCorpusTotal();
  const row = (name: string, va: string | null, vb: string | null) => ({
    name, va, vb, same: va != null && va === vb,
  });
  const groups = Array.from(new Set([
    ...Object.keys(a.attributes), ...Object.keys(b.attributes),
  ])).sort();
  const rows = [
    row("split", a.split, b.split),
    row("cluster",
      a.cluster == null ? null : `c${a.cluster}`,
      b.cluster == null ? null : `c${b.cluster}`),
    ...groups.map((g) => row(g, a.attributes[g] ?? null, b.attributes[g] ?? null)),
  ];

  const shared = a.vlm_tags.filter((t) => b.vlm_tags.includes(t));
  const onlyA = a.vlm_tags.filter((t) => !b.vlm_tags.includes(t));
  const onlyB = b.vlm_tags.filter((t) => !a.vlm_tags.includes(t));

  return (
    <section className="panel compare-diff">
      <h3>Shared &amp; different</h3>
      {sim !== null && (
        <p className="compare-sim">
          {sim === "absent"
            ? `#${b.id} is not in #${a.id}'s top-60 neighbours — no cosine to report`
            : `cosine ${sim.score.toFixed(4)} — #${b.id} is #${a.id}'s neighbour `
              + `${sim.rank} of 60 (${providerName} image embeddings)`}
        </p>
      )}

      <div className="diff-rows">
        {rows.map((r) => (
          <div key={r.name} className={`diff-row ${r.same ? "same" : "differs"}`}>
            <span className="diff-name">{r.name}</span>
            <span className="diff-verdict">{r.same ? "shared" : "differs"}</span>
            <span className="diff-value mono">
              {r.same ? r.va : `${r.va ?? "—"} vs ${r.vb ?? "—"}`}
            </span>
          </div>
        ))}
      </div>

      <div className="diff-tags">
        {a.vlm_tags.length === 0 && b.vlm_tags.length === 0
          ? <p className="diff-tags-none">No VLM tags on either sample.</p>
          : (
            <>
              <TagLine name="shared tags" tags={shared} empty="none shared" />
              <TagLine name={`only #${a.id}`} tags={onlyA} empty="none" />
              <TagLine name={`only #${b.id}`} tags={onlyB} empty="none" />
            </>
          )}
      </div>

      <div className="axis-pairs">
        <p className="axis-pair-key">
          <span className="key a" /> a = #{a.id}
          <span className="key b" /> b = #{b.id}
        </p>
        {AXES.map((axis) => {
          const va = axisValue(a, axis);
          const vb = axisValue(b, axis);
          return (
            <div key={axis} className="axis-pair">
              <span className="axis-pair-name">{axis}</span>
              <div className="axis-pair-bars">
                <span className="bar-track">
                  <span className="bar-fill a"
                    style={{ width: `${(va ?? 0) * 10}%` }} />
                </span>
                <span className="bar-track">
                  <span className="bar-fill b"
                    style={{ width: `${(vb ?? 0) * 10}%` }} />
                </span>
              </div>
              <span className="axis-pair-vals mono">
                {va ?? "—"} vs {vb ?? "—"}
              </span>
            </div>
          );
        })}
        <p className="axis-pair-note">
          Percentile ranks over the corpus, 0–10 — a longer bar is harder,
          rarer, or busier, relative to the other{" "}
          {corpusTotal != null ? (corpusTotal - 1).toLocaleString() : ""} frames
          in the corpus.
        </p>
      </div>
    </section>
  );
}

/** What to compare, when you arrived with nothing.
 *
 * An empty bench is not a starting point — it asks the reader to already know
 * two ids, which nobody does. The split-integrity view ends on "cross-split
 * pairs — judge these yourself" and then offers nowhere to judge them; this is
 * that queue, on the screen built for the judging. Same endpoint, same cached
 * pair list, same cosine: no second ranking implementation.
 *
 * The pairs are the highest-cosine cross-split ones because those are the pairs
 * with consequences — a held-out image that repeats a training image is the
 * case where a reported score is partly memorisation.
 */
function PairQueue() {
  const [pairs, setPairs] = useState<LeakagePair[] | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    api.leakage(LEAKAGE_THRESHOLD, ctrl.signal)
      .then((r) => setPairs(r.examples.filter((p) => p.cross_split).slice(0, PAIR_QUEUE_N)))
      .catch((e) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        // The bench still works by drag or by ?a=&b= — say the queue is
        // unavailable rather than rendering an empty list that looks like
        // "no pairs found".
        setFailed(e instanceof Error ? e.message : String(e));
      });
    return () => ctrl.abort();
  }, []);

  if (failed) {
    return (
      <section className="panel pair-queue">
        <div className="notice">Pair queue unavailable — {failed}</div>
      </section>
    );
  }
  if (!pairs) return <div className="loading">Finding pairs worth comparing…</div>;
  if (!pairs.length) return null;

  return (
    <section className="panel pair-queue">
      <h3>Start from a confusable pair</h3>
      <p className="meta-line">
        The {pairs.length} highest-cosine cross-split pairs — one side held out,
        one side in training. Open one to judge whether it is the same
        photograph or only a similar scene; the{" "}
        <Link to="/stats?view=integrity">split-integrity view</Link> is where the
        counts live.
      </p>
      <div className="pair-queue-list">
        {pairs.map((p) => (
          <Link key={`${p.a_id}-${p.b_id}`} className="pair-queue-row"
                to={`/compare?a=${p.a_id}&b=${p.b_id}`}
                title={`Open #${p.a_id} (${p.a_split}) beside #${p.b_id} (${p.b_split})`}>
            <img src={p.a_thumb} alt="" loading="lazy" />
            <img src={p.b_thumb} alt="" loading="lazy" />
            <span className="pair-queue-score">{p.score.toFixed(3)}</span>
            <span className="pair-queue-splits">{p.a_split} ↔ {p.b_split}</span>
          </Link>
        ))}
      </div>
    </section>
  );
}

function TagLine({ name, tags, empty }: { name: string; tags: string[]; empty: string }) {
  if (tags.length === 0 && empty === "none") return null;
  return (
    <div className="diff-tag-row">
      <span className="eyebrow">{name}</span>
      {tags.length === 0
        ? <span className="diff-tags-none">{empty}</span>
        : tags.map((t) => <span key={t} className="tag vlm">{t}</span>)}
    </div>
  );
}
