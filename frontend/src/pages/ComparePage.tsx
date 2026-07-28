import {
  useCallback, useEffect, useRef, useState,
} from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useActiveProviderName } from "../lib/activeProvider";
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
 */

// Must match AlbumShelf's DRAG_IDS. Declared locally rather than imported so
// this lazy chunk does not pull the shelf (and its client) along for one string.
const DRAG_IDS = "application/x-cvde-ids";

const AXES = ["legibility", "rarity", "difficulty", "clutter"] as const;

const MIN_SCALE = 0.5;
const MAX_SCALE = 12;

const STAGE_TITLE =
  "Wheel zooms toward the cursor, drag pans, double-click resets — both panes "
  + "move together. Keyboard: + / - zoom, arrow keys pan, 0 resets.";

interface Caption { text: string; agreement: number | null }
interface SampleDetail {
  id: number;
  filename: string;
  split: string;
  width: number;
  height: number;
  image_url: string;
  thumb_url: string;
  captions: Caption[];
  vlm_tags: string[];
  attributes: Record<string, string>;
  cluster: number | null;
  axes: Record<string, number | object | null>;
}
interface SimilarCard {
  id: number;
  thumb_url: string;
  caption: string | null;
  score: number | null;
}
interface RectGeom { x: number; y: number; w: number; h: number }
interface Annotation { kind: string; geometry: RectGeom; label?: string | null }
interface View { s: number; tx: number; ty: number }

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

/** Crop the ORIGINAL image (not the thumbnail — the region may be small, and
 * the embedder deserves the pixels) and return it as a JPEG blob. */
async function cropRegion(imageUrl: string, r: RectGeom): Promise<Blob> {
  const img = new Image();
  img.src = imageUrl;
  await img.decode();
  const sw = Math.max(1, Math.round(r.w * img.naturalWidth));
  const sh = Math.max(1, Math.round(r.h * img.naturalHeight));
  const canvas = document.createElement("canvas");
  canvas.width = sw;
  canvas.height = sh;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no 2d canvas context");
  ctx.drawImage(img,
    Math.round(r.x * img.naturalWidth), Math.round(r.y * img.naturalHeight),
    sw, sh, 0, 0, sw, sh);
  const blob = await new Promise<Blob | null>(
    (res) => canvas.toBlob(res, "image/jpeg", 0.92));
  if (!blob) throw new Error("cropping produced no image");
  return blob;
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
    getJSON<SimilarCard[]>(`/api/samples/${aId}/similar?top_k=60`)
      .then((cards) => {
        if (!live) return;
        const i = cards.findIndex((c) => c.id === bId);
        const hit = i >= 0 ? cards[i] : undefined;
        setSim(hit && hit.score != null ? { score: hit.score, rank: i + 1 } : "absent");
      })
      .catch(() => { /* unknown stays unknown — no line beats a fake number */ });
    return () => { live = false; };
  }, [aId, bId]);

  // The annotations API is being landed separately. Probed, not assumed: a 404
  // disables Save (with the reason in its title) while region search keeps
  // working. Anything else leaves the button live and lets the POST speak.
  const [annApiUp, setAnnApiUp] = useState<boolean | null>(null);
  const onAnnApi = useCallback((up: boolean) => { setAnnApiUp(up); }, []);

  const [regionHits, setRegionHits] =
    useState<{ fromId: number; cards: SimilarCard[] } | null>(null);
  const [searching, setSearching] = useState(false);
  const [regionErr, setRegionErr] = useState<string | null>(null);

  const searchRegion = useCallback(async (sample: SampleDetail, rect: RectGeom) => {
    setSearching(true);
    setRegionErr(null);
    try {
      const blob = await cropRegion(sample.image_url, rect);
      const r = await fetch("/api/search/by-image?top_k=24", {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: blob,
      });
      if (!r.ok) throw new Error(`region search failed: ${r.status}`);
      setRegionHits({ fromId: sample.id, cards: (await r.json()) as SimilarCard[] });
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
      <h2 className="section-title compare-title">Two frames, one loupe</h2>

      <div className="compare-panes">
        <Pane slot="a" sample={a.data} error={a.error} view={view} setView={setView}
          annApiUp={annApiUp} onAnnApi={onAnnApi} onFill={fill}
          onSearchRegion={searchRegion} />
        <Pane slot="b" sample={b.data} error={b.error} view={view} setView={setView}
          annApiUp={annApiUp} onAnnApi={onAnnApi} onFill={fill}
          onSearchRegion={searchRegion} />
      </div>

      {regionErr && <div className="error">{regionErr}</div>}
      {(searching || regionHits) && (
        <section className="panel region-results">
          <h3>
            Region search
            {regionHits && !searching
              && ` — region of #${regionHits.fromId}, nearest ${regionHits.cards.length} by image embedding`}
          </h3>
          {searching
            ? <div className="loading">Embedding the crop and ranking the corpus…</div>
            : (
              <div className="region-strip">
                {regionHits && regionHits.cards.map((c) => (
                  <Link key={c.id} to={`/samples/${c.id}`} className="region-hit"
                    title={c.caption ?? undefined}>
                    <img src={c.thumb_url} alt={`sample ${c.id}`} loading="lazy" />
                    <span className="mono">
                      #{c.id}{c.score != null && ` · ${c.score.toFixed(3)}`}
                    </span>
                  </Link>
                ))}
              </div>
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
  annApiUp: boolean | null;
  onAnnApi: (up: boolean) => void;
  onFill: (slot: "a" | "b", ids: number[]) => void;
  onSearchRegion: (sample: SampleDetail, rect: RectGeom) => void;
}

function Pane({
  slot, sample, error, view, setView, annApiUp, onAnnApi, onFill, onSearchRegion,
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
  const [draft, setDraft] = useState<RectGeom | null>(null);
  const [label, setLabel] = useState("");
  const [saved, setSaved] = useState<Annotation[]>([]);
  const [saveErr, setSaveErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const sampleId = sample ? sample.id : null;

  // A new occupant clears the previous one's marks.
  useEffect(() => {
    setDraft(null); setLabel(""); setDrawMode(false); setSaveErr(null);
  }, [sampleId]);

  // Existing saved regions, drawn as thin sage outlines. A 404 means the
  // annotations API is not mounted (yet) — skipped silently, by contract.
  useEffect(() => {
    setSaved([]);
    if (sampleId == null) return;
    let live = true;
    fetch(`/api/samples/${sampleId}/annotations`).then(async (r) => {
      if (!live) return;
      if (r.status === 404) { onAnnApi(false); return; }
      if (!r.ok) return;
      onAnnApi(true);
      const list = (await r.json()) as Annotation[];
      if (live && Array.isArray(list)) {
        setSaved(list.filter((x) => x.kind === "rect" && x.geometry != null));
      }
    }).catch(() => { /* optional layer; its absence is not an error */ });
    return () => { live = false; };
  }, [sampleId, onAnnApi]);

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

  const save = async () => {
    if (!sample || !draft) return;
    setSaving(true);
    setSaveErr(null);
    try {
      const r = await fetch(`/api/samples/${sample.id}/annotations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: "rect", geometry: draft, label: label || null }),
      });
      if (r.status === 404) { onAnnApi(false); return; }
      if (!r.ok) throw new Error(`save failed: ${r.status}`);
      setSaved((s) => [...s, { kind: "rect", geometry: draft, label: label || null }]);
      setDraft(null);
      setLabel("");
    } catch (e: unknown) {
      setSaveErr(String(e));
    } finally {
      setSaving(false);
    }
  };

  const pct = (r: RectGeom): React.CSSProperties => ({
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
            <button
              type="button"
              className={`ghost draw-toggle${drawMode ? " select-on" : ""}`}
              aria-pressed={drawMode}
              title="Drag on the image to mark a rectangle — drawn by you, no segmentation model involved; search it or save it"
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
              {saved.map((ann, i) => (
                <div key={i} className="region-rect saved" style={pct(ann.geometry)}>
                  {ann.label && <span className="region-label">{ann.label}</span>}
                </div>
              ))}
              {draft && <div className="region-rect draft" style={pct(draft)} />}
            </div>
          </div>
          {draft && (
            <div className="region-actions">
              <input
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="label (optional)"
                aria-label={`Label for the region on sample ${sample.id}`}
              />
              <button type="button" className="primary"
                onClick={() => onSearchRegion(sample, draft)}>
                Search this region
              </button>
              <button
                type="button"
                className="ghost"
                disabled={annApiUp === false || saving}
                title={annApiUp === false
                  ? "annotations API not up yet — region search still works"
                  : "Save this rectangle as an annotation"}
                onClick={() => { void save(); }}
              >
                {saving ? "Saving…" : "Save region"}
              </button>
              <button type="button" className="ghost region-clear"
                title="Discard this rectangle"
                onClick={() => { setDraft(null); setLabel(""); }}>
                ✕
              </button>
            </div>
          )}
          {saveErr && <div className="error region-save-err">{saveErr}</div>}
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
          rarer, or busier, relative to the other 8,090 frames.
        </p>
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
