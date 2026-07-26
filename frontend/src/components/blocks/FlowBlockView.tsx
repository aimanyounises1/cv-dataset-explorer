import { useId, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import type { FlowBlock, FlowEdge, FlowNode } from "../../api/blocks";
import { SURFACE, categorical } from "../../lib/viz";

/* Geometry. Fixed sizes rather than measured text: the node boxes are uniform
 * so the columns line up, and the label is wrapped to fit the box instead of
 * the box being grown to fit the label. */
// Sized so the common case — a four-stage pipeline — fits inside a chat bubble
// without scrolling: 28 + 4×112 + 3×68 = 680px. Anything deeper scrolls rather
// than shrinking below legible type. The gap is wide enough to hold an edge
// label between two columns, which is what keeps labels off the node boxes.
const NW = 112;          // node width
const NH = 46;           // node height
const COL_GAP = 68;
const ROW_GAP = 20;
const PAD = 14;
const DIP = 36;          // how far a backward edge bows below the rows
const CHAR_W = 5.6;      // approximate advance of the 9.5px mono edge label
const LINE_H = 11;

interface Pt { x: number; y: number; }

/** Longest-path layering, used only when the producer omitted `layers`. The
 * `flow()` builder always supplies them, so this is a fallback rather than the
 * main path — but a diagram that silently collapses to one column because a
 * field was missing is worse than twelve lines of graph code. Cycles stop
 * moving once depth is bounded by the node count. */
function deriveLayers(nodes: FlowNode[], edges: FlowEdge[]): string[][] {
  const depth = new Map<string, number>(nodes.map((n) => [n.id, 0]));
  for (let pass = 0; pass < nodes.length; pass++) {
    let changed = false;
    for (const e of edges) {
      const d = (depth.get(e.src) ?? 0) + 1;
      if (depth.has(e.dst) && d > (depth.get(e.dst) as number) && d < nodes.length) {
        depth.set(e.dst, d);
        changed = true;
      }
    }
    if (!changed) break;
  }
  const max = nodes.reduce((m, n) => Math.max(m, depth.get(n.id) ?? 0), 0);
  const cols: string[][] = Array.from({ length: max + 1 }, () => []);
  for (const n of nodes) cols[depth.get(n.id) ?? 0].push(n.id);
  return cols.filter((c) => c.length > 0);
}

/** Wrap to at most `maxLines` lines of roughly `max` characters. SVG has no
 * text wrapping, so this decides the `<tspan>`s. */
function wrap(text: string, max: number, maxLines: number): string[] {
  const out: string[] = [];
  let cur = "";
  for (const w of text.split(/\s+/).filter(Boolean)) {
    const next = cur === "" ? w : `${cur} ${w}`;
    if (next.length <= max) { cur = next; continue; }
    if (cur !== "") out.push(cur);
    cur = w.length > max ? `${w.slice(0, max - 1)}…` : w;
  }
  if (cur !== "") out.push(cur);
  if (out.length <= maxLines) return out;
  const kept = out.slice(0, maxLines);
  kept[maxLines - 1] = `${kept[maxLines - 1].slice(0, max - 1)}…`;
  return kept;
}

/** Midpoint of a cubic Bézier at t=0.5, where an edge label goes. */
function midOf(p0: Pt, p1: Pt, p2: Pt, p3: Pt): Pt {
  return {
    x: (p0.x + 3 * p1.x + 3 * p2.x + p3.x) / 8,
    y: (p0.y + 3 * p1.y + 3 * p2.y + p3.y) / 8,
  };
}

interface Drawn {
  key: string;
  d: string;
  mid: Pt;
  /** Wrapped to fit the space the edge crosses, so a label can never sit on a
   * node box. Empty when the edge is unlabelled. */
  labelLines: string[];
  labelWidth: number;
}

/** An edge label lives at the curve's midpoint, which for a forward edge is the
 * centre of the gap between two columns. That gap is all the room it has: a
 * plate wider than the gap would print over the node it points away from. */
function fitLabel(text: string | null, room: number): { lines: string[]; width: number } {
  if (text === null || text === "") return { lines: [], width: 0 };
  const perLine = Math.max(4, Math.floor((room - 10) / CHAR_W));
  const lines = wrap(text, perLine, 2);
  const longest = lines.reduce((m, l) => Math.max(m, l.length), 0);
  return { lines, width: Math.min(room, longest * CHAR_W + 10) };
}

export default function FlowBlockView({ block }: { block: FlowBlock }) {
  const navigate = useNavigate();
  // One arrowhead def per rendered block: two flows in the same transcript must
  // not share a marker id.
  const rawId = useId();
  const arrow = `vflow-arrow-${rawId.replace(/[^a-zA-Z0-9_-]/g, "")}`;

  const nodes = block.nodes;
  const edges = block.edges ?? [];

  const model = useMemo(() => {
    const byId = new Map(nodes.map((n) => [n.id, n]));

    // Ignore layer entries naming nodes that do not exist, and append any node
    // the layering forgot as a final column so nothing goes missing.
    let cols = (block.layers ?? [])
      .map((l) => l.filter((id) => byId.has(id)))
      .filter((l) => l.length > 0);
    if (cols.length === 0) {
      cols = deriveLayers(nodes, edges);
    } else {
      const placed = new Set(cols.flat());
      const missing = nodes.filter((n) => !placed.has(n.id)).map((n) => n.id);
      if (missing.length > 0) cols.push(missing);
    }
    if (cols.length === 0) cols = [[]];

    const tallest = cols.reduce((m, c) => Math.max(m, c.length), 1);
    const hasBackward = edges.some((e) => {
      const s = cols.findIndex((c) => c.includes(e.src));
      const d = cols.findIndex((c) => c.includes(e.dst));
      return s >= 0 && d >= 0 && d <= s;
    });

    const rowsH = tallest * NH + (tallest - 1) * ROW_GAP;
    const width = PAD * 2 + cols.length * NW + (cols.length - 1) * COL_GAP;
    const height = PAD * 2 + rowsH + (hasBackward ? DIP + 18 : 0);

    const pos = new Map<string, Pt>();
    cols.forEach((col, ci) => {
      const colH = col.length * NH + (col.length - 1) * ROW_GAP;
      const y0 = PAD + (rowsH - colH) / 2;
      col.forEach((id, ri) => pos.set(id, {
        x: PAD + ci * (NW + COL_GAP),
        y: y0 + ri * (NH + ROW_GAP),
      }));
    });

    const drawn: Drawn[] = [];
    edges.forEach((e, i) => {
      const a = pos.get(e.src);
      const b = pos.get(e.dst);
      if (a === undefined || b === undefined) return;   // builder rejects these
      const key = `${e.src}->${e.dst}-${i}`;
      const label = e.label ?? null;
      if (b.x > a.x) {
        // Forward: leave the right edge, enter the left edge.
        const p0 = { x: a.x + NW, y: a.y + NH / 2 };
        const p3 = { x: b.x, y: b.y + NH / 2 };
        const dx = Math.max(22, (p3.x - p0.x) / 2);
        const p1 = { x: p0.x + dx, y: p0.y };
        const p2 = { x: p3.x - dx, y: p3.y };
        const fit = fitLabel(label, p3.x - p0.x);
        drawn.push({
          key, labelLines: fit.lines, labelWidth: fit.width,
          mid: midOf(p0, p1, p2, p3),
          d: `M${p0.x},${p0.y} C${p1.x},${p1.y} ${p2.x},${p2.y} ${p3.x},${p3.y}`,
        });
      } else {
        // Backward or within a column: bow below the rows, where there are no
        // boxes to cross.
        const p0 = { x: a.x + NW / 2, y: a.y + NH };
        const p3 = { x: b.x + NW / 2, y: b.y + NH };
        const floor = PAD + rowsH + DIP;
        const p1 = { x: p0.x, y: floor };
        const p2 = { x: p3.x, y: floor };
        // Below the rows there are no boxes to collide with, so the label gets
        // the full plate width.
        const fit = fitLabel(label, 116);
        drawn.push({
          key, labelLines: fit.lines, labelWidth: fit.width,
          mid: midOf(p0, p1, p2, p3),
          d: `M${p0.x},${p0.y} C${p1.x},${p1.y} ${p2.x},${p2.y} ${p3.x},${p3.y}`,
        });
      }
    });

    const groups: string[] = [];
    for (const n of nodes) {
      if (n.group && !groups.includes(n.group)) groups.push(n.group);
    }

    return { cols, pos, drawn, width, height, groups, byId };
  }, [nodes, edges, block.layers]);

  const colorOf = (node: FlowNode): string => {
    if (!node.group) return SURFACE.borderStrong;
    return categorical(model.groups.indexOf(node.group));
  };

  const open = (drill: string) => navigate(`/?${drill}`);
  const anyDrill = nodes.some((n) => Boolean(n.drill));

  const label = `${block.title}. Flow diagram: ${nodes.length} nodes in `
    + `${model.cols.length} stages, ${model.drawn.length} connections.`;

  return (
    <div className="vblock-body">
      {model.groups.length > 0 && (
        <div className="vflow-legend">
          {model.groups.map((g, i) => (
            <span className="vflow-legend-item" key={g}>
              <span className="vblock-swatch" style={{ background: categorical(i) }} />
              {g}
            </span>
          ))}
        </div>
      )}

      {/* Natural size inside a scroller rather than scaled to fit: a six-stage
          pipeline squeezed into 700px would put its labels below legible size,
          and a diagram you cannot read is not a smaller diagram. */}
      <div className="vflow-scroll">
        <svg
          className="vflow"
          width={model.width}
          height={model.height}
          viewBox={`0 0 ${model.width} ${model.height}`}
          // Drillable nodes are focusable children, which `role="img"` would
          // hide from assistive technology; a plain diagram keeps `img`.
          role={anyDrill ? "group" : "img"}
          aria-label={label}
        >
          <defs>
            <marker
              id={arrow} viewBox="0 0 8 8" refX="7" refY="4"
              markerWidth="7" markerHeight="7" orient="auto-start-reverse"
            >
              <path d="M0,0 L8,4 L0,8 z" fill={SURFACE.textFaint} />
            </marker>
          </defs>

          {model.drawn.map((e) => (
            <path
              key={e.key} d={e.d} fill="none"
              stroke={SURFACE.borderStrong} strokeWidth={1.4}
              markerEnd={`url(#${arrow})`}
            />
          ))}

          {/* Edge labels after the paths so a line never runs through the text,
              and with a plate behind them so they stay readable on a crossing. */}
          {model.drawn.map((e) => e.labelLines.length === 0 ? null : (
            <g key={`${e.key}-label`} className="vflow-edge-label">
              <rect
                x={e.mid.x - e.labelWidth / 2}
                y={e.mid.y - (e.labelLines.length * LINE_H) / 2 - 2}
                width={e.labelWidth}
                height={e.labelLines.length * LINE_H + 4} rx={4}
                fill={SURFACE.bg} fillOpacity={0.92}
              />
              <text
                x={e.mid.x} textAnchor="middle" fontSize={9.5}
                y={e.mid.y - (e.labelLines.length - 1) * LINE_H / 2 + 3}
              >
                {e.labelLines.map((line, li) => (
                  <tspan key={li} x={e.mid.x} dy={li === 0 ? 0 : LINE_H}>{line}</tspan>
                ))}
              </text>
            </g>
          ))}

          {model.cols.flat().map((id) => {
            const node = model.byId.get(id);
            const p = model.pos.get(id);
            if (node === undefined || p === undefined) return null;
            const color = colorOf(node);
            const lines = wrap(node.label, 16, 2);
            const drill = node.drill ?? null;
            const y0 = p.y + NH / 2 + (lines.length === 1 ? 4 : -3);
            const body = (
              <>
                <rect
                  x={p.x} y={p.y} width={NW} height={NH} rx={6}
                  fill={color} fillOpacity={0.14}
                  stroke={color} strokeWidth={1.3}
                />
                <text
                  x={p.x + NW / 2} y={y0} textAnchor="middle"
                  fontSize={11.5} fill={SURFACE.text}
                >
                  {lines.map((line, li) => (
                    <tspan key={li} x={p.x + NW / 2} dy={li === 0 ? 0 : 14}>{line}</tspan>
                  ))}
                </text>
              </>
            );
            return drill !== null ? (
              <g
                key={id} className="vflow-node drillable"
                role="link" tabIndex={0}
                onClick={() => open(drill)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    open(drill);
                  }
                }}
              >
                <title>{`${node.label} — open this slice in the gallery (?${drill})`}</title>
                {body}
              </g>
            ) : (
              <g key={id} className="vflow-node">
                <title>{node.group ? `${node.label} (${node.group})` : node.label}</title>
                {body}
              </g>
            );
          })}
        </svg>
      </div>

      {/* The diagram as text. An `aria-label` can say how big the graph is but
          not what connects to what, and this is also the only way to read the
          edges when the labels are elided. */}
      <details className="vblock-values">
        <summary>Read as text ({model.drawn.length} connections)</summary>
        <ul>
          {model.drawn.length === 0 && <li><span className="vblock-value-static">No connections.</span></li>}
          {edges.map((e, i) => {
            const s = model.byId.get(e.src);
            const d = model.byId.get(e.dst);
            if (s === undefined || d === undefined) return null;
            return (
              <li key={`${e.src}-${e.dst}-${i}`}>
                <span className="vblock-value-static">
                  <span className="vblock-value-label">
                    {s.label} → {d.label}
                  </span>
                  <span className="vblock-value-num">{e.label ?? ""}</span>
                </span>
              </li>
            );
          })}
        </ul>
      </details>
    </div>
  );
}
