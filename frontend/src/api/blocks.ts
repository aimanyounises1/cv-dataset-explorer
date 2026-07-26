/**
 * Render blocks: the TypeScript half of the contract in
 * `backend/app/agent/blocks.py`.
 *
 * The agent's tools return *blocks* rather than prose, and this file is the
 * mirror of the pydantic union that produces them. Kept as a discriminated
 * union on `kind` for one reason: the renderer registry in
 * `components/blocks/index.tsx` is a mapped type over `BlockKind`, so adding a
 * kind here fails `tsc` until a component exists to draw it. The alternative —
 * an index signature or `any` — would let a new backend block type ship as a
 * silently empty box.
 *
 * Two conventions carry through every block and are load-bearing rather than
 * decorative:
 *
 * * `source` is **required**, and every renderer prints it. A chart whose
 *   provenance is unstated cannot be told apart from a chart the model
 *   invented.
 * * `drill` fields carry a *gallery query string* ("split=train",
 *   "attr=time_of_day:night"), not a URL. Renderers navigate to `/?<drill>`,
 *   which is what makes a bar part of the app instead of a picture of it.
 *
 * Fields that carry a non-`None` default on the backend are optional here.
 * FastAPI serializes them today, but a later `exclude_defaults` dump would drop
 * them, and a missing `stacked` should not be a type error at the call site.
 */

/** Every kind the backend union admits, in the backend's own order. The
 * registry is keyed by this, so the two sets cannot drift. */
export const BLOCK_KINDS = [
  "bar", "line", "pie", "histogram", "table", "stat", "flow", "images",
  "report", "qa",
] as const;

export type BlockKind = (typeof BLOCK_KINDS)[number];

// ------------------------------------------------------------------ shared --

/** One categorical value. `drill` opens the slice this point counts. */
export interface Point {
  label: string;
  value: number;
  drill?: string | null;
}

export interface XY {
  x: number;
  y: number;
}

export interface Series {
  name: string;
  points: Point[];
}

export interface XYSeries {
  name: string;
  points: XY[];
}

interface BlockBase {
  title: string;
  /** How the numbers were produced. Required by the backend, always rendered. */
  source: string;
  /** A caveat about this block's numbers — rendered when present. */
  note?: string | null;
}

// ------------------------------------------------------------------- kinds --

export interface BarBlock extends BlockBase {
  kind: "bar";
  series: Series[];
  x_label?: string | null;
  y_label?: string | null;
  /** Category labels on the y axis. Set when they are long, which for this
   * dataset is most of them. */
  horizontal?: boolean;
  stacked?: boolean;
}

export interface LineBlock extends BlockBase {
  kind: "line";
  series: XYSeries[];
  x_label?: string | null;
  y_label?: string | null;
  area?: boolean;
}

export interface PieBlock extends BlockBase {
  kind: "pie";
  points: Point[];
}

/** The backend types `bins` as a bare `list[dict]` documented as
 * `[{lo, hi, count}]`, so nothing validates the shape. Renderers coerce with
 * `Number()` rather than trusting it. */
export interface HistogramBin {
  lo: number;
  hi: number;
  count: number;
}

export interface HistogramBlock extends BlockBase {
  kind: "histogram";
  bins: HistogramBin[];
  x_label?: string | null;
  /** Where a meaningful cut sits — drawn as a labelled reference line. */
  marker?: number | null;
  marker_label?: string | null;
  /** e.g. "max_agreement": clicking a bin opens `/?<drill_param>=<bin.hi>`. */
  drill_param?: string | null;
}

export interface TableColumn {
  key: string;
  label: string;
  numeric?: boolean;
}

/** Rows are `list[dict]` on the backend: values are whatever the tool put
 * there, so they are `unknown` until a renderer formats them. */
export type TableRow = Record<string, unknown>;

export interface TableBlock extends BlockBase {
  kind: "table";
  columns: TableColumn[];
  rows: TableRow[];
  /** Per-row query string under this key turns rows into links. */
  drill_key?: string | null;
}

export interface StatItem {
  label: string;
  value: string;
  hint?: string | null;
  drill?: string | null;
}

export interface StatBlock extends BlockBase {
  kind: "stat";
  items: StatItem[];
}

export interface FlowNode {
  id: string;
  label: string;
  /** Role such as "agent", "tool", "store". Mapped to the shared categorical
   * palette so a diagram matches the rest of the app. */
  group?: string | null;
  drill?: string | null;
}

export interface FlowEdge {
  src: string;
  dst: string;
  label?: string | null;
}

export interface FlowBlock extends BlockBase {
  kind: "flow";
  nodes: FlowNode[];
  edges?: FlowEdge[];
  /** Explicit left-to-right columns of node ids. The producer supplies the
   * layering, which is why this block needs no graph-layout dependency. */
  layers?: string[][];
}

export interface ImagesBlock extends BlockBase {
  kind: "images";
  sample_ids: number[];
  /** Set when the tool had more than it could show, so the UI can offer the
   * rest in the gallery instead of implying this is everything. */
  total?: number | null;
  drill?: string | null;
}

export interface ReportSection {
  heading: string;
  text?: string | null;
  blocks?: Block[];
}

export interface ReportBlock extends BlockBase {
  kind: "report";
  sections: ReportSection[];
  download_md?: string | null;
  download_json?: string | null;
}

/** One assertion inside a QA flow. `list[dict]` on the backend. */
export interface QACheck {
  name: string;
  ok: boolean;
  detail?: string | null;
}

export interface QAFlowResult {
  name: string;
  status: "pass" | "fail" | "skip";
  checks?: QACheck[];
  /** URL under /media/qa/. */
  screenshot?: string | null;
  detail?: string | null;
  duration_s?: number | null;
}

export interface QABlock extends BlockBase {
  kind: "qa";
  status: "running" | "done" | "failed";
  run_id: string;
  flows?: QAFlowResult[];
  passed?: number;
  total?: number;
  console_errors?: string[];
  deck_url?: string | null;
  markdown_url?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
}

export type Block =
  | BarBlock
  | LineBlock
  | PieBlock
  | HistogramBlock
  | TableBlock
  | StatBlock
  | FlowBlock
  | ImagesBlock
  | ReportBlock
  | QABlock;

/** The block type for one kind, e.g. `BlockOfKind<"bar">` is `BarBlock`. Lets
 * the registry state "the renderer for K takes the block for K" without naming
 * ten pairs by hand. */
export type BlockOfKind<K extends BlockKind> = Extract<Block, { kind: K }>;

/** True when a string names a kind this build can draw. Blocks from a newer
 * backend fail this and get the fallback renderer rather than a blank space. */
export function isBlockKind(kind: string): kind is BlockKind {
  return (BLOCK_KINDS as readonly string[]).includes(kind);
}
