/**
 * The render-block registry: one component per block kind, and one entry point
 * that dispatches on `kind`.
 *
 * The registry is a mapped type over `BlockKind`, which is the point of the
 * file. Adding a kind to `api/blocks.ts` (because it was added to
 * `backend/app/agent/blocks.py`) makes `BLOCK_RENDERERS` fail `tsc` until a
 * component exists for it, so "the backend can emit it" and "the canvas can
 * draw it" cannot drift apart silently. A `Record<string, FC>` would compile
 * happily and ship an empty box.
 *
 * The frame around every block lives here rather than in each renderer, so the
 * three things that must never be optional — the title, the caveat, and the
 * provenance line — are structurally impossible to leave out of a new kind.
 */
import type { FC } from "react";
import type { Block, BlockKind, BlockOfKind } from "../../api/blocks";
import { isBlockKind } from "../../api/blocks";
import BarBlockView from "./BarBlockView";
import FlowBlockView from "./FlowBlockView";
import HistogramBlockView from "./HistogramBlockView";
import ImagesBlockView from "./ImagesBlockView";
import LineBlockView from "./LineBlockView";
import PieBlockView from "./PieBlockView";
import QABlockView from "./QABlockView";
import ReportBlockView from "./ReportBlockView";
import StatBlockView from "./StatBlockView";
import TableBlockView from "./TableBlockView";
import TagProposalBlockView from "./TagProposalBlockView";
import "../../styles/blocks.css";

/** A renderer for kind K accepts exactly the block for kind K. */
type Renderer<K extends BlockKind> = FC<{ block: BlockOfKind<K> }>;

/** Exhaustive by construction: a missing key is a type error, and so is a key
 * for a kind the union does not admit. */
export const BLOCK_RENDERERS: { [K in BlockKind]: Renderer<K> } = {
  bar: BarBlockView,
  line: LineBlockView,
  pie: PieBlockView,
  histogram: HistogramBlockView,
  table: TableBlockView,
  stat: StatBlockView,
  flow: FlowBlockView,
  images: ImagesBlockView,
  report: ReportBlockView,
  qa: QABlockView,
  tag_proposal: TagProposalBlockView,
};

/** Dispatch is a lookup rather than a ten-arm switch, so the registry above is
 * the single place a kind is wired up. The cast is the price: TypeScript cannot
 * prove that the renderer pulled out by `block.kind` is the one that accepts
 * this block, and the switch that would prove it is a second copy of the
 * registry that can fall out of step with the first. Widening to `string` also
 * makes the missing-renderer case expressible, which the typed lookup does not:
 * a block from a newer backend has a `kind` this build has never heard of. */
const LOOKUP = BLOCK_RENDERERS as unknown as
  Record<string, FC<{ block: Block }> | undefined>;

/** What a newer backend's block looks like in an older UI. Loud rather than
 * blank: the failure is a version skew the reader can act on, and the payload
 * is shown because it usually contains the answer they wanted. */
function UnknownBlockView({ block }: { block: Block }) {
  return (
    <div className="vblock-unknown">
      <p>
        <strong>No renderer for block type “{block.kind}”.</strong> This build of
        the interface is older than the agent that produced it.
      </p>
      <details>
        <summary>Show the raw payload</summary>
        <pre>{JSON.stringify(block, null, 2)}</pre>
      </details>
    </div>
  );
}

/**
 * Render one block: frame, body, caveat, provenance.
 *
 * Declared as a function statement, not a const arrow, because
 * `ReportBlockView` imports it back to draw its nested blocks. Hoisting makes
 * that cycle safe at module-evaluation time.
 */
export function BlockView({ block }: { block: Block }) {
  // `isBlockKind` is the guard that matters: a payload whose `kind` this build
  // has never heard of must reach the fallback, and the typed lookup on its own
  // cannot express that case.
  const Body = isBlockKind(block.kind) ? LOOKUP[block.kind] : undefined;

  return (
    <section className={`vblock vblock-${block.kind}`}>
      <header className="vblock-head">
        <h4 className="vblock-title">{block.title}</h4>
      </header>

      {Body !== undefined ? <Body block={block} /> : <UnknownBlockView block={block} />}

      {block.note && <p className="vblock-note">{block.note}</p>}

      {/* Always rendered, including the case the types say cannot happen: a
          block that reached the browser without a source is a broken contract,
          and saying so is the only honest thing to draw under a chart. */}
      <p className="vblock-source">
        <span className="vblock-source-label">source</span>{" "}
        {block.source
          ? block.source
          : <em>not stated by the tool that produced this block</em>}
      </p>
    </section>
  );
}

export default BlockView;
