import { useState } from "react";
import { AXES, Axis } from "../api/types";
import AxisSparkline from "./AxisSparkline";
import { AXIS_META } from "./AxisFilters";
import { Listbox } from "./controls";

/** A worked example, not a diagram: ascending bars so "taller = harder" is
 * legible from the shape alone, and four distinct heights so each position is
 * separable. It is rendered by the *same component* the cards use, so the key
 * and the thing it explains cannot drift apart. */
const EXAMPLE = { legibility: 2, rarity: 5, difficulty: 8, clutter: 10, detail: {} };

const INITIAL: Record<string, string> = {
  legibility: "L", rarity: "R", difficulty: "D", clutter: "C",
};

/**
 * The key for the four-bar sparkline on every image card.
 *
 * That sparkline is the densest encoding in the product — it appears on all 60
 * cards — and it was the only one with no legend anywhere. The map has one, four
 * chart renderers have one; the gallery had none. Its entire explanation was an
 * SVG `<title>`, which is to say: a hover you have to already suspect is there.
 * A reader saw four grey-and-amber bars and a pill reading "dif 9" and had no
 * way in.
 *
 * It sits in the result bar rather than above the grid because the redesign just
 * cut chrome-before-first-image from 481px to 159px, and a legend that costs a
 * row would give some of that straight back. The result bar already exists and
 * has had a free right-hand slot since the export pills moved to the selection
 * rail.
 */
export default function AxisLegend({ signal, onSignal }: {
  /** The axis the grid's frame edges are drawing, or "" for none. */
  signal?: string;
  /** Supplied only where the legend also *chooses* the encoding — the gallery.
   *  Without it this stays what it has always been: a key. The sample page and
   *  the assistant's image blocks show one card each, where an edge that
   *  encodes a corpus percentile has nothing to be compared against. */
  onSignal?: (axis: string) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="axis-legend">
      <AxisSparkline axes={EXAMPLE} />
      {/* The key used to spell out L·R·D·C beside the sparkline. It now names one
          axis at a time, because the frame edge draws one at a time — a legend
          for four encodings when the grid shows one is a legend for the wrong
          thing. The sparkline beside it still explains the four-bar chip in the
          hover strip, and the disclosure below still carries what the numbers
          are and are not. */}
      {onSignal ? (
        <Listbox
          inline
          label="Draw which measurement on the frames"
          trigger={signal ? AXIS_META[signal as Axis].label : "No edge"}
          selected={signal ?? ""}
          options={[
            { value: "", label: "No edge" },
            ...AXES.map((a) => ({ value: a, label: AXIS_META[a].label })),
          ]}
          onPick={onSignal}
        />
      ) : (
        <span className="axis-legend-keys">
          {AXES.map((a) => (
            <span className="axis-legend-key" key={a} title={AXIS_META[a].hint}>
              <b>{INITIAL[a]}</b>{AXIS_META[a].label}
            </span>
          ))}
        </span>
      )}
      {/* Only where a bar is actually drawn. In the gallery with the edge off,
          this described an encoding nothing on screen was using, which made the
          control read as broken rather than as off. The key mode (no picker)
          always draws its sparkline, so it always earns the caption. */}
      {(!onSignal || signal) && (
        <span className="axis-legend-scale">taller = harder</span>
      )}
      <button className="axis-legend-more" aria-expanded={open}
              onClick={() => setOpen((v) => !v)}
              title="What these bars measure">
        {open ? "hide" : "what is this?"}
      </button>

      {open && (
        <div className="axis-legend-panel">
          <p>
            Every card carries four bars — one per axis, always in this order — on
            a shared 0–10 scale, so an unusual <em>profile</em> is a shape you
            notice without reading anything. The badge beside them names the
            highest axis, which is why one image says <code>dif 9</code> and
            another <code>leg 10</code>.
          </p>
          <dl>
            {AXES.map((a) => (
              <div key={a}>
                <dt><b>{INITIAL[a]}</b> {AXIS_META[a].label}</dt>
                <dd>
                  <span className="axis-legend-ends">
                    {AXIS_META[a].low} → {AXIS_META[a].high}
                  </span>
                  {AXIS_META[a].hint}
                </dd>
              </div>
            ))}
          </dl>
          <p className="axis-legend-caveat">
            These are <strong>percentile ranks over this dataset</strong>, not
            absolute measurements: a 7 means “harder than about 70% of these 8,000
            images”. They do not transfer to another corpus, and ingesting more
            images can move a sample’s bucket without its pixels changing.
          </p>
        </div>
      )}
    </div>
  );
}
