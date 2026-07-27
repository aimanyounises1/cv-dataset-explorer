import { useState } from "react";
import { AXES } from "../api/types";
import AxisSparkline from "./AxisSparkline";
import { AXIS_META } from "./AxisFilters";

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
export default function AxisLegend() {
  const [open, setOpen] = useState(false);

  return (
    <div className="axis-legend">
      <AxisSparkline axes={EXAMPLE} />
      <span className="axis-legend-keys">
        {AXES.map((a) => (
          <span className="axis-legend-key" key={a} title={AXIS_META[a].hint}>
            <b>{INITIAL[a]}</b>{AXIS_META[a].label}
          </span>
        ))}
      </span>
      <span className="axis-legend-scale">taller = harder</span>
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
