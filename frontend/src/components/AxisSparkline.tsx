import { AXES, Axis, AxisScores } from "../api/types";
import { HEAT_COLOR, SURFACE, heatBand } from "../lib/viz";
import { AXIS_META } from "./AxisFilters";

/** Order is fixed and matches the filter panel, so the same shape always means
 * the same thing across the grid. */
const ORDER: readonly Axis[] = AXES;
const INITIAL: Record<Axis, string> = {
  legibility: "L", rarity: "R", difficulty: "D", clutter: "C",
};

const BAR_W = 7;
const GAP = 3;
const H = 18;

interface Props {
  axes: AxisScores;
  /** Larger bars for the detail page, where there is room and only one sample. */
  size?: "compact" | "regular";
}

/**
 * Four difficulty axes as a micro bar chart on one shared 0-10 scale.
 *
 * Replaces four bare numbers. The numbers were individually precise and
 * collectively unreadable: comparing sixty cards meant reading 240 digits and
 * ranking them mentally. Because all four axes are percentile ranks on the same
 * scale, height is directly comparable both within a card and across the grid,
 * so an unusual *profile* — high difficulty over low legibility, say — becomes a
 * shape you notice without reading anything.
 *
 * Colour repeats what height already says (it is a redundant encoding, not the
 * only one), so the chart survives greyscale and colour blindness. Exact values
 * stay available in the tooltip and on the detail page.
 */
export default function AxisSparkline({ axes, size = "compact" }: Props) {
  const scale = size === "regular" ? 1.7 : 1;
  const bw = BAR_W * scale;
  const gap = GAP * scale;
  const h = H * scale;
  const present = ORDER.filter((a) => axes[a] != null);
  if (present.length === 0) return null;

  const label = present
    .map((a) => `${AXIS_META[a].label} ${axes[a]} of 10`)
    .join("; ");

  return (
    <svg
      className="axis-spark"
      width={present.length * bw + (present.length - 1) * gap}
      height={h + (size === "regular" ? 12 : 0)}
      role="img"
      aria-label={`Difficulty axes: ${label}`}
    >
      {/* A <title> child, not a title attribute: it is the accessible name SVG
          actually defines, and browsers render it as the hover tooltip too. */}
      <title>
        {present
          .map((a) => `${AXIS_META[a].label} ${axes[a]}/10 — ${AXIS_META[a].low} → ${AXIS_META[a].high}`)
          .join("\n")}
      </title>
      {present.map((axis, i) => {
        const v = axes[axis] as number;
        // Floor of 1px so a zero still reads as "measured, and low" rather than
        // as missing data — those are different facts.
        const barH = Math.max(1, (v / 10) * h);
        const x = i * (bw + gap);
        return (
          <g key={axis}>
            <rect x={x} y={0} width={bw} height={h} rx={1} fill={SURFACE.inset} />
            <rect x={x} y={h - barH} width={bw} height={barH} rx={1}
                  fill={HEAT_COLOR[heatBand(v)]} />
            {size === "regular" && (
              <text x={x + bw / 2} y={h + 10} textAnchor="middle"
                    fontSize={9} fill={SURFACE.textFaint}>
                {INITIAL[axis]}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
