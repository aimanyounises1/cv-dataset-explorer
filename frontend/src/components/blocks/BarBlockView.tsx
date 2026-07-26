import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { BarBlock, Series } from "../../api/blocks";
import {
  AXIS_STROKE, GRID_STROKE, SERIES, TOOLTIP_STYLE, categorical, formatCount,
} from "../../lib/viz";

/** Series colours in the order viz.ts intends them, wrapping into the wider
 * categorical palette when a chart carries more series than SERIES names. */
const NAMED = [SERIES.blue, SERIES.green, SERIES.purple, SERIES.amber];
function seriesColor(i: number): string {
  return i < NAMED.length ? NAMED[i] : categorical(i);
}

/** Recharts rows are one object per category, so several series over the same
 * categories have to be pivoted. Series are addressed by a synthetic key (`s0`,
 * `s1`) rather than by name: `dataKey` is parsed as a property *path*, so a
 * series called "bbox.area" would silently plot nothing. The human name travels
 * on `<Bar name>`, which is what the tooltip and legend read. */
interface Row {
  __label: string;
  __drill: string | null;
  [key: string]: string | number | null;
}

function pivot(series: Series[]): { rows: Row[]; keys: string[] } {
  const keys = series.map((_, i) => `s${i}`);
  const order: string[] = [];
  const byLabel = new Map<string, Row>();
  series.forEach((s, i) => {
    for (const p of s.points) {
      let row = byLabel.get(p.label);
      if (row === undefined) {
        row = { __label: p.label, __drill: null };
        byLabel.set(p.label, row);
        order.push(p.label);
      }
      row[keys[i]] = Number(p.value);
      if (row.__drill === null && p.drill) row.__drill = p.drill;
    }
  });
  return { rows: order.map((l) => byLabel.get(l) as Row), keys };
}

/** Recharts spreads the datum onto the click argument in some versions and
 * nests it under `payload` in others; read both rather than depend on one. */
function drillOf(data: unknown): string | null {
  const d = data as { __drill?: unknown; payload?: { __drill?: unknown } } | null;
  const v = d?.__drill ?? d?.payload?.__drill;
  return typeof v === "string" && v !== "" ? v : null;
}

export default function BarBlockView({ block }: { block: BarBlock }) {
  const navigate = useNavigate();
  const [hidden, setHidden] = useState<ReadonlySet<string>>(new Set());
  const { rows, keys } = useMemo(() => pivot(block.series), [block.series]);

  const horizontal = block.horizontal === true;
  const multi = block.series.length > 1;
  const drillable = rows.some((r) => r.__drill !== null);

  const toggle = (name: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name); else next.add(name);
      // Hiding the last visible series leaves an empty chart with no way back,
      // so the final one refuses to hide.
      return next.size === block.series.length ? prev : next;
    });

  const open = (drill: string) => navigate(`/?${drill}`);

  /** Charts are announced, not described: one sentence with the shape of the
   * data and its extreme, which is what a sighted reader takes from a glance. */
  const peak = rows.reduce<{ label: string; value: number } | null>((best, r) => {
    for (const k of keys) {
      const v = r[k];
      if (typeof v === "number" && (best === null || v > best.value)) {
        best = { label: r.__label, value: v };
      }
    }
    return best;
  }, null);
  const label = `${block.title}. Bar chart of ${rows.length} categories`
    + `${multi ? ` across ${block.series.length} series` : ""}`
    + `${peak ? `; highest is ${peak.label} at ${formatCount(peak.value)}` : ""}.`;

  // The axes swap under `layout="vertical"`: categories move to the y axis and
  // the measured quantity to the x axis, so the two labels swap with them.
  const valAxis = block.y_label;

  /* Horizontal bars exist *because* the labels are long ("a man in a red shirt
   * climbing"), so the category axis is sized from the longest one instead of a
   * fixed width that silently clips the first character off every row. Past the
   * cap the tick is elided and the tooltip carries the whole label — the chart
   * gives up characters before it gives up bar length. */
  const longest = rows.reduce((m, r) => Math.max(m, r.__label.length), 0);
  const catWidth = Math.min(196, Math.max(84, Math.round(longest * 6.1) + 14));
  const catChars = Math.floor((catWidth - 14) / 6.1);
  const elide = (v: unknown, budget: number) => {
    const s = String(v);
    return s.length > budget ? `${s.slice(0, budget - 1)}…` : s;
  };
  const height = horizontal
    ? Math.max(170, rows.length * 30 + 44)
    : (block.x_label ? 272 : 250);

  return (
    <div className="vblock-body">
      {multi && (
        <div className="vblock-legend" role="group" aria-label="Toggle series">
          {block.series.map((s, i) => {
            const on = !hidden.has(s.name);
            return (
              <button
                key={s.name}
                type="button"
                className={`vblock-legend-item${on ? "" : " off"}`}
                aria-pressed={on}
                title={on ? `Hide “${s.name}”` : `Show “${s.name}”`}
                onClick={() => toggle(s.name)}
              >
                <span className="vblock-swatch" style={{ background: seriesColor(i) }} />
                {s.name}
              </button>
            );
          })}
        </div>
      )}

      <div className="vblock-chart" role="img" aria-label={label}>
        <ResponsiveContainer width="100%" height={height}>
          <BarChart
            data={rows}
            layout={horizontal ? "vertical" : "horizontal"}
            margin={{ top: 4, right: 14, left: 4, bottom: block.x_label ? 20 : 2 }}
          >
            <CartesianGrid stroke={GRID_STROKE} vertical={horizontal} horizontal={!horizontal} />
            {horizontal ? (
              <>
                <XAxis
                  type="number" stroke={AXIS_STROKE} fontSize={11}
                  label={valAxis ? {
                    value: valAxis, position: "insideBottom", offset: -14,
                    fill: AXIS_STROKE, fontSize: 11,
                  } : undefined}
                />
                <YAxis
                  type="category" dataKey="__label" stroke={AXIS_STROKE}
                  width={catWidth} fontSize={11} interval={0}
                  tickFormatter={(v) => elide(v, catChars)}
                />
              </>
            ) : (
              <>
                <XAxis
                  dataKey="__label" stroke={AXIS_STROKE} fontSize={11} interval={0}
                  angle={rows.length > 6 ? -32 : 0}
                  textAnchor={rows.length > 6 ? "end" : "middle"}
                  height={rows.length > 6 ? 62 : 28}
                  // Upright ticks have only the column to themselves; rotated
                  // ones only the gutter. Elide to what fits and let the
                  // tooltip carry the rest.
                  tickFormatter={(v) => elide(v, rows.length > 6 ? 14 : 18)}
                  label={block.x_label ? {
                    value: block.x_label, position: "insideBottom", offset: -16,
                    fill: AXIS_STROKE, fontSize: 11,
                  } : undefined}
                />
                <YAxis
                  stroke={AXIS_STROKE} fontSize={11}
                  label={valAxis ? {
                    value: valAxis, angle: -90, position: "insideLeft",
                    fill: AXIS_STROKE, fontSize: 11,
                  } : undefined}
                />
              </>
            )}
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              cursor={{ fill: "rgba(91, 157, 255, 0.08)" }}
              formatter={(v, n) => [formatCount(Number(v)), String(n)]}
              labelFormatter={(l) => String(l)}
            />
            {block.series.map((s, i) => hidden.has(s.name) ? null : (
              <Bar
                key={s.name}
                dataKey={keys[i]}
                name={s.name}
                fill={seriesColor(i)}
                stackId={block.stacked ? "one" : undefined}
                radius={horizontal ? [0, 3, 3, 0] : [3, 3, 0, 0]}
                // Without a cap, a two-category chart draws bars a third of the
                // panel wide, which reads as an area chart rather than as two
                // measurements to compare.
                maxBarSize={54}
                // A block lives in a transcript that re-renders whenever the
                // conversation moves, and recharts restarts its grow animation
                // on every re-render: the bars would replay on each new message,
                // and a chart interrupted mid-grow shows a height that is a
                // function of timing rather than of the data.
                isAnimationActive={false}
                cursor={drillable ? "pointer" : undefined}
                onClick={(data: unknown) => {
                  const drill = drillOf(data);
                  if (drill) open(drill);
                }}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* The chart is `role="img"`, so its bars are invisible to a screen
          reader and unreachable by keyboard. This list is the same data as
          text, and it is where the drill-down lives for anyone not using a
          mouse — not a duplicate, the accessible half of the control. */}
      <details className="vblock-values">
        <summary>
          {drillable
            ? `Values (${rows.length}) — open a slice in the gallery`
            : `Values (${rows.length})`}
        </summary>
        <ul>
          {rows.map((r) => {
            // Only the series currently on the chart, so hiding one hides it
            // here too and the list never disagrees with the picture.
            const cells = block.series
              .map((s, si) => ({ name: s.name, value: r[keys[si]] }))
              .filter(({ name }) => !hidden.has(name))
              .map(({ name, value }) =>
                `${multi ? `${name}: ` : ""}${typeof value === "number" ? formatCount(value) : "—"}`)
              .join("  ·  ");
            return (
              <li key={r.__label}>
                {r.__drill !== null ? (
                  <button
                    type="button" className="vblock-value-link"
                    title={`Open this slice in the gallery (?${r.__drill})`}
                    onClick={() => open(r.__drill as string)}
                  >
                    <span className="vblock-value-label">{r.__label}</span>
                    <span className="vblock-value-num">{cells}</span>
                  </button>
                ) : (
                  <span className="vblock-value-static">
                    <span className="vblock-value-label">{r.__label}</span>
                    <span className="vblock-value-num">{cells}</span>
                  </span>
                )}
              </li>
            );
          })}
        </ul>
      </details>
    </div>
  );
}
