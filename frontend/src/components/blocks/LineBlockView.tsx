import { useMemo, useState } from "react";
import {
  Area, Brush, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip,
  XAxis, YAxis,
} from "recharts";
import type { LineBlock, XYSeries } from "../../api/blocks";
import {
  AXIS_STROKE, GRID_STROKE, SERIES, SURFACE, TOOLTIP_STYLE, categorical,
} from "../../lib/viz";

const NAMED = [SERIES.blue, SERIES.green, SERIES.purple, SERIES.amber];
function seriesColor(i: number): string {
  return i < NAMED.length ? NAMED[i] : categorical(i);
}

/** A line block's y values are as often a ratio (recall@5, mean agreement) as a
 * count, so the axis and tooltip keep fractions instead of rounding a 0.42 to
 * zero. Trailing zeros are dropped: a column of "0.4200" reads as false
 * precision the measurement does not have. */
function fmt(n: number): string {
  if (!Number.isFinite(n)) return "—";
  return Number.isInteger(n)
    ? n.toLocaleString()
    : n.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

interface Row {
  x: number;
  [key: string]: number | undefined;
}

/** Series can be sampled at different x values (two eval runs at different
 * depths), so rows are the union of x values with gaps left as `undefined` for
 * `connectNulls` to bridge — rather than zero, which would draw a dip that is
 * not in the data. Synthetic keys for the same reason as the bar block: a
 * series name is not a safe `dataKey`. */
function mergeOnX(series: XYSeries[]): { rows: Row[]; keys: string[] } {
  const keys = series.map((_, i) => `s${i}`);
  const byX = new Map<number, Row>();
  series.forEach((s, i) => {
    for (const p of s.points) {
      const x = Number(p.x);
      let row = byX.get(x);
      if (row === undefined) {
        row = { x };
        byX.set(x, row);
      }
      row[keys[i]] = Number(p.y);
    }
  });
  return { rows: [...byX.values()].sort((a, b) => a.x - b.x), keys };
}

export default function LineBlockView({ block }: { block: LineBlock }) {
  const [hidden, setHidden] = useState<ReadonlySet<string>>(new Set());
  const { rows, keys } = useMemo(() => mergeOnX(block.series), [block.series]);

  const multi = block.series.length > 1;
  const area = block.area === true;

  const toggle = (name: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name); else next.add(name);
      return next.size === block.series.length ? prev : next;
    });

  const first = rows.length > 0 ? rows[0].x : 0;
  const last = rows.length > 0 ? rows[rows.length - 1].x : 0;
  const label = `${block.title}. Line chart of ${block.series.length} series`
    + ` over ${rows.length} points`
    + (rows.length > 0 ? `, x from ${fmt(first)} to ${fmt(last)}` : "")
    + `. ${block.series.map((s) => s.name).join(", ")}.`;

  // A brush needs somewhere to sit; only claim the space when there is enough
  // data for zooming to mean anything.
  const brushable = rows.length > 8;

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
        <ResponsiveContainer width="100%" height={brushable ? 288 : 250}>
          <ComposedChart
            data={rows}
            margin={{ top: 6, right: 16, left: 4, bottom: block.x_label ? 18 : 2 }}
          >
            <CartesianGrid stroke={GRID_STROKE} vertical={false} />
            <XAxis
              dataKey="x" type="number" domain={["dataMin", "dataMax"]}
              stroke={AXIS_STROKE} fontSize={11} tickFormatter={fmt}
              label={block.x_label ? {
                value: block.x_label, position: "insideBottom", offset: -14,
                fill: AXIS_STROKE, fontSize: 11,
              } : undefined}
            />
            <YAxis
              stroke={AXIS_STROKE} fontSize={11} tickFormatter={fmt}
              label={block.y_label ? {
                value: block.y_label, angle: -90, position: "insideLeft",
                fill: AXIS_STROKE, fontSize: 11,
              } : undefined}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={(v, n) => [fmt(Number(v)), String(n)]}
              labelFormatter={(l) => `${block.x_label ?? "x"} ${fmt(Number(l))}`}
            />
            {block.series.map((s, i) => {
              if (hidden.has(s.name)) return null;
              const color = seriesColor(i);
              return area ? (
                <Area
                  key={s.name} type="monotone" dataKey={keys[i]} name={s.name}
                  stroke={color} fill={color} fillOpacity={0.16} strokeWidth={2}
                  connectNulls dot={false} activeDot={{ r: 4 }}
                  // See BarBlockView: the transcript re-renders often, and a
                  // partially drawn line is a partially wrong line.
                  isAnimationActive={false}
                />
              ) : (
                <Line
                  key={s.name} type="monotone" dataKey={keys[i]} name={s.name}
                  stroke={color} strokeWidth={2}
                  connectNulls dot={false} activeDot={{ r: 4 }}
                  isAnimationActive={false}
                />
              );
            })}
            {brushable && (
              <Brush
                dataKey="x" height={24} travellerWidth={9}
                stroke={SURFACE.borderStrong} tickFormatter={fmt}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {brushable && (
        <p className="vblock-hint">
          Drag the handles under the chart to zoom into a range of{" "}
          {block.x_label ?? "x"}.
        </p>
      )}
    </div>
  );
}
