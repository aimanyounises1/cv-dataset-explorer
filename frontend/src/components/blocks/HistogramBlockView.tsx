import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  Bar, BarChart, Brush, CartesianGrid, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import type { HistogramBlock } from "../../api/blocks";
import {
  AXIS_STROKE, GRID_STROKE, SERIES, SURFACE, TOOLTIP_STYLE, formatCount,
} from "../../lib/viz";

/** Bin edges are usually fractions (agreement, cosine) and occasionally counts
 * (caption length), so the same formatter has to serve both without printing
 * "12.0000" or rounding 0.05 to 0. */
function fmt(n: number): string {
  if (!Number.isFinite(n)) return "—";
  return Number.isInteger(n)
    ? n.toLocaleString()
    : n.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

interface Bin {
  label: string;
  lo: number;
  hi: number;
  count: number;
}

export default function HistogramBlockView({ block }: { block: HistogramBlock }) {
  const navigate = useNavigate();

  // `bins` is `list[dict]` on the backend — unvalidated — so coerce here rather
  // than let a string "12" sort as a category and break the axis.
  const bins = useMemo<Bin[]>(() => (block.bins ?? []).map((b) => {
    const lo = Number(b.lo);
    const hi = Number(b.hi);
    return { lo, hi, count: Number(b.count), label: `${fmt(lo)}–${fmt(hi)}` };
  }), [block.bins]);

  const total = bins.reduce((s, b) => s + b.count, 0);
  const mode = bins.reduce<Bin | null>(
    (best, b) => (best === null || b.count > best.count ? b : best), null);
  const marker = typeof block.marker === "number" && Number.isFinite(block.marker)
    ? block.marker : null;

  /** A category axis cannot place a line at an arbitrary value, so the marker
   * is drawn on the bin that contains it and its exact value goes in the label.
   * The alternative — a numeric axis — puts recharts in charge of bar widths and
   * leaves hairline bars on non-uniform bins. */
  const markerBin = marker === null ? null
    : bins.find((b) => marker >= b.lo && marker <= b.hi)
      ?? bins.reduce<Bin | null>((best, b) => (
        best === null || Math.abs(b.lo - marker) < Math.abs(best.lo - marker) ? b : best
      ), null);

  const drillParam = block.drill_param ?? null;
  const open = (bin: Bin) => {
    if (drillParam) navigate(`/?${drillParam}=${encodeURIComponent(String(bin.hi))}`);
  };

  const label = `${block.title}. Histogram of ${bins.length} bins`
    + ` covering ${formatCount(total)} items`
    + (bins.length > 0 ? `, ${fmt(bins[0].lo)} to ${fmt(bins[bins.length - 1].hi)}` : "")
    + (mode ? `; the fullest bin is ${mode.label} with ${formatCount(mode.count)}` : "")
    + (marker !== null ? `; ${block.marker_label ?? "marker"} at ${fmt(marker)}` : "")
    + ".";

  const brushable = bins.length > 10;

  return (
    <div className="vblock-body">
      <div className="vblock-chart" role="img" aria-label={label}>
        <ResponsiveContainer width="100%" height={brushable ? 290 : 252}>
          <BarChart
            data={bins}
            margin={{ top: 14, right: 14, left: 4, bottom: block.x_label ? 20 : 2 }}
          >
            <CartesianGrid stroke={GRID_STROKE} vertical={false} />
            <XAxis
              dataKey="label" stroke={AXIS_STROKE} fontSize={10} interval="preserveStartEnd"
              angle={-30} textAnchor="end" height={54}
              label={block.x_label ? {
                value: block.x_label, position: "insideBottom", offset: -16,
                fill: AXIS_STROKE, fontSize: 11,
              } : undefined}
            />
            <YAxis stroke={AXIS_STROKE} fontSize={11} />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              cursor={{ fill: "rgba(91, 157, 255, 0.08)" }}
              formatter={(v) => [formatCount(Number(v)), "items"]}
              labelFormatter={(l) => `${block.x_label ?? "range"} ${String(l)}`}
            />
            {markerBin !== null && marker !== null && (
              <ReferenceLine
                x={markerBin.label}
                stroke={SURFACE.amber}
                strokeDasharray="4 3"
                label={{
                  value: `${block.marker_label ?? "threshold"} ${fmt(marker)}`,
                  position: "top",
                  fill: SURFACE.amber,
                  fontSize: 10.5,
                }}
              />
            )}
            <Bar
              dataKey="count" name="items" fill={SERIES.blue} radius={[3, 3, 0, 0]}
              maxBarSize={54}
              // See BarBlockView: a transcript re-render must not replay the
              // grow animation, and a bar caught mid-grow misstates its count.
              isAnimationActive={false}
              cursor={drillParam ? "pointer" : undefined}
              onClick={(data: unknown) => {
                const d = data as { payload?: Bin; hi?: number } | null;
                const bin = d?.payload ?? (d as Bin | null);
                if (bin && typeof bin.hi === "number") open(bin);
              }}
            />
            {brushable && (
              <Brush
                dataKey="label" height={24} travellerWidth={9}
                stroke={SURFACE.borderStrong}
              />
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="vblock-hint">
        {brushable && "Drag the handles under the chart to zoom into a range. "}
        {markerBin !== null && marker !== null && (
          <>The dashed line marks {fmt(marker)}, drawn on the bin that contains it. </>
        )}
        {drillParam && <>Click a bin to open everything up to its upper edge in the gallery.</>}
      </p>

      {drillParam && (
        // Bars are SVG inside a `role="img"` chart, so this list is the only
        // keyboard route to the same drill-down.
        <details className="vblock-values">
          <summary>Bins ({bins.length}) — open a range in the gallery</summary>
          <ul>
            {bins.map((b) => (
              <li key={b.label}>
                <button
                  type="button" className="vblock-value-link"
                  title={`Open ${drillParam} up to ${fmt(b.hi)} in the gallery`}
                  onClick={() => open(b)}
                >
                  <span className="vblock-value-label">{b.label}</span>
                  <span className="vblock-value-num">{formatCount(b.count)}</span>
                </button>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
