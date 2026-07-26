import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import type { PieBlock } from "../../api/blocks";
import { SURFACE, TOOLTIP_STYLE, categorical, formatCount } from "../../lib/viz";

function drillOf(data: unknown): string | null {
  const d = data as { drill?: unknown; payload?: { drill?: unknown } } | null;
  const v = d?.drill ?? d?.payload?.drill;
  return typeof v === "string" && v !== "" ? v : null;
}

export default function PieBlockView({ block }: { block: PieBlock }) {
  const navigate = useNavigate();
  const points = block.points;
  const total = useMemo(
    () => points.reduce((sum, p) => sum + Number(p.value), 0), [points]);

  const pct = (v: number) => (total > 0 ? `${((v / total) * 100).toFixed(1)}%` : "—");
  const drillable = points.some((p) => Boolean(p.drill));
  const open = (drill: string) => navigate(`/?${drill}`);

  const top = points.reduce<typeof points[number] | null>(
    (best, p) => (best === null || Number(p.value) > Number(best.value) ? p : best), null);
  const label = `${block.title}. Pie chart of ${points.length} categories`
    + ` totalling ${formatCount(total)}`
    + (top ? `; largest is ${top.label} at ${pct(Number(top.value))}` : "") + ".";

  return (
    <div className="vblock-body vblock-pie-body">
      <div className="vblock-chart vblock-pie-chart" role="img" aria-label={label}>
        <ResponsiveContainer width="100%" height={228}>
          <PieChart margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={(v, n) =>
                [`${formatCount(Number(v))} (${pct(Number(v))})`, String(n)]}
            />
            <Pie
              data={points as unknown as Record<string, unknown>[]}
              dataKey="value" nameKey="label"
              cx="50%" cy="50%" innerRadius={44} outerRadius={92}
              // A ring rather than a full disc: the hole carries the total, so
              // the reader does not have to add the slices up to know what the
              // percentages are of.
              stroke={SURFACE.bg} strokeWidth={2} isAnimationActive={false}
              cursor={drillable ? "pointer" : undefined}
              onClick={(data: unknown) => {
                const drill = drillOf(data);
                if (drill) open(drill);
              }}
            >
              {points.map((p, i) => (
                <Cell key={p.label} fill={categorical(i)} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="vblock-pie-total" aria-hidden="true">
          <span className="vblock-pie-total-value">{formatCount(total)}</span>
          <span className="vblock-pie-total-label">total</span>
        </div>
      </div>

      {/* The legend is the keyboard and screen-reader route into the slices:
          the chart itself is `role="img"`, and an SVG path cannot be tabbed to.
          Each entry states its own share, so hue never carries the value. */}
      <ul className="vblock-pie-legend">
        {points.map((p, i) => {
          const value = Number(p.value);
          const body = (
            <>
              <span className="vblock-swatch" style={{ background: categorical(i) }} />
              <span className="vblock-value-label">{p.label}</span>
              <span className="vblock-value-num">
                {formatCount(value)} · {pct(value)}
              </span>
            </>
          );
          return (
            <li key={p.label}>
              {p.drill ? (
                <button
                  type="button" className="vblock-value-link"
                  title={`Open this slice in the gallery (?${p.drill})`}
                  onClick={() => open(p.drill as string)}
                >
                  {body}
                </button>
              ) : (
                <span className="vblock-value-static">{body}</span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
