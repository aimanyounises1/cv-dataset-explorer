import { useNavigate } from "react-router-dom";
import type { StatBlock, StatItem } from "../../api/blocks";

/** Reuses the `.stat-card` look from index.css so a headline figure the agent
 * produced sits at the same weight as one the stats page produced — a number is
 * a number, whoever asked for it. */
function Card({ item }: { item: StatItem }) {
  return (
    <>
      <div className="value">{item.value}</div>
      <div className="label">{item.label}</div>
      {item.hint && <div className="vstat-hint">{item.hint}</div>}
    </>
  );
}

export default function StatBlockView({ block }: { block: StatBlock }) {
  const navigate = useNavigate();

  return (
    <div className="vblock-body">
      <div className="stat-cards vblock-stats">
        {block.items.map((item) => (
          item.drill ? (
            <button
              key={item.label}
              type="button"
              className="stat-card vstat-card drillable"
              title={`Open this slice in the gallery (?${item.drill})`}
              onClick={() => navigate(`/?${item.drill as string}`)}
            >
              <Card item={item} />
              <span className="vstat-go" aria-hidden="true">open →</span>
            </button>
          ) : (
            <div
              key={item.label}
              className="stat-card vstat-card"
              title={item.hint ?? undefined}
            >
              <Card item={item} />
            </div>
          )
        ))}
      </div>
    </div>
  );
}
