import { sequential } from "../lib/viz";

/** The only score bases that are similarities. A cosine has a scale a person
 * can reason about — 0.31 really is further than 0.44 — so a distribution and
 * a line drawn through it mean something. `rrf` is a rank score: its magnitude
 * says nothing about how alike two things are, and `composed` is only
 * comparable inside its own result list. Neither gets a distribution, because
 * the number a threshold would show would be meaningless. */
export const COSINE_BASES = new Set(["cosine", "cosine_adj"]);
const DIST_BUCKETS = 24;

/**
 * The visible results' scores as a shape, with the corpus's own similarity
 * floor drawn on it, and a line the reader can move.
 *
 * This is a VIEW over the ranking that is already on screen: it re-ranks
 * nothing, requests nothing, and stays out of the URL — dimming is how one
 * person reads one result set, not a filter a link should reproduce (the
 * shareable artifacts are the query and the `?ids=` slice). Everything below
 * the line stays on screen, greyed: absence of a match above the floor is
 * itself a finding, and a hidden cutoff would conceal it.
 *
 * The floor is measured, never assumed — the 10th percentile of every image's
 * nearest-neighbour cosine, published by the active index's manifest and read
 * from `/api/stats/overview`. An index that publishes none gets no line and
 * says so; inventing a plausible number here would be the worst kind of lie,
 * because it would look exactly like a measurement.
 *
 * And a measurement is only worth drawing if the reader knows what it measured:
 * that percentile is taken BETWEEN IMAGES, while a searched ranking scores a
 * query against images. Same space, same units, different population — so the
 * line is labelled as the corpus's own scale, never as a relevance cutoff.
 */
export default function ScoreDistribution({ scores, basis, floor, threshold, onThreshold }: {
  scores: number[];
  basis: string;
  /** undefined while the corpus figure is still being read; null when the
   * active index publishes none. */
  floor: number | null | undefined;
  threshold: number;
  onThreshold: (v: number) => void;
}) {
  const lo = Math.min(...scores);
  const hi = Math.max(...scores);
  const span = hi - lo;
  const bins = new Array<number>(DIST_BUCKETS).fill(0);
  for (const s of scores) {
    bins[Math.min(DIST_BUCKETS - 1, Math.floor(((s - lo) / span) * DIST_BUCKETS))] += 1;
  }
  const peak = Math.max(...bins);
  const dimmed = scores.filter((s) => s < threshold).length;
  const adjusted = basis === "cosine_adj";
  /* The floor is a RAW-cosine statistic — the 10th percentile of every image's
   * nearest-neighbour cosine — so it may only be drawn on a raw-cosine axis.
   * `cosine_adj` is that cosine after the hubness penalty is subtracted, which
   * moves every score by an amount that depends on the query bank: same units,
   * different quantity. Drawing the raw floor across it would put two scales on
   * one axis and invite exactly the comparison this app refuses everywhere
   * else. So cos* gets the distribution and the threshold, and no floor. */
  const floorApplies = !adjusted;
  // The floor only marks the chart when it falls inside what is on screen; a
  // line pinned to an edge would claim a precision the pixels do not have.
  const floorIn = floorApplies && floor != null && floor > lo && floor < hi;
  const floorAt = floorIn ? Math.round(((floor - lo) / span) * DIST_BUCKETS) : -1;

  const bars: React.ReactNode[] = [];
  for (let i = 0; i < DIST_BUCKETS; i++) {
    if (i === floorAt) {
      bars.push(
        <span key="floor" aria-hidden="true"
              title={`Measured similarity floor ${floor?.toFixed(4)} — 10th percentile `
                     + "of nearest-neighbour (image-to-image) cosine in this corpus"}
              style={{ flex: "0 0 0px", alignSelf: "stretch",
                       borderLeft: "1.5px dashed var(--accent-deep)" }} />,
      );
    }
    const binLo = lo + (span * i) / DIST_BUCKETS;
    const binHi = lo + (span * (i + 1)) / DIST_BUCKETS;
    bars.push(
      <div key={i}
           // Opacity carries the threshold, colour carries position on the
           // scale — the same division of labour as the quality histogram.
           className={`dist-bar ${binHi <= threshold ? "out" : ""}`}
           style={{ height: `${Math.max(2, (bins[i] / peak) * 100)}%`,
                    background: sequential(i / (DIST_BUCKETS - 1)) }}
           title={`${binLo.toFixed(3)}–${binHi.toFixed(3)}: ${bins[i]} result`
                  + `${bins[i] === 1 ? "" : "s"}`} />,
    );
  }

  return (
    <div className="dist-panel">
      <div className="dist-head">
        <div>
          <div className="eyebrow">Score distribution</div>
          <div className="meta-line" style={{ marginBottom: 0, marginTop: 4 }}>
            {scores.length.toLocaleString()} visible result
            {scores.length === 1 ? "" : "s"} ·{" "}
            {adjusted
              ? "hubness-adjusted semantic score (cos*) — cosine minus a "
                + "query-bank penalty, so not a raw cosine"
              : "cosine similarity"}{" "}
            · a view over this ranking — it re-ranks nothing and changes no link
          </div>
        </div>
        {/* The consequence of the line, not the range: the range is already
            printed under the bars, and at 390 a nowrap readout long enough to
            hold both spills out of the panel (measured: 385px right edge in a
            374px card). */}
        <div className="dist-readout">
          {dimmed.toLocaleString()} of {scores.length.toLocaleString()} dimmed
        </div>
      </div>

      <div className="dist-bars" role="img"
           aria-label={`Histogram of ${scores.length} `
                       + `${adjusted ? "hubness-adjusted (cos*)" : "cosine"} scores from `
                       + `${lo.toFixed(3)} to ${hi.toFixed(3)}; `
                       + (floorIn ? `measured floor at ${floor?.toFixed(4)}; ` : "")
                       + `${dimmed} dimmed below ${threshold.toFixed(3)}`}>
        {bars}
      </div>
      <div className="dist-axis">
        <span>{lo.toFixed(3)}</span>
        <span>lower ← {adjusted ? "cos*" : "cosine"} → higher</span>
        <span>{hi.toFixed(3)}</span>
      </div>

      <div className="dist-control">
        <label className="eyebrow" htmlFor="score-threshold">Dim below</label>
        <input id="score-threshold" type="range"
               min={lo} max={hi} step={span / 200} value={threshold}
               onChange={(e) => onThreshold(Number(e.target.value))} />
        <span className="dist-readout">{threshold.toFixed(3)}</span>
      </div>

      {/* Where the line came from, in one sentence, every time. A floor
          nobody measured is worse than no floor at all. */}
      <div className="meta-line" style={{ marginTop: 8, marginBottom: 0 }}>
        {adjusted
          ? "No floor is drawn here. The corpus's measured floor is a RAW cosine "
            + "(the 10th percentile of every image's nearest-neighbour cosine), "
            + "and these scores are cos* — that cosine minus a query-bank hubness "
            + "penalty. Same units, different quantity, so the two do not belong "
            + "on one axis. Switch to a raw-cosine ranking to see the floor."
          : floor === undefined
          ? "Reading this index's measured similarity floor…"
          : floor == null
            ? "This index publishes no measured similarity floor, so none is "
              + "drawn — the line above is yours, not the corpus's. The figure "
              + "would come from the embeddings manifest as sim_floor_p10 "
              + "(served as sim_floor by GET /api/stats/overview)."
            : `Dashed line: ${floor.toFixed(4)} — the measured 10th percentile of `
              + "every image's nearest-neighbour cosine in this corpus (GET "
              + "/api/stats/overview sim_floor). Measured between images, while these "
              + "are query-to-image scores: read it as the corpus's own scale, not as "
              + "a relevance cutoff"
              + `${floorIn ? "" : " — it falls outside the range of these results, so "
                              + "no line is drawn"}.`}
      </div>
    </div>
  );
}
