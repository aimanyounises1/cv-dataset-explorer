import { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api/client";
import type { EvalResponse } from "../api/types";
import { AXIS_STROKE, GRID_STROKE, SERIES, SURFACE, TOOLTIP_STYLE } from "../lib/viz";

/* The three retrieval modes, in three colours checked as a set rather than
   picked one at a time. `blue` is not among them on purpose: it is the brand
   sage, and pairing it with `green` put semantic and hybrid — the two series
   this whole figure exists to compare — 4.0 ΔE2000 apart, which is below the
   floor at which full-colour vision can separate a pair. */
const MODE_COLORS: Record<string, string> = {
  semantic: SERIES.green,
  keyword: SERIES.amber,
  hybrid: SERIES.purple,
};

/** Self-benchmark: each of the dataset's own captions should retrieve its own
 * image (standard Flickr8k text→image retrieval protocol). */
/* The server caches a completed run and returns it instantly, but the page
 * used to forget its own answer on every navigation and greet a returning
 * user with a bare button implying an expensive operation. The last result
 * rides in sessionStorage — ephemeral tab state, like scroll position — so
 * coming back shows the table immediately and the button says "Re-run". */
const EVAL_RESULT_KEY = "cvde-eval-result";

export default function EvalPage() {
  const [result, setResult] = useState<EvalResponse | null>(() => {
    try {
      const raw = sessionStorage.getItem(EVAL_RESULT_KEY);
      return raw ? (JSON.parse(raw) as EvalResponse) : null;
    } catch { return null; }
  });
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.evalRetrieval(1000);
      setResult(res);
      try { sessionStorage.setItem(EVAL_RESULT_KEY, JSON.stringify(res)); }
      catch { /* non-essential */ }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const chartData = result?.available
    ? ["1", "5", "10"].map((k) => {
        const row: Record<string, number | string> = { k: `R@${k}` };
        for (const r of result.results) row[r.mode] = r.recall_at[k] ?? 0;
        return row;
      })
    : [];

  return (
    <div>
      <h1 className="section-title" style={{ marginTop: 0 }}>Search quality benchmark</h1>
      <p className="meta-line measure">
        Every caption in Flickr8k is ground truth: querying with a caption should
        retrieve its own image. This runs the standard text→image retrieval
        protocol against each search mode. The query caption is excluded
        from the index it searches, so a mode has to find the image through the
        other four captions rather than matching itself; without that exclusion
        the number would measure nothing but self-retrieval.
      </p>
      <p className="meta-line measure">
        <strong>Read the keyword row as a property of the query, not of BM25.</strong>{" "}
        These queries are whole captions, and keyword mode requires every term to
        appear in the same caption — so for most of them no other caption in the
        corpus satisfies the conjunction and the lexical path has nothing to rank
        at all. The “candidates” column below reports that directly. It is a real
        finding about long queries in keyword mode, but it is not evidence that
        lexical ranking is weak, and the fused row inherits the same limitation.
      </p>
      <button className="primary" onClick={() => void run()} disabled={running}>
        {running ? "Running benchmark…" : result ? "Re-run benchmark" : "Run benchmark"}
      </button>

      {error && <div className="error">{error}</div>}
      {/* Any message, not only the unavailable one. The benchmark also attaches a
          message to an AVAILABLE result — when it could not encode the queries
          live and fell back to stored caption vectors, which is a different
          measurement. Gating on `!available` meant that notice could never
          render, so a machine without the model stack would have shown fallback
          numbers as though they were the shipped path. */}
      {result?.message && <div className="notice">{result.message}</div>}

      {result?.available && (
        <>
          {/* Recall@k is meaningless without the pool it was computed over:
              published Flickr8k baselines rank against 1,000 candidates, so a
              number measured against the full corpus is a harder task and not
              comparable to them. State the pool next to the metric, always. */}
          {/* One measure for every paragraph on the page, and it is the same
              one the intro uses — prose that suddenly runs ~200 characters
              wide past the button reads as a different, less finished
              document. Stated as the shared prose rung rather than a repeated
              720, which is how the three copies drifted apart. */}
          <div className="measure">
          <div className="meta-line" style={{ marginTop: 18 }}>
            {result.sample_size.toLocaleString()} caption queries, averaging{" "}
            {result.mean_query_words} words · higher recall is better
          </div>
          <div className="meta-line">
            The semantic path ranks the full corpus of{" "}
            <strong>{result.pool_size.toLocaleString()} images</strong> for every query;
            the lexical path only ever ranks what its query matched, which is what the
            candidates column shows. Not comparable to published Flickr8k numbers, which
            rank against a 1,000-image pool — the full corpus is the harder task.
            MRR and median rank are computed to depth {result.depth}.
          </div>
          <div className="meta-line">
            <strong>Read hybrid’s candidates figure as a sum, not a set.</strong> It is
            the semantic pool plus the mean lexical match count, and every lexical
            match is already inside the semantic pool — so two overlapping sets are
            added where they should be unioned, and the honest figure is the pool
            itself. The overstatement equals the lexical mean: small on this corpus,
            but a defect in the column rather than a rounding artifact. Stated here
            rather than silently re-derived, because changing the number would also
            change what every cached run means.
          </div>
          </div>
          <div className="panel chart-wide">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid stroke={GRID_STROKE} vertical={false} />
                <XAxis dataKey="k" stroke={AXIS_STROKE} />
                <YAxis stroke={AXIS_STROKE} domain={[0, 1]}
                       tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
                <Tooltip
                  contentStyle={TOOLTIP_STYLE}
                  formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`}
                />
                {/* The swatch carries the identity; the word is a label and
                    wears label ink. Recharts paints legend text in the series
                    colour by default, which makes three mode names three
                    different colours of text and puts the burden of telling
                    them apart back on hue. */}
                <Legend formatter={(value: string) => (
                  <span style={{ color: SURFACE.textDim }}>{value}</span>
                )} />
                {/* Bars only for rows the benchmark actually returned — a
                    legend entry for an absent series would read as a zero. */}
                {Object.entries(MODE_COLORS)
                  .filter(([mode]) => result.results.some((r) => r.mode === mode))
                  .map(([mode, color]) => (
                    <Bar key={mode} dataKey={mode} fill={color} radius={[4, 4, 0, 0]} />
                  ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          {/* Seven columns of figures out-measure a phone; the table scrolls in
              its own container so the page never does. */}
          <div className="table-scroll">
          <table className="eval-table">
            <thead>
              <tr>
                <th>mode</th><th>R@1</th><th>R@5</th><th>R@10</th>
                <th title={`Mean reciprocal rank within the top ${result.depth}`}>MRR@{result.depth}</th>
                <th title="Median rank of the correct image (lower is better)">median rank</th>
                <th title={"How many images this mode actually had to rank, averaged over queries. "
                         + "For hybrid the figure sums the semantic pool and the mean keyword pool, "
                         + "so an image both paths returned is counted twice — read it as an upper bound."}>
                  candidates
                </th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((r) => (
                <tr key={r.mode}>
                  <td title={r.note ?? undefined}>
                    {r.mode}
                    {r.queries != null && (
                      <span className="pill" style={{ marginLeft: 6 }}
                            title={`Measured on its own sample of ${r.queries.toLocaleString()} test-split queries (hover the mode name for why)`}>
                        n={r.queries.toLocaleString()}
                      </span>
                    )}
                  </td>
                  {["1", "5", "10"].map((k) => (
                    <td key={k}>{((r.recall_at[k] ?? 0) * 100).toFixed(1)}%</td>
                  ))}
                  <td>{r.mrr.toFixed(3)}</td>
                  {/* Only the semantic path scores the whole pool, so only it can
                      report a median past the evaluated depth. */}
                  <td title={r.median_rank == null
                    ? `The median falls outside the top ${result.depth}, which this path does not rank beyond`
                    : undefined}>
                    {r.median_rank == null ? `> ${result.depth}` : r.median_rank.toFixed(1)}
                  </td>
                  {/* The number that makes the recall column readable. */}
                  <td>
                    {r.mean_candidates >= 1000
                      ? r.mean_candidates.toLocaleString(undefined, { maximumFractionDigits: 0 })
                      : r.mean_candidates.toFixed(2)}
                    {r.empty_query_rate > 0 && (
                      <span className="pill warn-pill" style={{ marginLeft: 6 }}
                            title="Fraction of queries where this mode found no candidates at all">
                        {(r.empty_query_rate * 100).toFixed(0)}% empty
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        </>
      )}
    </div>
  );
}
