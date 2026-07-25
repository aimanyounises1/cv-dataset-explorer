import { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api/client";
import type { EvalResponse } from "../api/types";
import { AXIS_STROKE, GRID_STROKE, SERIES, TOOLTIP_STYLE } from "../lib/viz";

const MODE_COLORS: Record<string, string> = {
  semantic: SERIES.blue,
  keyword: SERIES.amber,
  hybrid: SERIES.green,
};

/** Self-benchmark: each of the dataset's own captions should retrieve its own
 * image (standard Flickr8k text→image retrieval protocol). */
export default function EvalPage() {
  const [result, setResult] = useState<EvalResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      setResult(await api.evalRetrieval(1000));
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
      <div className="section-title" style={{ marginTop: 0 }}>Search quality benchmark</div>
      <p className="meta-line" style={{ maxWidth: 720 }}>
        Every caption in Flickr8k is ground truth: querying with a caption should
        retrieve its own image. This runs the standard text→image retrieval
        protocol against all three search modes. The query caption is excluded
        from the index it searches, so a mode has to find the image through the
        other four captions rather than matching itself; without that exclusion
        the number would measure nothing but self-retrieval.
      </p>
      <p className="meta-line" style={{ maxWidth: 720 }}>
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
      {result && !result.available && <div className="notice">{result.message}</div>}

      {result?.available && (
        <>
          {/* Recall@k is meaningless without the pool it was computed over:
              published Flickr8k baselines rank against 1,000 candidates, so a
              number measured against the full corpus is a harder task and not
              comparable to them. State the pool next to the metric, always. */}
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
          <div className="panel" style={{ maxWidth: 720 }}>
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
                <Legend />
                {Object.entries(MODE_COLORS).map(([mode, color]) => (
                  <Bar key={mode} dataKey={mode} fill={color} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <table className="eval-table">
            <thead>
              <tr>
                <th>mode</th><th>R@1</th><th>R@5</th><th>R@10</th>
                <th title={`Mean reciprocal rank within the top ${result.depth}`}>MRR@{result.depth}</th>
                <th title="Median rank of the correct image (lower is better)">median rank</th>
                <th title="How many images this mode actually had to rank, averaged over queries">
                  candidates
                </th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((r) => (
                <tr key={r.mode}>
                  <td>{r.mode}</td>
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
        </>
      )}
    </div>
  );
}
