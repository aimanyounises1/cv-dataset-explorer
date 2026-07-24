import { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api/client";
import type { EvalResponse } from "../api/types";

const MODE_COLORS: Record<string, string> = {
  semantic: "#4f9cff",
  keyword: "#ffbe50",
  hybrid: "#3ecf8e",
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
        protocol (recall@k over a 1,000-caption sample) against all three search
        modes — measuring, not assuming, which one is best. Note: keyword search
        is flattered here because the query caption is literally in the index;
        semantic recall is the honest generalization signal.
      </p>
      <button className="primary" onClick={() => void run()} disabled={running}>
        {running ? "Running benchmark…" : result ? "Re-run benchmark" : "Run benchmark"}
      </button>

      {error && <div className="error">{error}</div>}
      {result && !result.available && <div className="notice">{result.message}</div>}

      {result?.available && (
        <>
          <div className="meta-line" style={{ marginTop: 18 }}>
            {result.sample_size.toLocaleString()} caption queries · higher is better
          </div>
          <div className="panel" style={{ maxWidth: 720 }}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid stroke="#2a3342" vertical={false} />
                <XAxis dataKey="k" stroke="#9aa4b2" />
                <YAxis stroke="#9aa4b2" domain={[0, 1]}
                       tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
                <Tooltip
                  contentStyle={{ background: "#161b24", border: "1px solid #2a3342" }}
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
              <tr><th>mode</th><th>R@1</th><th>R@5</th><th>R@10</th></tr>
            </thead>
            <tbody>
              {result.results.map((r) => (
                <tr key={r.mode}>
                  <td>{r.mode}</td>
                  {["1", "5", "10"].map((k) => (
                    <td key={k}>{((r.recall_at[k] ?? 0) * 100).toFixed(1)}%</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
