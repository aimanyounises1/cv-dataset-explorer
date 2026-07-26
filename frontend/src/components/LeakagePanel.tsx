import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { LeakageReport } from "../api/types";
import "../styles/leakage.css";

/**
 * Held-out images that have a near-duplicate in training.
 *
 * This is the highest-consequence thing embeddings alone can detect about a
 * dataset: a test image with a training twin means reported accuracy on it is
 * partly memorisation. Barz & Denzler measured 3.3% of CIFAR-10 and 10% of
 * CIFAR-100 test images in that state, worth 9-14% relative accuracy.
 *
 * The design decision that matters here is showing a **curve, not a number**.
 * "Near-duplicate" is a threshold on a cosine, not a fact, and on this corpus
 * the answer swings from 0.8% at 0.95 to 12.1% at 0.90. A headline figure at a
 * hard-coded cut would be an arbitrary choice wearing the costume of a
 * measurement. So the slider is the primary control, the whole ladder is always
 * visible, and the pairs are shown large enough to judge — because the only real
 * verification is looking at two images and deciding whether they are the same.
 */
export default function LeakagePanel() {
  const [threshold, setThreshold] = useState(0.9);
  const [data, setData] = useState<LeakageReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    const t = setTimeout(() => {
      api.leakage(threshold, ctrl.signal)
        .then(setData)
        .catch((e) => {
          if (e instanceof DOMException && e.name === "AbortError") return;
          setError(e instanceof Error ? e.message : String(e));
        });
    }, 180);
    return () => { clearTimeout(t); ctrl.abort(); };
  }, [threshold]);

  if (error) {
    return <div className="notice">Leakage check unavailable — {error}</div>;
  }
  if (!data) return <div className="loading">Scanning for near-duplicates…</div>;

  const peak = Math.max(1, ...data.curve.map((p) => p.contaminated));

  return (
    <div className="leakage">
      <div className="leakage-head">
        <div>
          <div className="leak-figure">
            {data.contaminated.toLocaleString()}
            <span className="leak-of"> / {data.held_out_total.toLocaleString()}</span>
          </div>
          <div className="leak-label">
            held-out images with a training near-duplicate ·{" "}
            <strong>{(data.contaminated_fraction * 100).toFixed(2)}%</strong>
          </div>
        </div>
        <div className="leak-side">
          <div><strong>{data.pairs.toLocaleString()}</strong> near-duplicate pairs</div>
          <div><strong>{data.cross_split_pairs.toLocaleString()}</strong> cross split</div>
          {Object.entries(data.by_split_pair).map(([k, v]) => (
            <div key={k} className="leak-pairtype">{k.replace("~", " ↔ ")} · {v}</div>
          ))}
        </div>
      </div>

      <div className="leak-control">
        <label className="eyebrow" htmlFor="leak-th">Similarity threshold</label>
        <input id="leak-th" type="range" min={data.floor} max={0.99} step={0.01}
               value={threshold}
               onChange={(e) => setThreshold(Number(e.target.value))} />
        <span className="leak-th-val">{threshold.toFixed(2)}</span>
      </div>

      {/* The whole ladder, always. A reader who sees only the current cut cannot
          tell a robust finding from an artefact of where the slider happens to be. */}
      <div className="leak-curve" role="img"
           aria-label={"Contaminated held-out images by threshold: "
             + data.curve.map((p) => `${p.threshold} gives ${p.contaminated}`).join("; ")}>
        {data.curve.map((p) => (
          <button key={p.threshold} className={`leak-rung ${
                    Math.abs(p.threshold - threshold) < 0.005 ? "active" : ""}`}
                  onClick={() => setThreshold(p.threshold)}
                  title={`At cosine ${p.threshold}: ${p.contaminated} contaminated `
                         + `held-out images, ${p.cross_split} cross-split pairs of `
                         + `${p.pairs} total`}>
            <span className="leak-bar"
                  style={{ height: `${Math.max(3, (p.contaminated / peak) * 100)}%` }} />
            <span className="leak-rung-n">{p.contaminated}</span>
            <span className="leak-rung-th">{p.threshold.toFixed(2)}</span>
          </button>
        ))}
      </div>

      {data.examples.length > 0 && (
        <>
          <div className="eyebrow" style={{ marginTop: 16 }}>
            Cross-split pairs — judge these yourself
          </div>
          <div className="leak-pairs">
            {data.examples.filter((e) => e.cross_split).slice(0, 8).map((e) => (
              <div className="leak-pair" key={`${e.a_id}-${e.b_id}`}>
                <Link to={`/samples/${e.a_id}`} title={`Sample ${e.a_id} (${e.a_split})`}>
                  <img src={e.a_thumb} alt="" loading="lazy" />
                  <span className={`leak-split ${e.a_split}`}>{e.a_split}</span>
                </Link>
                <span className="leak-score">{e.score.toFixed(3)}</span>
                <Link to={`/samples/${e.b_id}`} title={`Sample ${e.b_id} (${e.b_split})`}>
                  <img src={e.b_thumb} alt="" loading="lazy" />
                  <span className={`leak-split ${e.b_split}`}>{e.b_split}</span>
                </Link>
              </div>
            ))}
          </div>
        </>
      )}

      <p className="leak-caveat">{data.caveat}</p>
      <p className="leak-caveat">
        This does not affect the <Link to="/eval">retrieval benchmark</Link> on this
        page’s sibling tab: that trains nothing and ranks every query against the
        whole corpus. It matters for anyone who <em>trains</em> on this train split
        and reports accuracy on its test split.
      </p>
    </div>
  );
}
