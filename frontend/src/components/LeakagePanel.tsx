import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { LeakageContamination, LeakageReport } from "../api/types";

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
  const [hit, setHit] = useState<LeakageContamination | null>(null);
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
      /* The ids ride alongside the counts, on the same cut. Fetching them only
         when someone reaches for the button would let the set drift from the
         figure above it whenever the slider moved in between — and a leakage
         set that disagrees with its own headline is worse than none. A failure
         here costs the hand-off, not the report, so it clears the ids and
         leaves the measurement standing. */
      setHit(null);
      api.leakageContaminated(threshold, ctrl.signal)
        .then(setHit)
        .catch(() => { /* the panel simply offers no hand-off */ });
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

      {/* The measurement was the expensive half; handing the set over is the
          cheap half that makes it worth having. Without this the panel reports
          that 12% of the held-out split is contaminated and leaves the reader
          to re-derive *which* 240 images in numpy — at which point they did not
          need the tool. The ids are the gallery's own `?ids=` vocabulary, so
          the slice opens, exports, tags and excludes like any other.

          `dist-actions`, not a class of its own: this is the same hand-off the
          caption-quality panel makes — a primary link into the gallery beside
          the export formats — and it should not look like a different idea
          because it sits on a different page. */}
      {hit && hit.total > 0 && (
        <div className="dist-actions">
          <Link className="primary button-link"
                to={`/?ids=${hit.ids.join(",")}`}
                title="Open the contaminated held-out images in the gallery, where they behave like any other selection">
            Open {hit.total.toLocaleString()} contaminated images
          </Link>
          <span className="export-label">Export the set</span>
          {(["csv", "jsonl", "json"] as const).map((f) => (
            <a key={f} className="pill export-pill"
               href={`/api/export?ids=${hit.ids.join(",")}&format=${f}`}
               title={`Download the ${hit.total} contaminated held-out images as ${f.toUpperCase()}`}>
              {f}
            </a>
          ))}
          {hit.truncated && (
            <span className="meta-line" style={{ marginBottom: 0 }}>
              showing the first {hit.ids.length.toLocaleString()} — narrow the
              threshold for a set this hand-off can carry whole
            </span>
          )}
        </div>
      )}

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
