import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { QASelection, QASummary, SuspectCaption } from "../api/types";
import { SURFACE, sequential } from "../lib/viz";

/** Where to put the review threshold before the user touches it.
 *
 * A fixed top-50 answered "show me some bad captions"; a threshold answers
 * "show me everything below this line", which is the question that scales. The
 * default is the value below which ~1% of captions fall — chosen from the
 * distribution rather than hard-coded, so it means the same thing on a corpus
 * with a different agreement spread. */
function defaultThreshold(summary: QASummary): number {
  const total = summary.histogram.reduce((n, b) => n + b.count, 0);
  if (!total) return summary.max_agreement ?? 1;
  let seen = 0;
  for (const b of summary.histogram) {
    seen += b.count;
    if (seen / total >= 0.01) return b.hi;
  }
  return summary.max_agreement ?? 1;
}

function SuspectRow({ item, scoreLabel }: { item: SuspectCaption; scoreLabel: string }) {
  return (
    <Link className="suspect-row" to={`/samples/${item.sample.id}`}>
      <img src={item.sample.thumb_url} alt="" loading="lazy" />
      <div className="suspect-body">
        <div className="suspect-caption">“{item.caption}”</div>
        <div className="suspect-meta">
          <span className="pill warn-pill">{scoreLabel} {item.agreement.toFixed(3)}</span>
          {item.sibling_mean != null && (
            <span className="pill" title="Mean agreement of the sample's other captions">
              siblings {item.sibling_mean.toFixed(3)}
            </span>
          )}
          <span className="pill">{item.sample.split}</span>
        </div>
      </div>
    </Link>
  );
}

/** Annotation QA: CLIPScore-style ranking of captions least supported by
 * their image — the mislabel-hunting workflow. */
export default function QualityPage() {
  const [summary, setSummary] = useState<QASummary | null>(null);
  const [suspects, setSuspects] = useState<SuspectCaption[]>([]);
  const [inconsistent, setInconsistent] = useState<SuspectCaption[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sectionErrors, setSectionErrors] = useState<string[]>([]);
  const [threshold, setThreshold] = useState<number | null>(null);
  const [listLoading, setListLoading] = useState(false);
  const [selection, setSelection] = useState<QASelection | null>(null);

  useEffect(() => {
    // On a QA page, a swallowed error rendering as "no problems found" is the
    // worst failure mode — failed sections must announce themselves.
    const fail = (what: string) => () =>
      setSectionErrors((prev) => [...prev, what]);
    api.qaSummary()
      .then((s) => { setSummary(s); if (s.available) setThreshold(defaultThreshold(s)); })
      .catch((e) => setError(String(e)));
    api.inconsistentSamples().then(setInconsistent).catch(fail("consistency ranking"));
  }, []);

  // The list and the selection size both follow the threshold. Debounced
  // because the control is a slider and every intermediate value would
  // otherwise be a request.
  useEffect(() => {
    if (threshold == null) return;
    setListLoading(true);
    const ctrl = new AbortController();
    const t = setTimeout(() => {
      api.suspectCaptions({ limit: 100, max_agreement: threshold })
        .then(setSuspects)
        .catch(() => setSectionErrors((p) =>
          p.includes("suspect captions") ? p : [...p, "suspect captions"]))
        .finally(() => setListLoading(false));
      // Counted in SQL, not from the histogram: bins that straddle the line can
      // only give a rounded answer, and this number labels a button that hands
      // the set to another view — so it has to be the number that view shows.
      api.qaSelection(threshold, ctrl.signal)
        .then(setSelection)
        .catch(() => { /* the readout falls back to the binned estimate */ });
    }, 220);
    return () => { clearTimeout(t); ctrl.abort(); };
  }, [threshold]);

  const hist = summary?.histogram ?? [];
  const peak = useMemo(() => Math.max(1, ...hist.map((b) => b.count)), [hist]);
  // Binned estimate, used only until the exact count arrives (and if it fails).
  const binnedBelow = useMemo(
    () => (threshold == null ? 0
      : hist.filter((b) => b.hi <= threshold).reduce((n, b) => n + b.count, 0)),
    [hist, threshold]);
  const exact = selection != null && selection.max_agreement === threshold;
  const belowCount = exact ? selection!.captions : binnedBelow;
  const totalScored = summary?.scored_captions ?? 0;

  if (error) return <div className="error">{error}</div>;
  if (!summary) return <div className="loading">Loading QA…</div>;

  if (!summary.available) {
    return (
      <div className="empty">
        Caption QA scores are not computed yet.<br />
        Run <code>python -m app.analyze</code> in the backend, then{" "}
        <code>POST /api/admin/reload</code> (or restart the API).
      </div>
    );
  }

  return (
    <div>
      <div className="stat-cards">
        <div className="stat-card">
          <div className="value">{summary.scored_captions.toLocaleString()}</div>
          <div className="label">Captions scored (SigLIP agreement)</div>
        </div>
        <div className="stat-card">
          <div className="value">{summary.mean_agreement?.toFixed(3) ?? "—"}</div>
          <div className="label">Mean image-caption agreement</div>
        </div>
      </div>

      {sectionErrors.length > 0 && (
        <div className="error">Could not load: {sectionErrors.join(", ")}.</div>
      )}

      {/* The distribution, and a line you can move through it. A fixed top-50
          hid the one thing a reviewer needs to judge: whether the cutoff is in
          the tail or well inside the bulk. */}
      {hist.length > 0 && threshold != null && (
        <div className="dist-panel">
          <div className="dist-head">
            <div>
              <div className="eyebrow">Agreement distribution</div>
              <div className="meta-line" style={{ marginBottom: 0, marginTop: 4 }}>
                {totalScored.toLocaleString()} scored captions · mean{" "}
                {summary.mean_agreement?.toFixed(3)}
              </div>
            </div>
            <div className="dist-readout">
              ≤ {threshold.toFixed(3)} · {belowCount.toLocaleString()} captions (
              {((belowCount / Math.max(1, totalScored)) * 100).toFixed(1)}%)
            </div>
          </div>

          <div className="dist-bars" role="img"
               aria-label={`Agreement histogram; threshold at ${threshold.toFixed(3)} selects `
                           + `${belowCount} of ${totalScored} captions`}>
            {hist.map((b) => {
              const inRange = b.hi <= threshold;
              return (
                <div
                  key={b.lo}
                  className={`dist-bar ${inRange ? "" : "out"}`}
                  style={{
                    height: `${Math.max(2, (b.count / peak) * 100)}%`,
                    // Colour by position on the scale, not by selection state —
                    // opacity carries selection, so the two do not compete.
                    background: sequential(
                      (b.lo - (summary.min_agreement ?? 0)) /
                      Math.max(1e-6, (summary.max_agreement ?? 1) - (summary.min_agreement ?? 0))),
                  }}
                  title={`${b.lo.toFixed(3)}–${b.hi.toFixed(3)}: ${b.count.toLocaleString()} captions`}
                />
              );
            })}
          </div>
          <div className="dist-axis">
            <span>{summary.min_agreement?.toFixed(3)}</span>
            <span>worse ← agreement → better</span>
            <span>{summary.max_agreement?.toFixed(3)}</span>
          </div>

          <div className="dist-control">
            <label className="eyebrow" htmlFor="qa-threshold">Review below</label>
            <input
              id="qa-threshold"
              type="range"
              min={summary.min_agreement ?? 0}
              max={summary.max_agreement ?? 1}
              step={0.001}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
            <span className="dist-readout" style={{ color: SURFACE.textDim }}>
              {listLoading ? "updating…" : `${suspects.length} shown`}
            </span>
          </div>

          {/* A brush that only redraws its own page is a setting, not a
              selection. These carry the same predicate to the gallery and to
              export, so the triage set outlives this view. The list below is
              capped at 100 rows; the set itself is not, which is exactly why
              the hand-off has to exist. */}
          <div className="dist-actions">
            <Link className="primary button-link"
                  to={`/?max_agreement=${threshold}`}
                  title="Open these images in the gallery, where the threshold becomes a removable filter">
              Review {exact ? selection!.samples.toLocaleString() : "these"} images
              in gallery
            </Link>
            <span className="export-label">Export selection</span>
            {(["csv", "jsonl", "json"] as const).map((f) => (
              <a key={f} className="pill export-pill"
                 href={`/api/export?max_agreement=${threshold}&format=${f}`}
                 title={`Download every image with a caption at or below ${threshold.toFixed(3)} as ${f.toUpperCase()}`}>
                {f}
              </a>
            ))}
            {exact && selection!.captions !== selection!.samples && (
              <span className="meta-line" style={{ marginBottom: 0 }}>
                {selection!.captions.toLocaleString()} captions across{" "}
                {selection!.samples.toLocaleString()} images
              </span>
            )}
          </div>
        </div>
      )}

      <div className="section-title">Most suspect captions</div>
      <p className="meta-line">
        Lowest image-caption agreement first. Low score + high sibling mean ⇒ the
        caption is likely wrong; all-low ⇒ the image itself is unusual.
      </p>
      {suspects.length === 0 ? (
        <div className="empty">No scored captions to show.</div>
      ) : (
        <div className="suspect-list">
          {suspects.map((s, i) => (
            <SuspectRow key={`${s.sample.id}-${i}`} item={s} scoreLabel="agreement" />
          ))}
        </div>
      )}

      {inconsistent.length > 0 && (
        <>
          <div className="section-title">Least consistent samples</div>
          <p className="meta-line">
            Samples whose 5 captions disagree most with each other — ambiguous
            images or outlier annotations.
          </p>
          <div className="suspect-list">
            {inconsistent.map((s, i) => (
              <SuspectRow key={`${s.sample.id}-${i}`} item={s} scoreLabel="consistency" />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
