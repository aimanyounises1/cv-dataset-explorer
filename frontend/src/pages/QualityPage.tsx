import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { QASummary, SuspectCaption } from "../api/types";

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

  useEffect(() => {
    api.qaSummary().then(setSummary).catch((e) => setError(String(e)));
    api.suspectCaptions({ limit: 50 }).then(setSuspects).catch(() => setSuspects([]));
    api.inconsistentSamples().then(setInconsistent).catch(() => setInconsistent([]));
  }, []);

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

      <div className="section-title">Most suspect captions</div>
      <p className="meta-line">
        Lowest image-caption agreement first. Low score + high sibling mean ⇒ the
        caption is likely wrong; all-low ⇒ the image itself is unusual.
      </p>
      <div className="suspect-list">
        {suspects.map((s, i) => (
          <SuspectRow key={`${s.sample.id}-${i}`} item={s} scoreLabel="agreement" />
        ))}
      </div>

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
