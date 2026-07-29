import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { SampleDetail, AXES } from "../api/types";
import { AXIS_META } from "./AxisFilters";
import AgreementBadge from "./AgreementBadge";
import { showToast } from "./Toast";
import { VERDICTS, VERDICT_PREFIX, recordedVerdict } from "../lib/verdicts";

/** Sweeping a grid revisits the same handful of images constantly (arrow back
 *  one, then forward again). Keeping the fetched detail for the session makes
 *  that free; it is never invalidated because nothing this panel shows is
 *  mutable from here — captions do not change, and tags live on the sample
 *  page. */
const CACHE = new Map<number, SampleDetail>();

/**
 * The image under the cursor, at a size you can actually judge, beside the five
 * captions that claim to describe it.
 *
 * Deciding whether a caption fits an image is a *visual* act, and until now it
 * cost a page load each way — so in practice nobody did it twice. The grid is
 * where you notice something is wrong; this is where you confirm it, without
 * losing the grid, the scroll position, or your place in the sweep.
 *
 * It shows what the judgement needs and stops: the pixels, the five captions
 * with their measured agreement, and the axes that made this frame stand out.
 * Everything else about a sample — similar images, tags, the full component
 * breakdown, provenance — stays one click away on its own page, which is linked
 * at the bottom. A rail 300px wide that tries to be the sample page is a worse
 * sample page.
 */
export default function Inspector({ id }: { id: number }) {
  const [detail, setDetail] = useState<SampleDetail | null>(null);
  const [failed, setFailed] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let live = true;
    setFailed(false);
    const cached = CACHE.get(id);
    if (cached) { setDetail(cached); return; }
    api.getSample(id)
      .then((d) => { CACHE.set(id, d); if (live) setDetail(d); })
      .catch(() => { if (live) setFailed(true); });
    return () => { live = false; };
  }, [id]);

  // Only ever render the sample we were asked for. Sweeping the grid with the
  // arrow keys changes `id` faster than the fetches resolve, and showing the
  // previous image beside the new one's captions is the one failure this panel
  // must not have: it would invent a mismatch that isn't in the data.
  const s = detail && detail.id === id ? detail : null;

  if (failed && !s) {
    return (
      <div className="inspector">
        <p className="inspector-note">
          Couldn’t load sample {id}. The API may have stopped — the grid is
          showing what it already fetched.
        </p>
      </div>
    );
  }

  if (!s) {
    return (
      <div className="inspector">
        <div className="inspector-frame is-loading" />
        <p className="inspector-note">Loading sample {id}…</p>
      </div>
    );
  }

  const axes = s.axes;
  const recorded = recordedVerdict(s.tags);

  /** One call per sample: whatever was recorded before is removed, so the two
   *  writes are a replacement rather than an accumulation. Clicking the call
   *  that is already recorded retracts it — a reviewer who changes their mind
   *  must be able to leave the sample unjudged, not merely differently judged.
   *
   *  Not optimistic. A verdict is data a later export will be trusted to carry,
   *  so the button waits for the server rather than showing a call that may not
   *  have been saved. */
  async function record(value: string) {
    if (!s || saving) return;
    const tag = VERDICT_PREFIX + value;
    const next = recorded === tag ? null : tag;
    setSaving(true);
    try {
      if (recorded) await api.removeTag(s.id, recorded);
      if (next) await api.addTag(s.id, next);
      const kept = s.tags.filter((t) => !t.startsWith(VERDICT_PREFIX));
      const updated = { ...s, tags: next ? [...kept, next] : kept };
      CACHE.set(s.id, updated);
      setDetail(updated);
    } catch {
      // The cache still holds the pre-write tags, so the buttons snap back to
      // what the server actually has rather than to what was clicked.
      showToast(`Couldn’t record that call on #${s.id} — nothing was saved.`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="inspector">
      <div className="inspector-frame">
        <img src={s.image_url} alt={s.captions[0]?.text ?? s.filename}
             width={s.width ?? undefined} height={s.height ?? undefined} />
      </div>

      <div className="inspector-meta">
        <Link to={`/sample/${s.id}`} className="inspector-id">#{s.id}</Link>
        <span className="ev">{s.split}</span>
        {s.cluster != null && <span className="ev">cluster {s.cluster}</span>}
      </div>

      {/* The five captions are the reason this panel exists. Agreement is a
          SigLIP cosine between this image and this caption — a scorer's
          opinion, not ground truth — so it sits beside the text rather than
          re-ordering or hiding it. */}
      <ol className="inspector-captions">
        {s.captions.map((c, i) => (
          <li key={i}>
            <AgreementBadge value={c.agreement} />
            <span>{c.text}</span>
          </li>
        ))}
      </ol>

      {axes && (
        <dl className="inspector-axes">
          {AXES.filter((a) => axes[a] != null).map((a) => (
            <div key={a} title={AXIS_META[a].hint}>
              <dt>{AXIS_META[a].label}</dt>
              <dd>
                <span className="inspector-axis-track">
                  <span className="inspector-axis-bar"
                        style={{ width: `${(axes[a] as number) * 10}%` }} />
                </span>
                <b>{axes[a]}</b>
              </dd>
            </div>
          ))}
        </dl>
      )}

      {/* The judgement, recorded where it is made. Until now the review
          vocabulary existed only as prose in Quality's guide: you were told to
          type `verdict:caption-error` by hand, on another page, one sample at a
          time. These write the same tag, so a review session stays a filterable,
          exportable slice rather than a new kind of object. */}
      <div className="inspector-verdicts" role="group" aria-label="Record your call">
        {VERDICTS.map((v) => {
          const on = recorded === VERDICT_PREFIX + v.value;
          return (
            <button key={v.value} type="button" disabled={saving}
                    className={`verdict-btn${on ? " on" : ""}`}
                    aria-pressed={on}
                    title={on ? `${v.hint} — click again to retract` : v.hint}
                    onClick={() => record(v.value)}>
              {v.label}
            </button>
          );
        })}
      </div>
      {recorded && (
        <p className="inspector-note">
          Recorded as <span className="mono">{recorded}</span> —{" "}
          <Link to={`/?tag=${encodeURIComponent(recorded)}`}>
            see every sample you called this
          </Link>.
        </p>
      )}

      {s.caption_consistency != null && (
        <p className="inspector-note">
          The five captions agree with each other at{" "}
          <b>{s.caption_consistency.toFixed(3)}</b> — mean pairwise similarity,
          so a low number means the annotators saw different things here.
        </p>
      )}
    </div>
  );
}
