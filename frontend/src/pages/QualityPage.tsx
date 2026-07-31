import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { QASelection, QASummary, SuspectCaption } from "../api/types";
import ErrorState from "../components/ErrorState";
import { Segmented } from "../components/controls";
import { SURFACE, sequential } from "../lib/viz";
import { VERDICTS, VERDICT_PREFIX } from "../lib/verdicts";

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

/** A bin with no captions in it still gets a hairline, so the column stays
 * hoverable (its title carries the count) and a hole in the distribution reads
 * as a measured zero rather than as something the chart forgot to draw. */
const EMPTY_BIN_PCT = 2;

/** Bar height is log(1 + count), not count.
 *
 * Measured on this corpus: the mode bin holds 4,057 captions while the region
 * the review line actually cuts holds 2 to 200 — a 2,000:1 range. On a linear
 * height every bin below ~0.09 agreement collapsed onto the same 2% floor, so
 * dragging the line from 0.021 to 0.096 changed nothing on screen and the panel
 * could not answer the one question it exists for: is this cutoff in the tail,
 * or already inside the bulk? A log height resolves that range and keeps a bin
 * of one distinguishable from a bin of none. Because areas under it are no
 * longer proportional to counts, the panel head says so in words. */
function barHeightPct(count: number, peak: number): number {
  if (count <= 0) return EMPTY_BIN_PCT;
  return (Math.log1p(count) / Math.log1p(peak)) * 100;
}

/** One suspect per line: thumb, caption, figures. A reviewer triages this list
 * by the hundred, so the unit of reading is the row, not a card — the caption
 * ellipsizes rather than wraps and the full text rides on the title. */
function SuspectRow({ item, scoreLabel }: { item: SuspectCaption; scoreLabel: string }) {
  return (
    <Link className="suspect-row" to={`/samples/${item.sample.id}`}
          title={`“${item.caption}” — ${scoreLabel} ${item.agreement.toFixed(3)}`}>
      <img src={item.sample.thumb_url} alt="" loading="lazy" />
      <span className="suspect-caption">“{item.caption}”</span>
      {/* The column labels live in the (aria-hidden) header, which a screen
          reader never reaches — each figure carries its own, off-screen, so the
          row is read as named values and not as four bare numbers. */}
      <span className="suspect-score" title={`${scoreLabel} score`}>
        <span className="sr-only">{scoreLabel}&nbsp;</span>{item.agreement.toFixed(3)}
      </span>
      <span className="suspect-sib" title="Mean agreement of the sample's other captions">
        <span className="sr-only">siblings&nbsp;</span>
        {item.sibling_mean != null ? item.sibling_mean.toFixed(3) : "—"}
      </span>
      <span className="suspect-split">
        <span className="sr-only">split&nbsp;</span>{item.sample.split}
      </span>
    </Link>
  );
}

/** Column labels once, over rows that are all figures — the per-row pills that
 * used to carry them repeated the same three words a hundred times. */
function SuspectHead({ scoreLabel }: { scoreLabel: string }) {
  return (
    <div className="suspect-head" aria-hidden="true">
      <span className="suspect-caption">caption</span>
      <span className="suspect-score">{scoreLabel}</span>
      <span className="suspect-sib">siblings</span>
      <span className="suspect-split">split</span>
    </div>
  );
}

/** The lists load in pages: rows 26-100 are one click away, not 3,000px of
 * scroll between the reviewer and the consistency section below. */
const LIST_PAGE = 25;

/** The two rankings this page offers. Both are "worst first" over the same
 * corpus, but they answer different questions — one caption against its image,
 * one sample's captions against each other — so neither is a mode of the
 * other and the switch is a peer choice, not a setting. */
type ListKind = "suspect" | "consistency";

/** Annotation QA: CLIPScore-style ranking of captions least supported by
 * their image — the mislabel-hunting workflow. */
export default function QualityPage() {
  const [summary, setSummary] = useState<QASummary | null>(null);
  const [suspects, setSuspects] = useState<SuspectCaption[]>([]);
  const [inconsistent, setInconsistent] = useState<SuspectCaption[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sectionErrors, setSectionErrors] = useState<string[]>([]);
  const [threshold, setThreshold] = useState<number | null>(null);
  /* The brush is a selection, and a selection this app promises to keep in
   * the URL: open a suspect, press Back, and the cutoff you set must still be
   * there — as must a pasted /quality?max_agreement=… link. The computed
   * default stays OUT of the URL: only a value the user chose is user state. */
  const [searchParams, setSearchParams] = useSearchParams();
  const carriedThreshold = searchParams.get("max_agreement");
  const split = searchParams.get("split") ?? "";
  const touched = useRef(false);
  useEffect(() => {
    if (threshold == null || !touched.current) return;
    if (carriedThreshold === String(threshold)) return;
    const t = setTimeout(() => {
      setSearchParams((p) => {
        const next = new URLSearchParams(p);
        next.set("max_agreement", String(threshold));
        return next;
      }, { replace: true });
    }, 300);
    return () => clearTimeout(t);
  }, [threshold, carriedThreshold, setSearchParams]);
  const [listLoading, setListLoading] = useState(false);
  const [selection, setSelection] = useState<QASelection | null>(null);
  const [shownSuspects, setShownSuspects] = useState(LIST_PAGE);
  const [shownInconsistent, setShownInconsistent] = useState(LIST_PAGE);
  const [list, setList] = useState<ListKind>("suspect");

  useEffect(() => {
    api.qaSummary()
      .then(setSummary)
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!summary?.available) return;
    if (carriedThreshold !== null) {
      const carried = Number(carriedThreshold);
      if (Number.isFinite(carried)) {
        touched.current = true;
        setThreshold(carried);
      }
      return;
    }
    touched.current = false;
    setThreshold(defaultThreshold(summary));
  }, [carriedThreshold, summary]);

  useEffect(() => {
    // On a QA page, a swallowed error rendering as "no problems found" is the
    // worst failure mode — failed sections must announce themselves.
    const fail = (what: string) => () =>
      setSectionErrors((prev) => [...prev, what]);
    api.inconsistentSamples(split).then(setInconsistent).catch(fail("consistency ranking"));
  }, [split]);

  // The list and the selection size both follow the threshold. Debounced
  // because the control is a slider and every intermediate value would
  // otherwise be a request.
  useEffect(() => {
    if (threshold == null) return;
    setListLoading(true);
    // Paging describes one cut of the corpus: when the review line moves, the
    // "show 25 more" the reviewer clicked no longer refers to this list. (The
    // consistency list below is ranked independently of the threshold, so its
    // own paging is not this effect's to reset.)
    setShownSuspects(LIST_PAGE);
    const ctrl = new AbortController();
    const t = setTimeout(() => {
      api.suspectCaptions({ limit: 100, max_agreement: threshold,
                            split: split || undefined })
        .then(setSuspects)
        .catch(() => setSectionErrors((p) =>
          p.includes("suspect captions") ? p : [...p, "suspect captions"]))
        .finally(() => setListLoading(false));
      // Counted in SQL, not from the histogram: bins that straddle the line can
      // only give a rounded answer, and this number labels a button that hands
      // the set to another view — so it has to be the number that view shows.
      api.qaSelection(threshold, ctrl.signal, split)
        .then(setSelection)
        .catch(() => { /* the readout falls back to the binned estimate */ });
    }, 220);
    return () => { clearTimeout(t); ctrl.abort(); };
  }, [threshold, split]);

  const hist = summary?.histogram ?? [];
  const peak = useMemo(() => Math.max(1, ...hist.map((b) => b.count)), [hist]);
  // Binned estimate, used only until the exact count arrives (and if it fails).
  const binnedBelow = useMemo(
    () => (threshold == null ? 0
      : hist.filter((b) => b.hi <= threshold).reduce((n, b) => n + b.count, 0)),
    [hist, threshold]);
  /** Which bar the review line falls in front of.
   *
   * The bars are flex children with a gap between them, so an offset expressed
   * as a percentage of the container drifts by the accumulated gap — 78px of
   * the 1,114px chart at 40 bins, nearly two bins wide at the right-hand end.
   * The mark is therefore spliced into the same flex row as a zero-width item,
   * which lands it exactly on the boundary the opacity break already draws: a
   * bin counts as selected when `b.hi <= threshold`, so the cut belongs in
   * front of the first bin the threshold does not fully cover. */
  const cutAt = useMemo(() => {
    if (threshold == null) return 0;
    const i = hist.findIndex((b) => b.hi > threshold);
    return i < 0 ? hist.length : i;
  }, [hist, threshold]);
  const exact = selection != null && selection.max_agreement === threshold;
  const belowCount = exact ? selection!.captions : binnedBelow;
  const totalScored = summary?.scored_captions ?? 0;

  if (error) return <ErrorState error={error} />;
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
      {/* The page opens on the review line and the captions under it. The three
          summary cards that used to sit here — captions scored, mean agreement,
          and the count below the line — said exactly what the panel immediately
          below says in its own header and readout, and cost ~105px at 1440 and
          ~240px at 390 to say it twice. Restating a number is not the same as
          leading with it. */}
      <h1 className="section-title" style={{ marginTop: 0 }}>Caption quality</h1>

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
              {/* The height transform is named where the chart is named. It
                  belongs here rather than on .dist-axis, which labels the
                  agreement scale — the log is on the other axis — and which is
                  a single non-wrapping line already 330px wide at 390. */}
              <div className="meta-line" style={{ marginBottom: 0, marginTop: 4 }}>
                {totalScored.toLocaleString()} scored captions · mean{" "}
                {summary.mean_agreement?.toFixed(3)} · bar height on a log(1+count) scale
              </div>
            </div>
            <div className="dist-readout">
              ≤ {threshold.toFixed(3)} · {belowCount.toLocaleString()} captions (
              {((belowCount / Math.max(1, totalScored)) * 100).toFixed(1)}%)
            </div>
          </div>

          {/* The cut, drawn. Until now the only trace of the review line in the
              chart was the `.dist-bar.out` opacity step, which in the tail
              falls between two bars of equal height and reads as nothing at
              all. The line a reviewer is asked to place is now a mark they can
              see, carrying the value it stands for; the count it selects is on
              the readout one line above, so it is not repeated here. */}
          {(() => {
            const cutMark = (
              <div
                key="cut"
                className="dist-cut"
                style={{
                  flex: "0 0 auto", width: 0, alignSelf: "stretch",
                  position: "relative", pointerEvents: "none",
                  borderLeft: "2px solid var(--accent)",
                }}
              >
                {/* Anchored away from the nearer edge so the label always
                    opens into the chart and never out of the panel. */}
                <span
                  className="dist-readout"
                  style={{
                    position: "absolute", top: 0,
                    ...(cutAt * 2 <= hist.length ? { left: 0 } : { right: 0 }),
                    background: "var(--bg-raised)",
                    borderRadius: "var(--radius-sm)",
                    padding: "0 4px",
                  }}
                >
                  ≤ {threshold.toFixed(3)}
                </span>
              </div>
            );
            return (
              <div className="dist-bars" role="img"
                   aria-label={`Agreement histogram, bar height on a log of one plus bin count scale; `
                               + `threshold at ${threshold.toFixed(3)} selects `
                               + `${belowCount} of ${totalScored} captions`}>
                {hist.flatMap((b, i) => {
                  const inRange = b.hi <= threshold;
                  const bar = (
                    <div
                      key={b.lo}
                      className={`dist-bar ${inRange ? "" : "out"}`}
                      style={{
                        height: `${barHeightPct(b.count, peak)}%`,
                        // Colour by position on the scale, not by selection state —
                        // opacity carries selection, so the two do not compete.
                        background: sequential(
                          (b.lo - (summary.min_agreement ?? 0)) /
                          Math.max(1e-6, (summary.max_agreement ?? 1) - (summary.min_agreement ?? 0))),
                      }}
                      title={`${b.lo.toFixed(3)}–${b.hi.toFixed(3)}: ${b.count.toLocaleString()} captions`}
                    />
                  );
                  return i === cutAt ? [cutMark, bar] : [bar];
                })}
                {cutAt >= hist.length && cutMark}
              </div>
            );
          })()}
          <div className="dist-axis">
            <span>{summary.min_agreement?.toFixed(3)}</span>
            <span>worse ← agreement → better</span>
            <span>{summary.max_agreement?.toFixed(3)}</span>
          </div>

          {/* The brush now spans exactly the chart it brushes. As a flex item
              between its label and its readout the track measured 885px against
              the bars' 1,114px at 1440, and 129px against 330px at 390 — so the
              thumb and the cut it was setting could sit ~120px apart, and on a
              phone the right two thirds of the histogram had no track under it
              at all. Wrapping the track onto its own row makes it a full-width
              child of the same panel as `.dist-bars`, which is the whole of the
              alignment; what remains is the native thumb's own half-width,
              which no layout can remove without replacing the control. */}
          <div className="dist-control"
               style={{ flexWrap: "wrap", justifyContent: "space-between" }}>
            <label className="eyebrow" htmlFor="qa-threshold">Review below</label>
            <span className="dist-readout" style={{ color: SURFACE.textDim }}>
              {listLoading ? "updating…"
                : `${Math.min(shownSuspects, suspects.length)} of ${suspects.length} listed`}
            </span>
            <input
              id="qa-threshold"
              type="range"
              style={{ flexBasis: "100%" }}
              min={summary.min_agreement ?? 0}
              max={summary.max_agreement ?? 1}
              step={0.001}
              value={threshold}
              onChange={(e) => { touched.current = true; setThreshold(Number(e.target.value)); }}
            />
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
                 href={`/api/export?${split ? `split=${encodeURIComponent(split)}&` : ""}`
                       + `max_agreement=${threshold}&format=${f}`}
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

      {/* Two rankings of the same corpus, and a reviewer reads one at a time.
          Stacked, the second one began ~1,900px below the fold: to learn that
          the consistency ranking existed you had to scroll past every row of
          the first. They are peers, so they are tabs — and the list scrolls
          inside its own frame, which keeps the distribution, the review line
          and the export controls on screen while the rows move. */}
      <div className="list-switch">
        <div className="section-title tight" style={{ margin: 0 }}>
          {list === "suspect" ? "Most suspect captions" : "Least consistent samples"}
        </div>
        {inconsistent.length > 0 && (
          <Segmented
            label="Which ranking to review"
            value={list}
            onChange={(v) => setList(v as ListKind)}
            options={[
              { value: "suspect", label: `Suspect captions${suspects.length ? ` (${suspects.length})` : ""}` },
              { value: "consistency", label: `Least consistent (${inconsistent.length})` },
            ]}
          />
        )}
      </div>

      {list === "suspect" ? (
        <>
          {/* How to read a row and how to record a call: three lines at 1440 and
              six at 390, between the reviewer and the rows they came for. Folded,
              not cut — the verdict:* convention is the review workflow and has to
              stay one click from the list it governs. */}
          <details className="caveat review-guide">
            <summary>How to read a row, and how to record your call</summary>
            <p>
              Lowest image-caption agreement first. Low score + high sibling mean ⇒
              the caption is likely wrong; all-low ⇒ the image itself is unusual.
              Record your call with the buttons in the gallery’s inspector, or as a
              tag on the sample page —{" "}
              {/* Named from the same constant the inspector's buttons are built
                  from, so the guide cannot come to describe a vocabulary the
                  controls no longer offer. */}
              <span className="mono">
                {VERDICTS.map((v) => VERDICT_PREFIX + v.value).join(" · ")}
              </span>{" "}
              — and the review session becomes a filterable, exportable slice.
            </p>
          </details>
          {suspects.length === 0 ? (
            <div className="empty">No scored captions to show.</div>
          ) : (
            <div className="suspect-frame">
              <div className="suspect-list">
                <SuspectHead scoreLabel="agreement" />
                {suspects.slice(0, shownSuspects).map((s, i) => (
                  <SuspectRow key={`${s.sample.id}-${i}`} item={s} scoreLabel="agreement" />
                ))}
              </div>
              {suspects.length > shownSuspects && (
                <button className="ghost suspect-more"
                        onClick={() => setShownSuspects(shownSuspects + LIST_PAGE)}>
                  Show {Math.min(LIST_PAGE, suspects.length - shownSuspects)} more
                  ({suspects.length - shownSuspects} loaded beyond this point)
                </button>
              )}
            </div>
          )}
        </>
      ) : (
        <>
          <p className="meta-line tight">
            Samples whose 5 captions disagree most with each other — ambiguous
            images or outlier annotations. The score is the sample's caption
            consistency, lowest first.
          </p>
          <div className="suspect-frame">
            <div className="suspect-list">
              <SuspectHead scoreLabel="consistency" />
              {inconsistent.slice(0, shownInconsistent).map((s, i) => (
                <SuspectRow key={`${s.sample.id}-${i}`} item={s} scoreLabel="consistency" />
              ))}
            </div>
            {inconsistent.length > shownInconsistent && (
              <button className="ghost suspect-more"
                      onClick={() => setShownInconsistent(shownInconsistent + LIST_PAGE)}>
                Show {Math.min(LIST_PAGE, inconsistent.length - shownInconsistent)} more
                ({inconsistent.length - shownInconsistent} loaded beyond this point)
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
