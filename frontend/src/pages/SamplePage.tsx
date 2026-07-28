import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { SampleCard, SampleDetail } from "../api/types";
import AxisBreakdown from "../components/AxisBreakdown";
import AxisLegend from "../components/AxisLegend";
import ImageCard from "../components/ImageCard";
import ProvenanceBanner from "../components/ProvenanceBanner";
import TagEditor from "../components/TagEditor";
import { useNeighbours } from "../hooks/useResultOrder";

/** The 10th percentile of nearest-neighbour similarity across this corpus: for
 * every image, the cosine to its closest other image, self excluded. The
 * similar grid always returns k cards, so k alone cannot tell a real class
 * from padding; this is the line between "similar images" and "the least
 * dissimilar of 8,000". Provider-aware: an index whose manifest publishes its
 * own measured floor (stats/overview `sim_floor` — e.g. 0.5069 for the Qwen
 * space) uses it; the constant below is the SigLIP-measured fallback for the
 * legacy layout. Overridable per-URL with ?min_sim= (0 disables). */
const DEFAULT_SIM_FLOOR = 0.76;

/* One fetch per page load lifetime — the floor is corpus-wide, not per-sample. */
let floorPromise: Promise<number> | null = null;
const activeSimFloor = () => {
  floorPromise ??= api.overview()
    .then((o) => (typeof o.sim_floor === "number" ? o.sim_floor : DEFAULT_SIM_FLOOR))
    .catch(() => DEFAULT_SIM_FLOOR);
  return floorPromise;
};

function AgreementBadge({ value }: { value?: number | null }) {
  if (value == null) return null;
  const cls = value < 0.05 ? "warn" : value < 0.09 ? "mid" : "ok";
  return (
    <span className={`agree-badge ${cls}`}
          title="SigLIP image-caption agreement (low = suspect caption)">
      {value.toFixed(3)}
    </span>
  );
}

export default function SamplePage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [detail, setDetail] = useState<SampleDetail | null>(null);
  const [similar, setSimilar] = useState<SampleCard[]>([]);
  const [similarError, setSimilarError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // The active index's own measured floor, fetched once; the SigLIP constant
  // covers the legacy layout and the fetch-failed path. Declared with the
  // other hooks — above every early return — because hook order is identity.
  const [measuredFloor, setMeasuredFloor] = useState(DEFAULT_SIM_FLOOR);
  useEffect(() => { activeSimFloor().then(setMeasuredFloor); }, []);
  const [gapBusy, setGapBusy] = useState(false);
  const [gapError, setGapError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    if (!id) return;
    api.getSample(id).then(setDetail).catch((e) => setError(String(e)));
  }, [id]);

  useEffect(() => {
    setDetail(null);
    setSimilar([]);
    setSimilarError(null);
    refresh();
    if (id) {
      api.similar(id)
        .then(setSimilar)
        .catch(() => setSimilarError("Similarity unavailable — embeddings not computed yet."));
    }
    window.scrollTo(0, 0);
  }, [id, refresh]);

  // Sequential triage: step through the result list without returning to it.
  const neighbours = useNeighbours(id);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      // Stepping REPLACES the history entry: twenty j-presses used to mean
      // twenty Back-presses to reach the grid again. Arriving from the grid
      // was the one push; the walk itself is one place, revisited.
      if ((e.key === "ArrowLeft" || e.key === "k") && neighbours.prev != null) {
        navigate(`/samples/${neighbours.prev}`, { replace: true });
      } else if ((e.key === "ArrowRight" || e.key === "j") && neighbours.next != null) {
        navigate(`/samples/${neighbours.next}`, { replace: true });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [neighbours, navigate]);

  if (error) {
    // A raw `Error: 404: {"detail":...}` on an otherwise empty pane reads as
    // a crash. The missing-sample case gets a human sentence and a way back;
    // genuinely unexpected errors keep the raw string, which is then a clue.
    const missing = error.includes("404");
    return (
      <div>
        <button className="ghost back-btn"
                onClick={() => window.history.length > 1 ? navigate(-1) : navigate("/")}>
          ← Back
        </button>
        <div className="error">
          {missing
            ? `Sample ${id} does not exist — the id may be mistyped, or the link
               may predate a re-ingest of the dataset.`
            : error}
        </div>
      </div>
    );
  }
  if (!detail) return <div className="loading">Loading…</div>;

  const back = () =>
    window.history.length > 1 ? navigate(-1) : navigate("/");

  // The floor arrives from the URL so a pasted link carries the same cutoff its
  // author saw; anything unparseable falls back to the measured default rather
  // than to "no floor", which would silently upgrade padding to neighbours.
  const rawFloor = searchParams.get("min_sim");
  const parsedFloor = rawFloor === null ? NaN : Number(rawFloor);
  const floor = Number.isFinite(parsedFloor) && parsedFloor >= 0 && parsedFloor <= 1
    ? parsedFloor : measuredFloor;
  const above = similar.filter((s) => s.score == null || s.score >= floor);
  const below = similar.filter((s) => s.score != null && s.score < floor);

  const tagCoverageGap = () => {
    setGapBusy(true);
    setGapError(null);
    api.addTag(detail.id, "coverage-gap")
      .then(refresh)
      .catch((e) => setGapError(e instanceof Error ? e.message : "Failed to tag"))
      .finally(() => setGapBusy(false));
  };

  return (
    <div>
      {/* Screen-reader heading: the visual layout leads with the image, and a
          visible title would duplicate the stepper's context line. */}
      <h1 className="sr-only">Sample {detail.id} — {detail.filename}</h1>
      <ProvenanceBanner />
      <div className="detail-nav">
        <button className="ghost back-btn" onClick={back}>← Back</button>
        {neighbours.position != null && (
          <div className="detail-stepper">
            <button className="ghost" disabled={neighbours.prev == null}
                    onClick={() => neighbours.prev != null
                      && navigate(`/samples/${neighbours.prev}`, { replace: true })}
                    title="Previous result (← or k)">← Prev</button>
            <span className="meta-line" style={{ margin: 0 }}>
              {neighbours.position} / {neighbours.total}
            </span>
            <button className="ghost" disabled={neighbours.next == null}
                    onClick={() => neighbours.next != null
                      && navigate(`/samples/${neighbours.next}`, { replace: true })}
                    title="Next result (→ or j)">Next →</button>
          </div>
        )}
      </div>
      <div className="detail" style={{ marginTop: 12 }}>
        <div>
          <img className="detail-image" src={detail.image_url} alt={detail.captions[0]?.text ?? detail.filename} />
        </div>
        <div>
          <div className="panel">
            <h3>
              Captions ({detail.captions.length})
              {detail.caption_consistency != null && (
                <span className="pill" style={{ marginLeft: 8 }}
                      title="Mean pairwise caption similarity (low = captions disagree)">
                  consistency {detail.caption_consistency.toFixed(3)}
                </span>
              )}
            </h3>
            <ol className="caption-list">
              {detail.captions.map((c, i) => (
                <li key={i}>{c.text} <AgreementBadge value={c.agreement} /></li>
              ))}
            </ol>
          </div>
          <div className="panel">
            <h3>Metadata</h3>
            <dl className="kv">
              <dt>Filename</dt><dd>{detail.filename}</dd>
              <dt>Split</dt><dd>{detail.split}</dd>
              <dt>Dimensions</dt>
              <dd>{detail.width ?? "?"} × {detail.height ?? "?"} px</dd>
              <dt>File size</dt>
              <dd>{detail.filesize ? `${(detail.filesize / 1024).toFixed(0)} KB` : "?"}</dd>
              {/* Every identifier on this page that names a *set* is a link to
                  that set. A cluster number the reader cannot open is trivia;
                  one click away it is the answer to "is this whole group like
                  that, or just this frame?". */}
              {detail.cluster != null && (
                <><dt>Cluster</dt>
                <dd>
                  <Link className="attr-link" to={`/?cluster=${detail.cluster}`}
                        title="Open every image in this embedding cluster">
                    #{detail.cluster}
                  </Link>
                </dd></>
              )}
              {Object.entries(detail.attributes).map(([grp, label]) => (
                <span key={grp} style={{ display: "contents" }}>
                  <dt>{grp.replace(/_/g, " ")}</dt>
                  <dd>
                    <Link className="attr-link" to={`/?attr=${encodeURIComponent(`${grp}:${label}`)}`}>
                      {label}
                    </Link>
                  </dd>
                </span>
              ))}
            </dl>
          </div>
          <div className="panel">
            <h3>Difficulty</h3>
            <AxisBreakdown axes={detail.axes} />
          </div>
          {detail.vlm_tags.length > 0 && (
            <div className="panel">
              <h3>VLM tags</h3>
              <div className="tag-row">
                {detail.vlm_tags.map((t) => (
                  <Link className="tag vlm" key={t} to={`/?vlm_tag=${encodeURIComponent(t)}`}
                        title={`Open every image tagged “${t}”`}>
                    {t}
                  </Link>
                ))}
              </div>
            </div>
          )}
          <div className="panel">
            <h3>My tags</h3>
            <TagEditor sampleId={detail.id} tags={detail.tags} onChanged={refresh} />
          </div>
        </div>
      </div>

      {/* The same four bars appear on these cards, so the same key belongs
          here. Without it the encoding is unexplained everywhere except the
          gallery — and a user can reach this page directly from a link. */}
      <div className="section-title similar-head">
        <span className="similar-title">
          Similar images
          {/* This grid is the only place the tool turns one frame into a class,
              and until now the class died here: you could look at the twelve
              neighbours but not filter, sort or export them. The gallery already
              accepts an explicit id list, so hand it over — same construction the
              map's lasso uses. */}
          {/* The hand-off respects the floor it sits under: the primary link
              carries only the ids the page itself called a class — promoting
              the greyed context into a slice would feed padding straight into
              any export made there. The full set stays one click away,
              labelled as what it is. */}
          {similar.length > 0 && above.length > 0 && (
            <Link className="open-all" to={`/?ids=${above.map((s) => s.id).join(",")}`}
                  title="Open the neighbours above the similarity floor in the gallery — the class, without the greyed context">
              Open {above.length} above floor in gallery →
            </Link>
          )}
          {similar.length > 0 && above.length > 0 && below.length > 0 && (
            <Link className="open-all open-all-dim"
                  to={`/?ids=${similar.map((s) => s.id).join(",")}`}
                  title="Include the below-floor context too — remember it travels into any export made there">
              all {similar.length}
            </Link>
          )}
          {similar.length > 0 && above.length === 0 && (
            <Link className="open-all" to={`/?ids=${similar.map((s) => s.id).join(",")}`}
                  title="Every neighbour here is below the similarity floor — this opens context, not a class">
              Open all {similar.length} (below floor) in gallery →
            </Link>
          )}
        </span>
        {similar.some((s) => s.axes) && <AxisLegend />}
      </div>
      {similarError && <div className="notice">{similarError}</div>}
      {/* A singleton is a finding, not an empty state: when nothing clears the
          floor, say so and make the gap tangible — the tag turns "I noticed a
          hole" into a filterable, exportable slice via the existing tag API. */}
      {similar.length > 0 && above.length === 0 && (
        <div className="notice">
          No related cases above the similarity floor. This looks like a
          singleton — possible coverage gap.{" "}
          {detail.tags.includes("coverage-gap") ? (
            <strong>Tagged coverage-gap.</strong>
          ) : (
            <button className="ghost" onClick={tagCoverageGap} disabled={gapBusy}
                    title="Tag this sample 'coverage-gap' so the gap becomes a filterable, exportable slice">
              Tag as coverage-gap
            </button>
          )}
          {gapError && <span style={{ marginLeft: 8, fontSize: 12 }}>{gapError}</span>}
        </div>
      )}
      {above.length > 0 && (
        <div className="grid">
          {above.map((s) => <ImageCard key={s.id} sample={s} scoreBasis="cosine" />)}
        </div>
      )}
      {below.length > 0 && (
        <>
          <div className="sim-divider"
               title={"The floor is the 10th percentile of nearest-neighbour similarity "
                    + "across this corpus, measured on the active retrieval index "
                    + "(each provider's space has its own). "
                    + "Override with ?min_sim= in the URL (0 disables)."}>
            below similarity floor {Number(floor.toFixed(3))} — shown for context
          </div>
          <div className="grid dim">
            {below.map((s) => <ImageCard key={s.id} sample={s} scoreBasis="cosine" />)}
          </div>
        </>
      )}
    </div>
  );
}
