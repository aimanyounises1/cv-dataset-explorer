import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import type { SampleCard, SampleDetail } from "../api/types";
import AxisBreakdown from "../components/AxisBreakdown";
import AxisLegend from "../components/AxisLegend";
import ImageCard from "../components/ImageCard";
import ProvenanceBanner from "../components/ProvenanceBanner";
import TagEditor from "../components/TagEditor";
import { useNeighbours } from "../hooks/useResultOrder";

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
  const [detail, setDetail] = useState<SampleDetail | null>(null);
  const [similar, setSimilar] = useState<SampleCard[]>([]);
  const [similarError, setSimilarError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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
      if ((e.key === "ArrowLeft" || e.key === "k") && neighbours.prev != null) {
        navigate(`/samples/${neighbours.prev}`);
      } else if ((e.key === "ArrowRight" || e.key === "j") && neighbours.next != null) {
        navigate(`/samples/${neighbours.next}`);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [neighbours, navigate]);

  if (error) return <div className="error">{error}</div>;
  if (!detail) return <div className="loading">Loading…</div>;

  const back = () =>
    window.history.length > 1 ? navigate(-1) : navigate("/");

  return (
    <div>
      <ProvenanceBanner />
      <div className="detail-nav">
        <button className="ghost back-btn" onClick={back}>← Back</button>
        {neighbours.position != null && (
          <div className="detail-stepper">
            <button className="ghost" disabled={neighbours.prev == null}
                    onClick={() => neighbours.prev != null && navigate(`/samples/${neighbours.prev}`)}
                    title="Previous result (← or k)">← Prev</button>
            <span className="meta-line" style={{ margin: 0 }}>
              {neighbours.position} / {neighbours.total}
            </span>
            <button className="ghost" disabled={neighbours.next == null}
                    onClick={() => neighbours.next != null && navigate(`/samples/${neighbours.next}`)}
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
          {similar.length > 0 && (
            <Link className="open-all" to={`/?ids=${similar.map((s) => s.id).join(",")}`}
                  title="Open exactly these neighbours in the gallery, where they can be filtered, sorted and exported">
              Open all {similar.length} in gallery →
            </Link>
          )}
        </span>
        {similar.some((s) => s.axes) && <AxisLegend />}
      </div>
      {similarError && <div className="notice">{similarError}</div>}
      <div className="grid">
        {similar.map((s) => <ImageCard key={s.id} sample={s} scoreBasis="cosine" />)}
      </div>
    </div>
  );
}
