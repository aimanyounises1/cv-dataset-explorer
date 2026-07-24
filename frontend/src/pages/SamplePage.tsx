import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import type { SampleCard, SampleDetail } from "../api/types";
import ImageCard from "../components/ImageCard";
import TagEditor from "../components/TagEditor";

export default function SamplePage() {
  const { id } = useParams<{ id: string }>();
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

  if (error) return <div className="error">{error}</div>;
  if (!detail) return <div className="loading">Loading…</div>;

  return (
    <div>
      <Link className="back-link" to="/">← Back to gallery</Link>
      <div className="detail" style={{ marginTop: 12 }}>
        <div>
          <img className="detail-image" src={detail.image_url} alt={detail.filename} />
        </div>
        <div>
          <div className="panel">
            <h3>Captions ({detail.captions.length})</h3>
            <ol className="caption-list">
              {detail.captions.map((c, i) => <li key={i}>{c}</li>)}
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
              {detail.cluster != null && (<><dt>Cluster</dt><dd>#{detail.cluster}</dd></>)}
            </dl>
          </div>
          {detail.vlm_tags.length > 0 && (
            <div className="panel">
              <h3>VLM tags</h3>
              <div className="tag-row">
                {detail.vlm_tags.map((t) => <span className="tag vlm" key={t}>{t}</span>)}
              </div>
            </div>
          )}
          <div className="panel">
            <h3>My tags</h3>
            <TagEditor sampleId={detail.id} tags={detail.tags} onChanged={refresh} />
          </div>
        </div>
      </div>

      <div className="section-title">Similar images</div>
      {similarError && <div className="notice">{similarError}</div>}
      <div className="grid">
        {similar.map((s) => <ImageCard key={s.id} sample={s} />)}
      </div>
    </div>
  );
}
