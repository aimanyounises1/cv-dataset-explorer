import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api/client";
import type { AttributeGroup, CaptionStats, DuplicatePair, StatsOverview } from "../api/types";
import { AXIS_STROKE, GRID_STROKE, SERIES, TOOLTIP_STYLE } from "../components/chartTheme";

export default function StatsPage() {
  const [overview, setOverview] = useState<StatsOverview | null>(null);
  const [captions, setCaptions] = useState<CaptionStats | null>(null);
  const [dups, setDups] = useState<DuplicatePair[]>([]);
  const [coverage, setCoverage] = useState<AttributeGroup[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sectionErrors, setSectionErrors] = useState<string[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    // A failed section must say so — an error rendering as an empty chart
    // would be indistinguishable from "nothing to report".
    const fail = (what: string) => () =>
      setSectionErrors((prev) => [...prev, what]);
    api.overview().then(setOverview).catch((e) => setError(String(e)));
    api.captionStats().then(setCaptions).catch(fail("caption statistics"));
    api.duplicates().then(setDups).catch(fail("near-duplicates"));
    api.coverage().then(setCoverage).catch(fail("attribute coverage"));
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!overview) return <div className="loading">Loading statistics…</div>;

  const splitData = Object.entries(overview.splits).map(([name, count]) => ({ name, count }));
  const sizeData = Object.entries(overview.image_size_buckets).map(([name, count]) => ({ name, count }));

  return (
    <div>
      <div className="stat-cards">
        <div className="stat-card">
          <div className="value">{overview.total_samples.toLocaleString()}</div>
          <div className="label">Images</div>
        </div>
        <div className="stat-card">
          <div className="value">{overview.total_captions.toLocaleString()}</div>
          <div className="label">Captions</div>
        </div>
        <div className="stat-card">
          <div className="value">{overview.avg_caption_length_words}</div>
          <div className="label">Avg caption length (words)</div>
        </div>
        <div className="stat-card">
          <div className={`value ${overview.embeddings_available ? "ok" : "warn"}`}>
            {overview.embeddings_available ? "Ready" : "Off"}
          </div>
          <div className="label">Semantic search (SigLIP)</div>
        </div>
        <div className="stat-card">
          <div className={`value ${overview.vlm_enriched > 0 ? "ok" : "warn"}`}>
            {overview.vlm_enriched > 0 ? overview.vlm_enriched.toLocaleString() : "Off"}
          </div>
          <div className="label">VLM-enriched samples</div>
        </div>
      </div>

      {sectionErrors.length > 0 && (
        <div className="error">
          Failed to load: {sectionErrors.join(", ")}. The sections below may be incomplete.
        </div>
      )}

      {/* Where the data came from and what is known to be off about it —
          a dataset tool that hides its own provenance is asking to be trusted
          on faith. */}
      <details className="caveat">
        <summary>Dataset provenance and known limitations</summary>
        <ul>
          <li>
            <strong>Source.</strong> The <code>jxie/flickr8k</code> copy on Hugging Face,
            ingested locally. Its dataset card carries no construction methodology and
            specifies no license.
          </li>
          <li>
            <strong>Row count.</strong> This copy contains exactly{" "}
            {overview.total_samples.toLocaleString()} images (6,000 / 1,000 / 1,000 across
            train / validation / test), while the original Flickr8k distribution has about
            8,091. Roughly 90 images are absent, with no explanation given upstream.
          </li>
          <li>
            <strong>Splits.</strong> The counts match the canonical Hodosh split, but the
            per-image assignments are undocumented in this copy and have not been verified
            against the original split files.
          </li>
          <li>
            <strong>Licensing.</strong> Upstream Flickr8k is distributed for
            non-commercial research and education only. This copy states no license of its
            own, so the upstream terms are the safe assumption. Images are not
            redistributed by this repository — they are downloaded to your machine.
          </li>
          <li>
            <strong>Composition.</strong> Captions were written by US-based crowdworkers
            and the images were drawn from a handful of Flickr hobby groups, so the corpus
            is not a neutral sample of the visual world — expect people, dogs, and outdoor
            action to dominate.
          </li>
        </ul>
      </details>

      <div className="charts">
        <div className="panel">
          <h3>Samples per split</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={splitData}>
              <CartesianGrid stroke={GRID_STROKE} vertical={false} />
              <XAxis dataKey="name" stroke={AXIS_STROKE} />
              <YAxis stroke={AXIS_STROKE} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Bar dataKey="count" fill={SERIES.blue} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="panel">
          <h3>Image size (longest side)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={sizeData}>
              <CartesianGrid stroke={GRID_STROKE} vertical={false} />
              <XAxis dataKey="name" stroke={AXIS_STROKE} />
              <YAxis stroke={AXIS_STROKE} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Bar dataKey="count" fill={SERIES.green} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        {captions && (
          <>
            <div className="panel">
              <h3>Caption length distribution (words)</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={captions.length_histogram}>
                  <CartesianGrid stroke={GRID_STROKE} vertical={false} />
                  <XAxis dataKey="bucket" stroke={AXIS_STROKE} />
                  <YAxis stroke={AXIS_STROKE} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Bar dataKey="count" fill={SERIES.purple} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="panel">
              <h3>Most frequent caption words</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={captions.top_words.slice(0, 15)}>
                  <CartesianGrid stroke={GRID_STROKE} vertical={false} />
                  <XAxis dataKey="word" stroke={AXIS_STROKE} interval={0} angle={-35} textAnchor="end" height={60} />
                  <YAxis stroke={AXIS_STROKE} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Bar dataKey="count" fill={SERIES.amber} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </div>

      {coverage.length > 0 && (
        <>
          <div className="section-title">Attribute coverage (zero-shot)</div>
          <p className="meta-line">
            SigLIP label-bank classification over existing embeddings. Small
            slices are the dataset's long tail — click any bar to open that
            slice in the gallery.
          </p>
          <div className="charts">
            {coverage.map((g) => (
              <div className="panel" key={g.grp}>
                <h3>{g.grp.replace(/_/g, " ")}</h3>
                <ResponsiveContainer width="100%" height={Math.max(160, g.labels.length * 34)}>
                  <BarChart data={g.labels} layout="vertical">
                    <CartesianGrid stroke={GRID_STROKE} horizontal={false} />
                    <XAxis type="number" stroke={AXIS_STROKE} />
                    <YAxis type="category" dataKey="label" stroke={AXIS_STROKE} width={110} />
                    <Tooltip
                      contentStyle={TOOLTIP_STYLE}
                      formatter={(v, _n, item) =>
                        [`${v} (${((item?.payload?.fraction ?? 0) * 100).toFixed(1)}%)`, "count"]}
                    />
                    <Bar dataKey="count" fill={SERIES.blue} radius={[0, 4, 4, 0]}
                         cursor="pointer"
                         onClick={(data) => {
                           const label = (data as unknown as { label?: string }).label;
                           if (label) navigate(`/?attr=${encodeURIComponent(`${g.grp}:${label}`)}`);
                         }} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        </>
      )}

      <div className="section-title">
        Near-duplicate pairs {dups.length > 0 && `(${dups.length})`}
      </div>
      {dups.length === 0 ? (
        <div className="empty">
          {overview.embeddings_available
            ? "No near-duplicate pairs above the similarity threshold."
            : <>Requires embeddings — run <code>python -m app.ingest</code> first.</>}
        </div>
      ) : (
        <div className="dup-list">
          {dups.map((d, i) => (
            <div className="dup-pair" key={i}>
              <Link to={`/samples/${d.a.id}`}><img src={d.a.thumb_url} alt="" /></Link>
              <Link to={`/samples/${d.b.id}`}><img src={d.b.thumb_url} alt="" /></Link>
              <span className="pill score">{(d.similarity * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
