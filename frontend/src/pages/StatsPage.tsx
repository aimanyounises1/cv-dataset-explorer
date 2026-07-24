import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api/client";
import type { AttributeGroup, CaptionStats, DuplicatePair, StatsOverview } from "../api/types";

export default function StatsPage() {
  const [overview, setOverview] = useState<StatsOverview | null>(null);
  const [captions, setCaptions] = useState<CaptionStats | null>(null);
  const [dups, setDups] = useState<DuplicatePair[]>([]);
  const [coverage, setCoverage] = useState<AttributeGroup[]>([]);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    api.overview().then(setOverview).catch((e) => setError(String(e)));
    api.captionStats().then(setCaptions).catch(() => undefined);
    api.duplicates().then(setDups).catch(() => undefined);
    api.coverage().then(setCoverage).catch(() => undefined);
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

      <div className="charts">
        <div className="panel">
          <h3>Samples per split</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={splitData}>
              <CartesianGrid stroke="#2a3342" vertical={false} />
              <XAxis dataKey="name" stroke="#9aa4b2" />
              <YAxis stroke="#9aa4b2" />
              <Tooltip contentStyle={{ background: "#161b24", border: "1px solid #2a3342" }} />
              <Bar dataKey="count" fill="#4f9cff" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="panel">
          <h3>Image size (longest side)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={sizeData}>
              <CartesianGrid stroke="#2a3342" vertical={false} />
              <XAxis dataKey="name" stroke="#9aa4b2" />
              <YAxis stroke="#9aa4b2" />
              <Tooltip contentStyle={{ background: "#161b24", border: "1px solid #2a3342" }} />
              <Bar dataKey="count" fill="#3ecf8e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        {captions && (
          <>
            <div className="panel">
              <h3>Caption length distribution (words)</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={captions.length_histogram}>
                  <CartesianGrid stroke="#2a3342" vertical={false} />
                  <XAxis dataKey="bucket" stroke="#9aa4b2" />
                  <YAxis stroke="#9aa4b2" />
                  <Tooltip contentStyle={{ background: "#161b24", border: "1px solid #2a3342" }} />
                  <Bar dataKey="count" fill="#c792ea" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="panel">
              <h3>Most frequent caption words</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={captions.top_words.slice(0, 15)}>
                  <CartesianGrid stroke="#2a3342" vertical={false} />
                  <XAxis dataKey="word" stroke="#9aa4b2" interval={0} angle={-35} textAnchor="end" height={60} />
                  <YAxis stroke="#9aa4b2" />
                  <Tooltip contentStyle={{ background: "#161b24", border: "1px solid #2a3342" }} />
                  <Bar dataKey="count" fill="#ffbe50" radius={[4, 4, 0, 0]} />
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
                    <CartesianGrid stroke="#2a3342" horizontal={false} />
                    <XAxis type="number" stroke="#9aa4b2" />
                    <YAxis type="category" dataKey="label" stroke="#9aa4b2" width={110} />
                    <Tooltip
                      contentStyle={{ background: "#161b24", border: "1px solid #2a3342" }}
                      formatter={(v, _n, item) =>
                        [`${v} (${((item?.payload?.fraction ?? 0) * 100).toFixed(1)}%)`, "count"]}
                    />
                    <Bar dataKey="count" fill="#4f9cff" radius={[0, 4, 4, 0]}
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
          No near-duplicates found (or embeddings not computed yet).
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
