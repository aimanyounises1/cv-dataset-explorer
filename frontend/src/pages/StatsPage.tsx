import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api/client";
import type { AttributeGroup, CaptionStats, DuplicatePair, StatsOverview } from "../api/types";
import LeakagePanel from "../components/LeakagePanel";
import { AXIS_STROKE, GRID_STROKE, SERIES, TOOLTIP_STYLE } from "../lib/viz";
import "../styles/profile.css";

/** The five questions this page answers, in the order a reviewer asks them:
 * how big is it, can the splits be trusted, what is in it, are the captions
 * sound, and where did any of this come from.
 *
 * Each is a URL — `?view=<id>` on /stats — because "look at the caption
 * lengths" is a thing one researcher sends another, and a scroll position is
 * not a thing anyone can send. Bare /stats keeps working and means Overview,
 * so every link that already points at this page still lands somewhere real. */
const VIEWS = [
  { id: "overview", label: "Overview" },
  { id: "integrity", label: "Split integrity" },
  { id: "coverage", label: "Prompt slices" },
  { id: "captions", label: "Caption health" },
  { id: "provenance", label: "Provenance" },
] as const;

type ViewId = (typeof VIEWS)[number]["id"];

/** Longest-side size buckets, smallest first. They are an ordinal variable that
 * arrives as a JSON object, and an object's key order is the order the backend
 * happened to count them in (`backend/app/api/stats.py` fills a Counter in row
 * order), not the order a reader thinks in — so the reading order is stated
 * here rather than inherited from the serialization. A bucket the API adds
 * later sorts after the named ones instead of disappearing. */
const SIZE_BUCKET_ORDER = ["<400px", "400-499px", "500px+"];

/** Bar width cap, the same 54px the block views use
 * (components/blocks/BarBlockView, HistogramBlockView): with three splits in a
 * full-width panel an uncapped bar is wider than it is tall and reads as a
 * block, not a bar. Wants to live in lib/viz next to the other chart tokens —
 * deferred only because that file is open in another agent's working tree. */
const MAX_BAR_SIZE = 54;

const sizeBucketRank = (name: string) => {
  const i = SIZE_BUCKET_ORDER.indexOf(name);
  return i === -1 ? SIZE_BUCKET_ORDER.length : i;
};

const isViewId = (v: string | null): v is ViewId =>
  v !== null && VIEWS.some((x) => x.id === v);

// Overview is addressed by the bare path, so the page's canonical URL stays the
// one already in the rail, the command palette and the README. The hrefs
// themselves are written in LeftRail, which is where these views are listed.

export default function StatsPage() {
  const [overview, setOverview] = useState<StatsOverview | null>(null);
  const [captions, setCaptions] = useState<CaptionStats | null>(null);
  const [dups, setDups] = useState<DuplicatePair[]>([]);
  const [allDups, setAllDups] = useState(false);
  const [coverage, setCoverage] = useState<AttributeGroup[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sectionErrors, setSectionErrors] = useState<string[]>([]);
  const navigate = useNavigate();
  const [params] = useSearchParams();

  // An unrecognised ?view= lands on Overview rather than on a blank column,
  // and the URL is left alone: rewriting it would bury the reader's own typo
  // in the history stack where Back cannot get past it.
  const raw = params.get("view");
  const view: ViewId = isViewId(raw) ? raw : "overview";

  useEffect(() => {
    // A failed section must say so — an error rendering as an empty chart
    // would be indistinguishable from "nothing to report".
    const fail = (what: string) => () =>
      setSectionErrors((prev) => [...prev, what]);
    api.overview().then(setOverview).catch((e) => setError(String(e)));
    api.captionStats().then(setCaptions).catch(fail("caption statistics"));
    api.duplicates().then(setDups).catch(fail("near-duplicates"));
    api.coverage().then(setCoverage).catch(fail("zero-shot prompt slices"));
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!overview) return <div className="loading">Loading statistics…</div>;

  const splitData = Object.entries(overview.splits).map(([name, count]) => ({ name, count }));
  // Sorted at the point of construction, so the buckets read smallest-to-largest
  // whatever order the API serialized them in.
  const sizeBuckets = Object.entries(overview.image_size_buckets)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => sizeBucketRank(a.name) - sizeBucketRank(b.name) || a.name.localeCompare(b.name));
  // Samples whose width/height are stored: the backend skips rows without both,
  // so this is the population the percentages below are shares of.
  const sizeMeasured = sizeBuckets.reduce((n, b) => n + b.count, 0);

  return (
    <div>
      <h1 className="section-title" style={{ marginTop: 0 }}>Dataset profile</h1>

      {/* The views are navigation, and navigation lives in the rail — a second
          column of links on the same left edge read as one broken navigation.
          They are listed under "Dataset profile" in components/shell/LeftRail,
          which is also the only place that knows the benchmark is elsewhere.
          Every view is still its own URL, so Back and a pasted link work. */}
      <div className="profile-page">
        <div className="profile-body">
          {/* A failed section is announced wherever the reader is standing: an
              error discovered by switching views is an error that reads as an
              empty chart. */}
          {sectionErrors.length > 0 && (
            <div className="error">
              Failed to load: {sectionErrors.join(", ")}. The sections below may be incomplete.
            </div>
          )}

          {view === "overview" && (
            <section className="profile-view">
              <div className="section-title tight">Corpus at a glance</div>
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
                  <div className="label">
                    Semantic search ({overview.embed_provider === "qwen3_vl" ? "Qwen3-VL"
                      : overview.embed_provider === "siglip2" ? "SigLIP 2" : "off"})
                  </div>
                </div>
                <div className="stat-card">
                  <div className={`value ${overview.vlm_enriched > 0 ? "ok" : "warn"}`}>
                    {overview.vlm_enriched > 0 ? overview.vlm_enriched.toLocaleString() : "Off"}
                  </div>
                  <div className="label">VLM-enriched samples</div>
                </div>
              </div>

              {/* Image size is one line, not a chart: a distribution where one
                  bucket holds ~99% draws two bars too short to see, and a
                  reader has to squint at an axis to learn what a sentence says
                  outright. The counts stay complete so the tail is still
                  auditable. */}
              {sizeMeasured > 0 && (
                <p className="meta-line tight">
                  <strong>Image size (longest side):</strong>{" "}
                  {sizeBuckets.map((b, i) => (
                    <span key={b.name}>
                      {i > 0 && " · "}
                      {b.name} {b.count.toLocaleString()}{" "}
                      ({((b.count / sizeMeasured) * 100).toFixed(1)}%)
                    </span>
                  ))}
                  {" — "}
                  {sizeMeasured.toLocaleString()} of{" "}
                  {overview.total_samples.toLocaleString()} images carry stored dimensions.
                </p>
              )}

              <div className="panel">
                <h3>Samples per split</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={splitData}>
                    <CartesianGrid stroke={GRID_STROKE} vertical={false} />
                    <XAxis dataKey="name" stroke={AXIS_STROKE} />
                    <YAxis stroke={AXIS_STROKE} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Bar dataKey="count" fill={SERIES.blue} radius={[4, 4, 0, 0]}
                         maxBarSize={MAX_BAR_SIZE} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>
          )}

          {view === "integrity" && (
            /* Integrity leads its own view for the reason it used to lead the
               whole page: whether the splits can be trusted changes how every
               other number here is read. The headline, the ladder and the
               first judgeable pairs clear the fold at 1440x900 and 390x844. */
            <section className="profile-view">
              <div className="section-title tight">Train/test leakage</div>
              <p className="meta-line tight">
                A held-out image with a near-duplicate in training is a held-out
                image the model has effectively already seen the subject of. Move
                the threshold and look at the pairs — “near-duplicate” is a cut on
                a cosine, not a fact, and on this corpus the pairs turn out to be
                the same subject photographed again rather than the same
                photograph.
              </p>
              <div className="panel"><LeakagePanel /></div>

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
                <>
                  {/* The list arrives sorted by similarity, so the top rows are the
                      finding; two hundred rows of the same egret is scrolling, not
                      information. The rest stays one click away for auditing. */}
                  <div className="dup-list">
                    {(allDups ? dups : dups.slice(0, 9)).map((d, i) => (
                      <div className="dup-pair" key={i}>
                        <Link to={`/samples/${d.a.id}`}><img src={d.a.thumb_url} alt="" /></Link>
                        <Link to={`/samples/${d.b.id}`}><img src={d.b.thumb_url} alt="" /></Link>
                        <span className="pill score">{(d.similarity * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                  {dups.length > 9 && (
                    <button className="ghost dup-more" onClick={() => setAllDups(!allDups)}>
                      {allDups ? "Show only the top 9" : `Show all ${dups.length} pairs`}
                    </button>
                  )}
                </>
              )}
            </section>
          )}

          {view === "coverage" && (
            <section className="profile-view">
              <div className="section-title tight">Zero-shot prompt distribution</div>
              <p className="meta-line tight">
                Exploratory SigLIP assignments within each hand-authored prompt
                bank. These are review hypotheses, not ground-truth labels or an
                accuracy estimate. Ambiguous images abstain; click a bar to
                inspect its slice before drawing a long-tail conclusion.
              </p>
              {coverage.length === 0 ? (
                <div className="empty">
                  {overview.embeddings_available
                    ? "No attribute labels stored for this corpus."
                    : <>Requires embeddings — run <code>python -m app.ingest</code> first.</>}
                </div>
              ) : (
                <div className="charts">
                  {coverage.map((g) => (
                    <div className="panel" key={g.grp}>
                      <h3>{g.grp.replace(/_/g, " ")}</h3>
                      <p className="meta-line tight">
                        {g.labelled.toLocaleString()} assigned ·{" "}
                        {g.abstained.toLocaleString()} abstained
                      </p>
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
              )}
            </section>
          )}

          {view === "captions" && (
            <section className="profile-view">
              <div className="section-title tight">Caption health</div>
              <p className="meta-line tight">
                Five captions per image, written by crowdworkers. Length is the
                cheap proxy for how much a caption commits to; the word counts
                are what a keyword query is actually searching.
              </p>
              {captions === null ? (
                <div className="empty">
                  {sectionErrors.includes("caption statistics")
                    ? "Caption statistics failed to load — see the message above."
                    : "Loading caption statistics…"}
                </div>
              ) : (
                <div className="charts">
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
                </div>
              )}
            </section>
          )}

          {view === "provenance" && (
            /* Where the data came from and what is known to be off about it —
               a dataset tool that hides its own provenance is asking to be
               trusted on faith. It keeps a view of its own rather than a
               footnote at the bottom of one. */
            <section className="profile-view">
              <div className="section-title tight">Provenance</div>
              <p className="meta-line tight">
                What this copy of Flickr8k is, and every way it is known to
                differ from the distribution it is named after.
              </p>
              <details className="caveat" open>
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
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
