import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { AXES, SampleCard, SearchMode, TermStat } from "../api/types";
import { AXIS_META } from "../components/AxisFilters";
import AxisLegend from "../components/AxisLegend";
import ImageCard from "../components/ImageCard";
import { useDebounce } from "../hooks/useDebounce";
import { useSelection } from "../hooks/useSelection";
import { saveResultOrder } from "../hooks/useResultOrder";

interface SearchMeta {
  basis?: string | null;
  rrfK?: number | null;
  terms: TermStat[];
  idsResolved?: number | null;
  depthLimit?: number;
  depthReached?: boolean;
}

const PER_PAGE = 60;
const MODES: SearchMode[] = ["hybrid", "semantic", "keyword", "boosted"];

/** Frame width per density. Scanning thousands of thumbnails for one anomaly
 * and reading a handful of captions closely are different jobs. */
const DENSITY: Record<string, string> = { S: "128px", M: "190px", L: "280px" };
const DENSITY_KEY = "cvde-density";
const SUGGESTIONS = [
  "a dog jumping into water",
  "children playing soccer",
  "climbing a steep rock face",
  "a crowded street at night",
  "splashing through snow",
];

/** All search/filter state — including pagination depth — lives in the URL:
 * shareable links, working back-button, and "Load more" depth survives
 * navigating to a sample and back. */
export default function GalleryPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const query = searchParams.get("q") ?? "";
  const mode = (searchParams.get("mode") ?? "hybrid") as SearchMode;
  const page = Math.max(1, Number(searchParams.get("page")) || 1);
  const sort = searchParams.get("sort") ?? "";
  // Every membership constraint — split, tag, attribute, axes, id list, the
  // quality threshold, the cluster — is now owned by the rail and read back
  // from the URL. The gallery keeps only what orders or renders the view.
  const selection = useSelection();
  const [input, setInput] = useState(query);
  const [items, setItems] = useState<SampleCard[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<SearchMeta>({ terms: [] });
  const [hasMore, setHasMore] = useState(false);
  const [density, setDensity] = useState<string>(
    () => localStorage.getItem(DENSITY_KEY) ?? "M");

  useEffect(() => {
    try { localStorage.setItem(DENSITY_KEY, density); } catch { /* non-essential */ }
  }, [density]);

  // Lets the fetch effect see the current list without depending on it.
  const itemsRef = useRef<SampleCard[]>([]);
  useEffect(() => { itemsRef.current = items; }, [items]);

  // Publish the result order for keyboard triage on the sample page.
  useEffect(() => { saveResultOrder(items.map((s) => s.id)); }, [items]);

  const setParams = (updates: Record<string, string>) => {
    setSearchParams((prev) => {
      const p = new URLSearchParams(prev);
      for (const [k, v] of Object.entries(updates)) {
        if (v) p.set(k, v);
        else p.delete(k);
      }
      return p;
    }, { replace: true });
  };

  // Keep the input in sync when the URL changes externally (back/forward).
  useEffect(() => { setInput(query); }, [query]);

  // Debounce typed input into the URL; a query change restarts pagination.
  const debouncedInput = useDebounce(input, 400);
  useEffect(() => {
    if (debouncedInput.trim() !== query) setParams({ q: debouncedInput.trim(), page: "" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedInput]);

  const filterKey = [query, mode, sort, JSON.stringify(selection.params)].join("|");

  useEffect(() => {
    const ctrl = new AbortController();
    const run = async () => {
      setLoading(true);
      setError(null);
      // One page of whichever ranking is active. Search pages by offset into a
      // fusion held at a fixed depth, so pages partition one stable ranking.
      const fetchPage = async (p: number) => {
        if (query) {
          const res = await api.search(
            query, mode,
            { ...selection.params, sort: sort || undefined,
              top_k: PER_PAGE, offset: (p - 1) * PER_PAGE }, ctrl.signal);
          setNotice(res.degraded ? res.message ?? null : null);
          setMeta({ basis: res.score_basis, rrfK: res.rrf_k, terms: res.term_stats ?? [],
                    idsResolved: res.ids_resolved, depthLimit: res.depth_limit,
                    depthReached: res.depth_reached });
          return { items: res.items, total: null as number | null, more: res.has_more };
        }
        const res = await api.listSamples(
          { page: p, per_page: PER_PAGE, ...selection.params,
            sort: sort || undefined }, ctrl.signal);
        setNotice(null);
        setMeta({ terms: [] });
        return { items: res.items, total: res.total, more: p * PER_PAGE < res.total };
      };

      try {
        if (page > 1 && itemsRef.current.length === (page - 1) * PER_PAGE) {
          // "Load more": append just the next page.
          const res = await fetchPage(page);
          setItems((prev) => [...prev, ...res.items]);
          if (res.total != null) setTotal(res.total);
          setHasMore(res.more);
        } else {
          // Fresh mount (possibly at ?page=N after back-navigation): load 1..N.
          let all: SampleCard[] = [];
          let count: number | null = null;
          let more = false;
          for (let p = 1; p <= page; p++) {
            const res = await fetchPage(p);
            all = all.concat(res.items);
            count = res.total;
            more = res.more;
            if (res.items.length < PER_PAGE) break;
          }
          setItems(all);
          setTotal(count ?? all.length);
          setHasMore(more);
        }
        setLoading(false);
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") return; // superseded
        setError(e instanceof Error ? e.message : String(e));
        // Drop the previous answer. Leaving it on screen was worse than an empty
        // grid, because the heading and the cards came from different queries: a
        // failed `boosted` request kept the earlier hybrid results, relabelled
        // them "60+ results for … (boosted)", and left every card wearing an
        // `rrf 0.007` badge and a "fused by reciprocal rank, k=60" note that the
        // boosted path never produces. The error was visible and still the page
        // read as an answer.
        setItems([]);
        setTotal(0);
        setHasMore(false);
        setMeta({ terms: [] });
        setLoading(false);
      }
    };
    void run();
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filterKey, page]);

  const common = meta.terms.filter((t) => t.common);
  const missing = meta.terms.filter((t) => t.images === 0);
  const hasFilters = selection.active;
  // Every term matches something, yet the query returns nothing: the lexical
  // index ANDs terms, so a long query can be unsatisfiable while each of its
  // words is common. Without this the empty page looks like a broken search.
  const conjunctionFailed =
    items.length === 0 && meta.terms.length > 1 && missing.length === 0 && !hasFilters;


  return (
    <div>
      <div className="controls">
        <div className="search-box">
          <input
            aria-label="Search images"
            placeholder='Search images… e.g. "dog jumping into water", "crowded market at night"'
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
        </div>
        <div className="mode-toggle" role="group" aria-label="Search mode">
          {MODES.map((m) => (
            <button
              key={m}
              className={mode === m ? "active" : ""}
              aria-pressed={mode === m}
              onClick={() => setParams({ mode: m === "hybrid" ? "" : m, page: "" })}
              title={
                m === "semantic" ? "SigLIP text-to-image similarity"
                : m === "keyword" ? "BM25 full-text over captions + VLM tags (Porter-stemmed)"
                : m === "boosted" ? "Semantic ranking through PRISM speaker models trained on "
                  + "this corpus (measured +2.2 pts R@1; falls back to semantic if untrained)"
                : "Reciprocal-rank fusion of semantic + keyword"
              }
            >
              {m}
            </button>
          ))}
        </div>
        <select value={sort} aria-label="Sort results"
                onChange={(e) => setParams({ sort: e.target.value, page: "" })}>
          <option value="">Sort: relevance</option>
          {AXES.map((a) => [
            <option key={`${a}_desc`} value={`${a}_desc`}>
              Sort: {AXIS_META[a].label} — hardest first
            </option>,
            <option key={`${a}_asc`} value={`${a}_asc`}>
              Sort: {AXIS_META[a].label} — easiest first
            </option>,
          ])}
        </select>
        <div className="density">
          <span className="density-label">Size</span>
          <div className="density-group" role="group" aria-label="Thumbnail size">
            {Object.keys(DENSITY).map((d) => (
              <button key={d} className={density === d ? "active" : ""}
                      aria-pressed={density === d}
                      onClick={() => setDensity(d)}>{d}</button>
            ))}
          </div>
        </div>
      </div>






      {/* Zero-state only. With a filter applied these are noise between
          the controls and the results the filter just produced. */}
      {!query && !selection.active && (
        <div className="chip-row">
          <span className="chip-label">Try:</span>
          {SUGGESTIONS.map((s) => (
            <button key={s} className="chip" onClick={() => setInput(s)}>{s}</button>
          ))}
        </div>
      )}

      {notice && <div className="notice">{notice}</div>}
      {error && <div className="error">{error}</div>}

      {/* Keyword ranking cannot discriminate on a term that most of the corpus
          shares, and a term nothing matches explains an empty page. Say both. */}
      {common.length > 0 && (
        <div className="notice">
          {common.map((t) => (
            <div key={t.term}>
              <strong>“{t.term}”</strong> appears in {t.images.toLocaleString()} images
              ({(t.fraction * 100).toFixed(0)}% of the dataset) — too common for keyword
              ranking to separate. Add another term, or switch to semantic search.
            </div>
          ))}
        </div>
      )}
      {missing.length > 0 && (
        <div className="notice">
          No caption contains {missing.map((t) => `“${t.term}”`).join(", ")}
          {mode === "keyword"
            ? " — keyword search matches caption text only. Semantic search can still find it."
            : " — only semantic matches contribute for that term."}
        </div>
      )}

      <div className="result-bar">
        <div className="meta-line" aria-live="polite" style={{ marginBottom: 0 }}>
          {query
            ? `${items.length}${hasMore ? "+" : ""} result${items.length === 1 ? "" : "s"} for “${query}” (${mode})`
            : `${total.toLocaleString()} samples`}
          {query && meta.basis === "rrf" && meta.rrfK != null &&
            ` · fused by reciprocal rank, k=${meta.rrfK}`}
        </div>
        {/* The key for the sparkline every card carries. Only shown when the
            cards actually have axes — a legend for an absent encoding is
            noise, and axes are absent until `python -m app.analyze` has run. */}
        {items.some((s) => s.axes) && <AxisLegend />}
      </div>

      <div className="grid"
           style={{ "--frame-min": DENSITY[density] } as React.CSSProperties}>
        {items.map((s) => <ImageCard key={s.id} sample={s} scoreBasis={meta.basis} />)}
      </div>

      {loading && <div className="loading">Loading…</div>}
      {!loading && items.length === 0 && (
        <div className="empty">
          No samples found.
          {missing.length > 0
            ? ` No caption contains ${missing.map((t) => `“${t.term}”`).join(", ")}.`
            : conjunctionFailed
              ? ` Keyword search requires every term in the same caption, and no
                  single caption contains all ${meta.terms.length} of them — even
                  though each one matches on its own (${meta.terms
                    .map((t) => `“${t.term}” ${t.images.toLocaleString()}`)
                    .join(", ")}). Drop a term, or switch to semantic search,
                  which matches meaning rather than words.`
              : hasFilters
                ? " Try clearing a filter — every active filter is applied before ranking."
                : ""}
        </div>
      )}

      {/* Paging stops where the fusion stopped ranking. Saying so beats a
          "Load more" button that silently disappears with matches left over. */}
      {!hasMore && query && meta.depthReached && (
        <div className="meta-line" style={{ textAlign: "center", marginTop: 18 }}>
          End of the ranked results — this query ranked its top{" "}
          {meta.depthLimit?.toLocaleString()}. More images match; narrow the query
          or add a filter to bring them into range.
        </div>
      )}

      {hasMore && (
        <div className="load-more">
          <button className="primary" onClick={() => setParams({ page: String(page + 1) })}
                  disabled={loading}>
            {loading ? "Loading…" : query
              ? `Load more results (${items.length.toLocaleString()} so far)`
              : `Load more (${items.length.toLocaleString()} / ${total.toLocaleString()})`}
          </button>
        </div>
      )}
    </div>
  );
}
