import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { SampleCard, SearchMode } from "../api/types";
import FilterBar, { Filters } from "../components/FilterBar";
import ImageCard from "../components/ImageCard";

const PER_PAGE = 60;
const MODES: SearchMode[] = ["hybrid", "semantic", "keyword"];
const SUGGESTIONS = [
  "a dog jumping into water",
  "children playing soccer",
  "climbing a steep rock face",
  "a crowded street at night",
  "splashing through snow",
];

/** All search/filter state lives in the URL: shareable links, working
 * back-button, and filters persist when navigating to a sample and back. */
export default function GalleryPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const query = searchParams.get("q") ?? "";
  const mode = (searchParams.get("mode") ?? "hybrid") as SearchMode;
  const filters: Filters = useMemo(() => ({
    split: searchParams.get("split") ?? "",
    tag: searchParams.get("tag") ?? "",
    vlm_tag: searchParams.get("vlm_tag") ?? "",
    attr: searchParams.get("attr") ?? "",
  }), [searchParams]);

  const [input, setInput] = useState(query);
  const [items, setItems] = useState<SampleCard[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  // Debounce typed input into the URL.
  useEffect(() => {
    const t = setTimeout(() => {
      if (input.trim() !== query) setParams({ q: input.trim() });
    }, 400);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input]);

  const filterKey = `${query}|${mode}|${filters.split}|${filters.tag}|${filters.vlm_tag}|${filters.attr}`;
  useEffect(() => { setPage(1); }, [filterKey]);

  useEffect(() => {
    const ctrl = new AbortController();
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        if (query) {
          const res = await api.search(query, mode, { ...filters, top_k: 60 }, ctrl.signal);
          setItems(res.items);
          setTotal(res.items.length);
          setNotice(res.degraded ? res.message ?? null : null);
        } else {
          const res = await api.listSamples(
            { page, per_page: PER_PAGE, ...filters }, ctrl.signal);
          setItems((prev) => (page > 1 ? [...prev, ...res.items] : res.items));
          setTotal(res.total);
          setNotice(null);
        }
        setLoading(false);
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") return; // superseded
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      }
    };
    void run();
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filterKey, page]);

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
              onClick={() => setParams({ mode: m === "hybrid" ? "" : m })}
              title={
                m === "semantic" ? "SigLIP text-to-image similarity"
                : m === "keyword" ? "BM25 full-text over captions + VLM tags (Porter-stemmed)"
                : "Reciprocal-rank fusion of semantic + keyword"
              }
            >
              {m}
            </button>
          ))}
        </div>
        <FilterBar
          filters={filters}
          onChange={(f) => setParams({ split: f.split, tag: f.tag, vlm_tag: f.vlm_tag, attr: f.attr })}
        />
      </div>

      {!query && (
        <div className="chip-row">
          <span className="chip-label">Try:</span>
          {SUGGESTIONS.map((s) => (
            <button key={s} className="chip" onClick={() => setInput(s)}>{s}</button>
          ))}
        </div>
      )}

      {notice && <div className="notice">{notice}</div>}
      {error && <div className="error">{error}</div>}

      <div className="meta-line" aria-live="polite">
        {query
          ? `${items.length} result${items.length === 1 ? "" : "s"} for “${query}” (${mode})`
          : `${total.toLocaleString()} samples`}
      </div>

      <div className="grid">
        {items.map((s) => <ImageCard key={s.id} sample={s} />)}
      </div>

      {loading && <div className="loading">Loading…</div>}
      {!loading && items.length === 0 && <div className="empty">No samples found.</div>}

      {!query && items.length < total && (
        <div className="load-more">
          <button className="primary" onClick={() => setPage((p) => p + 1)} disabled={loading}>
            Load more ({items.length.toLocaleString()} / {total.toLocaleString()})
          </button>
        </div>
      )}
    </div>
  );
}
