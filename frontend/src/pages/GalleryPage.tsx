import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { SampleCard, SearchMode, TermStat } from "../api/types";
import FilterBar, { Filters } from "../components/FilterBar";
import ImageCard from "../components/ImageCard";
import { useDebounce } from "../hooks/useDebounce";
import { saveResultOrder } from "../hooks/useResultOrder";

interface SearchMeta {
  basis?: string | null;
  rrfK?: number | null;
  terms: TermStat[];
}

const PER_PAGE = 60;
const MODES: SearchMode[] = ["hybrid", "semantic", "keyword"];
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
  const filters: Filters = useMemo(() => ({
    split: searchParams.get("split") ?? "",
    tag: searchParams.get("tag") ?? "",
    vlm_tag: searchParams.get("vlm_tag") ?? "",
    attr: searchParams.get("attr") ?? "",
  }), [searchParams]);

  const [input, setInput] = useState(query);
  const [items, setItems] = useState<SampleCard[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<SearchMeta>({ terms: [] });

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

  const filterKey = `${query}|${mode}|${filters.split}|${filters.tag}|${filters.vlm_tag}|${filters.attr}`;

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
          setMeta({ basis: res.score_basis, rrfK: res.rrf_k, terms: res.term_stats ?? [] });
        } else if (page > 1 && itemsRef.current.length === (page - 1) * PER_PAGE) {
          // "Load more": append just the next page.
          const res = await api.listSamples({ page, per_page: PER_PAGE, ...filters }, ctrl.signal);
          setItems((prev) => [...prev, ...res.items]);
          setTotal(res.total);
          setNotice(null);
          setMeta({ terms: [] });
        } else {
          // Fresh mount (possibly at ?page=N after back-navigation): load 1..N.
          let all: SampleCard[] = [];
          let count = 0;
          for (let p = 1; p <= page; p++) {
            const res = await api.listSamples({ page: p, per_page: PER_PAGE, ...filters }, ctrl.signal);
            all = all.concat(res.items);
            count = res.total;
            if (res.items.length < PER_PAGE) break;
          }
          setItems(all);
          setTotal(count);
          setNotice(null);
          setMeta({ terms: [] });
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

  const common = meta.terms.filter((t) => t.common);
  const missing = meta.terms.filter((t) => t.images === 0);
  const hasFilters = Boolean(filters.split || filters.tag || filters.vlm_tag || filters.attr);
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
                : "Reciprocal-rank fusion of semantic + keyword"
              }
            >
              {m}
            </button>
          ))}
        </div>
        <FilterBar
          filters={filters}
          onChange={(f) => setParams({
            split: f.split, tag: f.tag, vlm_tag: f.vlm_tag, attr: f.attr, page: "",
          })}
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

      <div className="meta-line" aria-live="polite">
        {query
          ? `${items.length} result${items.length === 1 ? "" : "s"} for “${query}” (${mode})`
          : `${total.toLocaleString()} samples`}
        {query && meta.basis === "rrf" && meta.rrfK != null &&
          ` · fused by reciprocal rank, k=${meta.rrfK}`}
      </div>

      <div className="grid">
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

      {!query && items.length < total && (
        <div className="load-more">
          <button className="primary" onClick={() => setParams({ page: String(page + 1) })}
                  disabled={loading}>
            Load more ({items.length.toLocaleString()} / {total.toLocaleString()})
          </button>
        </div>
      )}
    </div>
  );
}
