import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { SampleCard, SearchMode } from "../api/types";
import FilterBar, { Filters } from "../components/FilterBar";
import ImageCard from "../components/ImageCard";
import { useDebounce } from "../hooks/useDebounce";

const PER_PAGE = 60;
const MODES: SearchMode[] = ["hybrid", "semantic", "keyword"];

export default function GalleryPage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [filters, setFilters] = useState<Filters>({ split: "", tag: "", vlm_tag: "" });
  const [items, setItems] = useState<SampleCard[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const debouncedQuery = useDebounce(query.trim(), 400);
  const searching = debouncedQuery.length > 0;

  const load = useCallback(
    async (pageToLoad: number, append: boolean) => {
      setLoading(true);
      setError(null);
      try {
        if (searching) {
          const res = await api.search(debouncedQuery, mode, {
            split: filters.split, tag: filters.tag, vlm_tag: filters.vlm_tag,
            top_k: 60,
          });
          setItems(res.items);
          setTotal(res.items.length);
          setNotice(res.degraded ? res.message ?? null : null);
        } else {
          const res = await api.listSamples({
            page: pageToLoad, per_page: PER_PAGE,
            split: filters.split, tag: filters.tag, vlm_tag: filters.vlm_tag,
          });
          setItems((prev) => (append ? [...prev, ...res.items] : res.items));
          setTotal(res.total);
          setNotice(null);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [searching, debouncedQuery, mode, filters],
  );

  useEffect(() => {
    setPage(1);
    void load(1, false);
  }, [load]);

  const loadMore = () => {
    const next = page + 1;
    setPage(next);
    void load(next, true);
  };

  return (
    <div>
      <div className="controls">
        <div className="search-box">
          <input
            placeholder='Search images… e.g. "dog jumping into water", "crowded market at night"'
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div className="mode-toggle">
          {MODES.map((m) => (
            <button
              key={m}
              className={mode === m ? "active" : ""}
              onClick={() => setMode(m)}
              title={
                m === "semantic" ? "SigLIP text-to-image similarity"
                : m === "keyword" ? "BM25 full-text over captions + VLM tags"
                : "Reciprocal-rank fusion of semantic + keyword"
              }
            >
              {m}
            </button>
          ))}
        </div>
        <FilterBar filters={filters} onChange={setFilters} />
      </div>

      {notice && <div className="notice">{notice}</div>}
      {error && <div className="error">{error}</div>}

      <div className="meta-line">
        {searching
          ? `${items.length} result${items.length === 1 ? "" : "s"} for “${debouncedQuery}” (${mode})`
          : `${total.toLocaleString()} samples`}
      </div>

      <div className="grid">
        {items.map((s) => <ImageCard key={s.id} sample={s} />)}
      </div>

      {loading && <div className="loading">Loading…</div>}
      {!loading && items.length === 0 && <div className="empty">No samples found.</div>}

      {!searching && items.length < total && (
        <div className="load-more">
          <button className="primary" onClick={loadMore} disabled={loading}>
            Load more ({items.length.toLocaleString()} / {total.toLocaleString()})
          </button>
        </div>
      )}
    </div>
  );
}
