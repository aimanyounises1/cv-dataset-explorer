import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { AXES, AlbumSummary, SampleCard, ScenarioGroup, SearchMode, TermStat } from "../api/types";
import { AXIS_META } from "../components/AxisFilters";
import AxisLegend from "../components/AxisLegend";
import { albumsChanged } from "../components/AlbumShelf";
import AlbumHeader from "../components/AlbumHeader";
import ImageCard from "../components/ImageCard";
import { showToast } from "../components/Toast";
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
/* A rejected composed request arrives as "422: {json}" from the client. The
 * page shows the validator's own sentence, never the wire payload. */
const composedProblem = (msg: string): string => {
  try {
    const body = JSON.parse(msg.slice(msg.indexOf(":") + 1).trim());
    const detail = Array.isArray(body.detail) ? body.detail[0]?.msg : body.detail;
    if (typeof detail === "string" && detail) {
      return detail.replace(/^Value error, /, "");
    }
  } catch { /* fall through to the generic sentence */ }
  return "The composed query was rejected";
};

export default function GalleryPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const query = searchParams.get("q") ?? "";
  const mode = (searchParams.get("mode") ?? "hybrid") as SearchMode;
  const page = Math.max(1, Number(searchParams.get("page")) || 1);
  const sort = searchParams.get("sort") ?? "";
  const navigate = useNavigate();
  /* Reference images live in the URL (?like=76,2259&unlike=13): a composed
   * search is a search, and a colleague must be able to open the same one
   * from a pasted link. The similarity trail IS these chips plus the
   * browser's own history — stepping back is Back, narrowing is a chip. */
  const likeIds = useMemo(() => (searchParams.get("like") ?? "")
    .split(",").map(Number).filter((n) => Number.isInteger(n) && n > 0), [searchParams]);
  const unlikeIds = useMemo(() => (searchParams.get("unlike") ?? "")
    .split(",").map(Number).filter((n) => Number.isInteger(n) && n > 0), [searchParams]);
  const composed = likeIds.length > 0 || unlikeIds.length > 0;
  /* The server's validity rule, mirrored: a composed query needs text or at
   * least one positive reference. An exclusion alone has nothing to steer, so
   * the UI never sends that request — it shows the plain ranking with the
   * chip kept and says inline what would make the exclusion take effect. */
  const composedValid = likeIds.length > 0 || query.trim() !== "";
  const albumId = Number(searchParams.get("album")) || null;
  const [refThumbs, setRefThumbs] = useState<Record<number, string>>({});
  /* Scenario groups are proposals, not state: at most three, on demand,
   * temporary until "Save as album" makes one durable. */
  const [scenarios, setScenarios] = useState<ScenarioGroup[] | null>(null);
  const [scenBusy, setScenBusy] = useState(false);
  // Every membership constraint — split, tag, attribute, axes, id list, the
  // quality threshold, the cluster — is now owned by the rail and read back
  // from the URL. The gallery keeps only what orders or renders the view.
  const selection = useSelection();
  const [input, setInput] = useState(query);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const [items, setItems] = useState<SampleCard[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<SearchMeta>({ terms: [] });
  const [hasMore, setHasMore] = useState(false);
  const [density, setDensity] = useState<string>(
    () => localStorage.getItem(DENSITY_KEY) ?? "M");
  /* An image query cannot live in the URL — the query IS the image — so this
   * is the one result set held in memory instead. The URL stays the boss:
   * any change to it clears the image results, and the shareable artifact is
   * the ranked id list, offered as an ?ids= link. */
  const [imageQuery, setImageQuery] =
    useState<{ name: string; items: SampleCard[] } | null>(null);
  const [imageBusy, setImageBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);
  /* Select mode only changes what a click means: the picked set is
   * transient, and the album it feeds is the durable thing — a first-class
   * ordered collection with provenance, no longer a tag. Tags remain labels;
   * converting one into an album is an explicit act elsewhere. */
  const [selecting, setSelecting] = useState(false);
  const [picked, setPicked] = useState<Set<number>>(new Set());
  const [albumName, setAlbumName] = useState("");
  const [albumBusy, setAlbumBusy] = useState(false);
  const [albums, setAlbums] = useState<AlbumSummary[]>([]);

  useEffect(() => {
    const missing = [...likeIds, ...unlikeIds].filter((id) => !refThumbs[id]);
    missing.forEach((id) => {
      api.getSample(id)
        .then((d) => setRefThumbs((prev) => ({ ...prev, [id]: d.thumb_url })))
        .catch(() => {});
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [likeIds, unlikeIds]);

  const editRef = (id: number, role: "like" | "unlike", add: boolean) => {
    const read = (k: string) => new Set((searchParams.get(k) ?? "").split(",").filter(Boolean));
    const mine = read(role);
    const other = role === "like" ? "unlike" : "like";
    const theirs = read(other);
    if (add) { mine.add(String(id)); theirs.delete(String(id)); }
    else mine.delete(String(id));
    // Adding a reference PUSHES history — each step of the similarity trail
    // is a place Back can return to. Removing one replaces, because undoing
    // a removal is just adding again.
    setSearchParams((prev) => {
      const p = new URLSearchParams(prev);
      const write = (k: string, v: string) => { if (v) p.set(k, v); else p.delete(k); };
      write(role, [...mine].join(","));
      write(other, [...theirs].join(","));
      p.delete("page");
      return p;
    }, { replace: !add });
  };

  useEffect(() => {
    if (!selecting) return;
    // Existing albums feed the datalist so "add to an existing album" is one
    // keystroke, not a memory test.
    api.listAlbums().then(setAlbums).catch(() => {});
  }, [selecting]);

  /** A drag from a picked card carries the whole picked set — a selection is
   * one object, and dragging it should feel like moving that object. An
   * unpicked card drags alone even in select mode. */
  const getDragIds = (id: number) =>
    picked.has(id) && picked.size > 0 ? [...picked] : [id];

  const togglePick = (id: number) =>
    setPicked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const saveAlbum = () => {
    const name = albumName.trim();
    if (!name || picked.size === 0) return;
    setAlbumBusy(true);
    setError(null);
    const ids = [...picked];
    // A name that matches an existing album adds to it; a new name creates
    // it first. Albums are first-class now — the tray stopped writing tags
    // when albums stopped being tags.
    const existing = albums.find((a) => a.name === name);
    const ensure = existing
      ? Promise.resolve(existing.id)
      : api.createAlbum(name).then((a) => a.id);
    ensure
      .then((id) => api.addToAlbum(id, ids).then((r) => ({ id, added: r.added })))
      .then(({ id, added }) => {
        albumsChanged();
        showToast(added === ids.length
          ? `Added ${added} to “${name}”`
          : `Added ${added} to “${name}” — ${ids.length - added} already there`);
        // Land inside the album, where the set is revisitable, shareable and
        // exportable like any other slice.
        setSelecting(false);
        setPicked(new Set());
        setAlbumName("");
        setInput("");
        setParams({ album: String(id), tag: "", q: "", page: "" });
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setAlbumBusy(false));
  };

  const runImageSearch = (file: Blob & { name?: string }) => {
    if (!file.type.startsWith("image/")) return;
    setImageBusy(true);
    setError(null);
    api.searchByImage(file)
      .then((cards) => setImageQuery({ name: file.name || "pasted image", items: cards }))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setImageBusy(false));
  };

  // A copied image pasted anywhere on the page is a query; text pastes (the
  // search box, the id list) have no files attached and pass through untouched.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const f = e.clipboardData?.files?.[0];
      if (f && f.type.startsWith("image/")) runImageSearch(f);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    try { localStorage.setItem(DENSITY_KEY, density); } catch { /* non-essential */ }
  }, [density]);

  // Scroll restore for back-nav. Position is keyed by the full query string in
  // sessionStorage: scan position is ephemeral scan-order state like the result
  // order, not something a pasted link should reproduce — the URL deliberately
  // does not own it. Restored once per mount, after the first result set
  // renders, because before that the page has no height to scroll into.
  const scrollKey = `cvde-scroll:${searchParams.toString()}`;
  const scrollKeyRef = useRef(scrollKey);
  const pageRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => { scrollKeyRef.current = scrollKey; }, [scrollKey]);
  useEffect(() => {
    const save = () => {
      // Navigating away fires one last scroll event: the next page's shorter
      // document clamps the position to 0 before this listener is removed
      // (passive-effect cleanup runs after paint). Recording that 0 would
      // overwrite the very position this key exists to keep, so a scroll only
      // counts while the gallery's own DOM is still attached. Measured: the
      // clamp event arrives with the root already disconnected.
      if (!pageRef.current || !pageRef.current.isConnected) return;
      try { sessionStorage.setItem(scrollKeyRef.current, String(window.scrollY)); }
      catch { /* non-essential */ }
    };
    window.addEventListener("scroll", save, { passive: true });
    return () => window.removeEventListener("scroll", save);
  }, []);
  const restoredScroll = useRef(false);
  useEffect(() => {
    if (restoredScroll.current || loading || items.length === 0) return;
    restoredScroll.current = true;
    const saved = Number(sessionStorage.getItem(scrollKey) ?? NaN);
    if (Number.isFinite(saved) && saved > 0) window.scrollTo(0, saved);
  }, [loading, items.length, scrollKey]);

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

  const filterKey = [query, mode, sort, searchParams.get("like") ?? "",
    searchParams.get("unlike") ?? "", JSON.stringify(selection.params)].join("|");

  useEffect(() => {
    const ctrl = new AbortController();
    const run = async () => {
      setLoading(true);
      setError(null);
      // One page of whichever ranking is active. Search pages by offset into a
      // fusion held at a fixed depth, so pages partition one stable ranking.
      // Set when composed search falls back: the branch that ultimately
      // serves the request must carry this message instead of clearing it.
      let fallbackMsg: string | null = null;
      const fetchPage = async (p: number) => {
        if (composed && !composedValid) {
          fallbackMsg = "An exclusion alone can\u2019t steer a search \u2014 type a phrase, "
                      + "or add \u201cMore like this\u201d on a result. The excluded image "
                      + "is kept as a chip.";
        } else if (composed) {
          try {
            const res = await api.composedSearch({
              text: query || undefined,
              positive_ids: likeIds, negative_ids: unlikeIds,
              top_k: PER_PAGE, offset: (p - 1) * PER_PAGE,
              ...(selection.params as object),
            }, ctrl.signal);
            setNotice(res.degraded ? res.message ?? null : null);
            setMeta({ basis: res.score_basis, rrfK: null, terms: [],
                      depthLimit: res.depth_limit, depthReached: res.depth_reached });
            return { items: res.items, total: null as number | null, more: res.has_more };
          } catch (e) {
            if (e instanceof Error && e.message.startsWith("404")) {
              // The endpoint ships in a parallel lane: degrade honestly to the
              // text ranking rather than a blank page, and say so.
              fallbackMsg = "Composed search is not available on this backend yet — "
                          + "showing the unsteered ranking; reference chips are kept.";
            } else if (e instanceof Error && /^4\d\d:/.test(e.message)) {
              // A rejected request is explained in a sentence, never rendered
              // as the wire payload, and the page still shows a ranking.
              fallbackMsg = composedProblem(e.message)
                          + " — showing the unsteered ranking; reference chips are kept.";
            } else { throw e; }
          }
        }
        if (query) {
          const res = await api.search(
            query, mode,
            { ...selection.params, sort: sort || undefined,
              top_k: PER_PAGE, offset: (p - 1) * PER_PAGE }, ctrl.signal);
          setNotice(res.degraded ? res.message ?? null : fallbackMsg);
          setMeta({ basis: res.score_basis, rrfK: res.rrf_k, terms: res.term_stats ?? [],
                    idsResolved: res.ids_resolved, depthLimit: res.depth_limit,
                    depthReached: res.depth_reached });
          return { items: res.items, total: null as number | null, more: res.has_more };
        }
        const res = await api.listSamples(
          { page: p, per_page: PER_PAGE, ...selection.params,
            sort: sort || undefined }, ctrl.signal);
        setNotice(fallbackMsg);
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

  // The URL remains the source of truth: touching any filter, query or page
  // dismisses the in-memory image results rather than competing with them.
  useEffect(() => { setImageQuery(null); setScenarios(null); }, [filterKey, page]);

  const suggestGroups = () => {
    setScenBusy(true);
    api.scenarioGroups({
      text: query || undefined,
      positive_ids: likeIds, negative_ids: unlikeIds,
      ...(selection.params as object),
    })
      .then((r) => setScenarios(r.groups))
      .catch((e) => showToast(e instanceof Error && e.message.startsWith("404")
        ? "Scenario groups need the updated backend — not available yet"
        : "Could not group these results"))
      .finally(() => setScenBusy(false));
  };

  const saveGroupAsAlbum = (gr: ScenarioGroup) => {
    api.createAlbum(gr.label)
      .then((a) => api.addToAlbum(a.id, gr.sample_ids).then(() => a))
      .then((a) => { albumsChanged(); showToast(`Saved “${gr.label}” (${gr.sample_ids.length})`);
                     setParams({ album: String(a.id), q: "", like: "", unlike: "", page: "" }); })
      .catch((e) => showToast(e instanceof Error ? e.message : "Could not save album"));
  };

  const common = meta.terms.filter((t) => t.common);
  const missing = meta.terms.filter((t) => t.images === 0);
  const hasFilters = selection.active;
  // Every term matches something, yet the query returns nothing: the lexical
  // index ANDs terms, so a long query can be unsatisfiable while each of its
  // words is common. Without this the empty page looks like a broken search.
  const conjunctionFailed =
    items.length === 0 && meta.terms.length > 1 && missing.length === 0 && !hasFilters;


  return (
    <div ref={pageRef}
         onDragOver={(e) => { e.preventDefault(); }}
         onDrop={(e) => {
           const f = e.dataTransfer.files?.[0];
           if (f && f.type.startsWith("image/")) { e.preventDefault(); runImageSearch(f); }
         }}>
      <div className="controls">
        <div className="search-box">
          <input
            ref={searchRef}
            aria-label="Search images"
            placeholder='Search images… e.g. "dog jumping into water", "crowded market at night"'
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
        </div>
        <button className="ghost by-image" disabled={imageBusy}
                onClick={() => fileRef.current?.click()}
                title="Rank the corpus against a picture: pick a file, drop one anywhere on this page, or paste a copied image">
          {imageBusy ? "Embedding…" : "By image…"}
        </button>
        <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
               onChange={(e) => {
                 const f = e.target.files?.[0];
                 if (f) runImageSearch(f);
                 e.target.value = "";   // the same file, picked twice, still fires
               }} />
        {/* One command bar: the query's dials live behind a single quiet
            button — the spec's own rule (advanced controls hidden until
            requested). The dot marks a non-default dial, so collapsing never
            silently hides a live choice. */}
        <details className="search-settings">
          <summary title="Mode, ordering and density">
            Search settings
            {(mode !== "hybrid" || sort !== "") && (
              <span className="settings-dot" aria-hidden="true" />
            )}
          </summary>
          <div className="search-settings-pop">
            <div className="setting-row">
              <span className="eyebrow">Mode</span>
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
            </div>
            <div className="setting-row">
              <span className="eyebrow">Order</span>
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
            </div>
            <div className="setting-row">
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
          </div>
        </details>
        <button className={`ghost${selecting ? " select-on" : ""}`}
                aria-pressed={selecting}
                onClick={() => { setSelecting(!selecting); setPicked(new Set()); }}
                title="Hand-pick images into an album — a tag you can filter by, revisit and export">
          {selecting ? "Done picking" : "Select"}
        </button>
      </div>

      {/* The tray exists only while something is picked — a tray of zero
          would be furniture. Fixed to the viewport bottom so the running
          count and the album action stay in reach however deep the scroll,
          which is exactly when hand-picking happens. */}
      {selecting && picked.size > 0 && (
        <div className="selection-tray" role="toolbar" aria-label="Selected images">
          <span className="select-count" aria-live="polite">
            {picked.size} picked
          </span>
          <input list="album-names" value={albumName}
                 onChange={(e) => setAlbumName(e.target.value)}
                 placeholder="Album name…"
                 aria-label="Album name" />
          <datalist id="album-names">
            {albums.map((a) => <option key={a.id} value={a.name} />)}
          </datalist>
          <button className="primary" disabled={!picked.size || !albumName.trim() || albumBusy}
                  onClick={saveAlbum}>
            {albumBusy ? "Saving…" : `Add ${picked.size || ""} to album`}
          </button>
          <button className="ghost" disabled={!picked.size}
                  onClick={() => setPicked(new Set())}>
            Clear
          </button>
          {picked.size === 2 && (
            <button className="ghost"
                    title="Open these two side by side with synchronized zoom"
                    onClick={() => {
                      const [a, b] = [...picked];
                      navigate(`/compare?a=${a}&b=${b}`);
                    }}>
              Compare
            </button>
          )}
          <button className="ghost"
                  onClick={() => { setSelecting(false); setPicked(new Set()); }}>
            Done
          </button>
        </div>
      )}






      {composed && (
        <div className="ref-row" aria-label="Reference images steering this search">
          {likeIds.map((id) => (
            <span className="ref-chip like" key={`l${id}`}
                  title="Positive reference — results should feel like this">
              {refThumbs[id] ? <img src={refThumbs[id]} alt="" /> : <span className="ref-ph" />}
              <button aria-label={`Remove reference ${id}`}
                      onClick={() => editRef(id, "like", false)}>×</button>
            </span>
          ))}
          {unlikeIds.map((id) => (
            <span className="ref-chip unlike" key={`u${id}`}
                  title="Negative example — push results away from this">
              {refThumbs[id] ? <img src={refThumbs[id]} alt="" /> : <span className="ref-ph" />}
              <button aria-label={`Remove exclusion ${id}`}
                      onClick={() => editRef(id, "unlike", false)}>×</button>
            </span>
          ))}
          <span className="ref-hint">
            {query ? `steered by “${query}”` : "type to steer — “but at night”, “in a village”…"}
          </span>
          <button className="ghost" onClick={() => setParams({ like: "", unlike: "", page: "" })}>
            Clear references
          </button>
        </div>
      )}

      {/* The album, editable where it is read: name, details, analysis. */}
      {albumId != null && (
        <AlbumHeader albumId={albumId}
                     onGone={() => setParams({ album: "", page: "" })} />
      )}

      {/* The hero exists only on the bare gallery — the workspace's front
          door. The moment a query, filter or image lands, it yields the room
          back to results. Its h1 is also the page's heading. */}
      {!query && !selection.active && !imageQuery && !composed && (
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">Local visual intelligence</p>
            <h1>Find the moments that matter.</h1>
            <p className="hero-sub">
              Search, understand and curate this corpus with local models and
              agents that show their work.
            </p>
          </div>
          <div className="hero-flows">
            <button type="button" className="flow-card" onClick={() => searchRef.current?.focus()}>
              <span className="flow-no">01</span>
              <span className="flow-name">Find</span>
              <span className="flow-hint">Words, an image, or both — every score labelled.</span>
            </button>
            <button type="button" className="flow-card"
                    onClick={() => { setSelecting(true); window.scrollTo({ top: 0 }); }}>
              <span className="flow-no">02</span>
              <span className="flow-name">Curate</span>
              <span className="flow-hint">Pick images into albums; drag a card to file it.</span>
            </button>
            <Link className="flow-card" to="/quality">
              <span className="flow-no">03</span>
              <span className="flow-name">Audit</span>
              <span className="flow-hint">Captions their own images don’t support.</span>
            </Link>
            <Link className="flow-card" to="/chat">
              <span className="flow-no">04</span>
              <span className="flow-name">Ask</span>
              <span className="flow-hint">Agents that search and show their work.</span>
            </Link>
          </div>
        </section>
      )}

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

      {/* Image-query results replace the grid but never the URL: the image
          cannot travel in a link, so the ranked ids are offered as the
          shareable ?ids= slice instead, and any URL change dismisses this. */}
      {imageQuery && (
        <>
          <div className="result-bar">
            <div className="meta-line" aria-live="polite" style={{ marginBottom: 0 }}>
              {imageQuery.items.length} images ranked against{" "}
              <strong>“{imageQuery.name}”</strong> · cosine, same index as text search
              {" · "}
              <a className="open-all"
                 href={`/?ids=${imageQuery.items.map((s) => s.id).join(",")}`}>
                open as id list →
              </a>
            </div>
            <button className="ghost" onClick={() => setImageQuery(null)}>
              × Clear image query
            </button>
          </div>
          <div className="grid"
               style={{ "--frame-min": DENSITY[density] } as React.CSSProperties}>
            {imageQuery.items.map((s) => (
              <ImageCard key={s.id} sample={s} scoreBasis="cosine"
                         selectMode={selecting} selected={picked.has(s.id)}
                         onToggleSelect={togglePick} getDragIds={getDragIds} />
            ))}
          </div>
        </>
      )}

      {!imageQuery && (<>
      <div className="result-bar">
        <div className="meta-line" aria-live="polite" style={{ marginBottom: 0 }}>
          {query
            ? `${items.length}${hasMore ? "+" : ""} result${items.length === 1 ? "" : "s"} for “${query}” (${mode})`
            : `${total.toLocaleString()} samples`}
          {query && meta.basis === "rrf" && meta.rrfK != null &&
            ` · fused by reciprocal rank, k=${meta.rrfK}`}
          {(query || composed) && items.length >= 8 && (
            <>
              {" · "}
              <button className="link-btn" onClick={suggestGroups} disabled={scenBusy}>
                {scenBusy ? "grouping…" : "suggest groups"}
              </button>
            </>
          )}
        </div>
        {/* The key for the sparkline every card carries. Only shown when the
            cards actually have axes — a legend for an absent encoding is
            noise, and axes are absent until `python -m app.analyze` has run. */}
        {items.some((s) => s.axes) && <AxisLegend />}
      </div>

      {scenarios && scenarios.length > 0 && (
        <div className="scenario-row">
          {scenarios.map((gr) => (
            <div className="scenario-card" key={gr.label}>
              <div className="scenario-head">
                <strong>{gr.label}</strong>
                <span className="scenario-count">{gr.count}</span>
              </div>
              <div className="scenario-evidence">{gr.evidence}</div>
              <div className="scenario-actions">
                <Link className="open-all"
                      to={`/?ids=${gr.sample_ids.join(",")}`}>
                  Open {gr.count} →
                </Link>
                <button className="ghost" onClick={() => saveGroupAsAlbum(gr)}>
                  Save as album
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="grid"
           style={{ "--frame-min": DENSITY[density] } as React.CSSProperties}>
        {/* Each card links with the query, mode and score that put it here.
            `items` accumulates every page loaded so far, so the array index is
            already the rank within the whole result set: the first card on
            page 3 is rank 121, not rank 1. */}
        {items.map((s, i) => (
          <ImageCard key={s.id} sample={s} scoreBasis={meta.basis}
                     query={query} mode={mode} rank={i + 1}
                     selectMode={selecting} selected={picked.has(s.id)}
                     onToggleSelect={togglePick} getDragIds={getDragIds}
                     onLike={(id) => editRef(id, "like", true)}
                     onExclude={(id) => editRef(id, "unlike", true)} />
        ))}
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
      </>)}
    </div>
  );
}
