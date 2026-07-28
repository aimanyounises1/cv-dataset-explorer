import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { AlbumSummary, SampleCard, ScenarioGroup, SearchMode, TermStat } from "../api/types";
import AxisLegend from "../components/AxisLegend";
import SearchSettings from "../components/SearchSettings";
import { albumsChanged } from "../components/AlbumShelf";
import AlbumHeader from "../components/AlbumHeader";
import ImageCard from "../components/ImageCard";
import { showToast } from "../components/Toast";
import { useActiveProviderName } from "../lib/activeProvider";
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

/** Frame width per density. Scanning thousands of thumbnails for one anomaly
 * and reading a handful of captions closely are different jobs. */
const DENSITY: Record<string, string> = { S: "128px", M: "190px", L: "280px" };
const DENSITY_KEY = "cvde-density";

/** Recent queries, most recent first. Local recency is per-browser scan state
 * (like density), not shareable selection state — the URL owns the latter. */
const HISTORY_KEY = "cvde-search-history";
const HISTORY_SHOWN = 8;   // merged dropdown cap
const HISTORY_KEPT = 20;   // stored recency list cap

const readLocalHistory = (): string[] => {
  try {
    const v: unknown = JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]");
    return Array.isArray(v) ? v.filter((s): s is string => typeof s === "string") : [];
  } catch { return []; }
};

/** A server search_snapshot payload is opaque JSON with two writers — read
 * defensively, taking whichever field carries the query text. */
const snapshotQuery = (payload: Record<string, unknown>): string | null => {
  for (const k of ["q", "query", "text"]) {
    const v = payload[k];
    if (typeof v === "string" && v.trim()) return v.trim();
  }
  const qs = payload["query_string"];
  if (typeof qs === "string" && qs) {
    const q = new URLSearchParams(qs.replace(/^\?/, "")).get("q");
    if (q?.trim()) return q.trim();
  }
  return null;
};
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
  const albumId = Number(searchParams.get("album")) || null;
  // Labels naming the ACTIVE embedding model read the one truth source.
  const providerName = useActiveProviderName();
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
  /* Picking is modeless: every card carries its own check control, so a click
   * on the card always navigates and a click on the check always picks. The
   * picked set is transient — the album it feeds is the durable thing, a
   * first-class ordered collection with provenance, no longer a tag. Tags
   * remain labels; converting one into an album is an explicit act elsewhere. */
  const [picked, setPicked] = useState<Set<number>>(new Set());
  const [albumName, setAlbumName] = useState("");
  const [albumBusy, setAlbumBusy] = useState(false);
  const [albums, setAlbums] = useState<AlbumSummary[]>([]);
  /* Search history: a local recency list merged with the workspace trail's
   * search snapshots. Both are reads — the dropdown never writes the URL
   * until an entry is chosen. */
  const [localHist, setLocalHist] = useState<string[]>(readLocalHistory);
  const [serverHist, setServerHist] = useState<string[]>([]);
  const [histOpen, setHistOpen] = useState(false);
  const [histIdx, setHistIdx] = useState(-1);
  const histFetched = useRef(false);

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

  const anyPicked = picked.size > 0;
  useEffect(() => {
    if (!anyPicked) return;
    // Existing albums feed the datalist so "add to an existing album" is one
    // keystroke, not a memory test.
    api.listAlbums().then(setAlbums).catch(() => {});
  }, [anyPicked]);

  /** A drag from a picked card carries the whole picked set — a selection is
   * one object, and dragging it should feel like moving that object. An
   * unpicked card drags alone. */
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
    const name = file.name || "pasted image";
    api.searchByImage(file)
      .then((cards) => {
        setImageQuery({ name, items: cards });
        // The image itself is never stored — the trail records that a picture
        // was used and what it was called, which is what a person retracing
        // the session needs.
        api.recordActivity("image_search", { name, n: cards.length }).catch(() => {});
      })
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

  /* Keep the input in sync when the URL changes externally — Back/forward, a
   * chosen recent query, a removed chip. Only then: the URL carries the
   * TRIMMED query, so echoing it back into a box that already holds this query
   * deletes the space someone just typed, and the next word runs into the last
   * one ("dog on beach" typed with a pause becomes "dogon beach"). A box whose
   * text already commits to this query is left exactly as its owner left it. */
  useEffect(() => {
    setInput((cur) => (cur.trim() === query ? cur : query));
  }, [query]);

  // Debounce typed input into the URL; a query change restarts pagination.
  const debouncedInput = useDebounce(input, 400);
  useEffect(() => {
    if (debouncedInput.trim() !== query) setParams({ q: debouncedInput.trim(), page: "" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedInput]);

  /** Typing is not a sequence of searches. The box commits every 400 ms, so
   * "dog on beach" arrives as "dog", "dog on", "dog on beach" — each one a real
   * ranking, none of them a thing anyone meant to keep. A new entry therefore
   * absorbs the prefixes it grew out of, and only the phrase the person stopped
   * on survives in the list. */
  const rememberSearch = (q: string) => {
    const key = q.toLowerCase();
    const kept = readLocalHistory()
      .filter((s) => s.toLowerCase() !== key && !key.startsWith(s.toLowerCase()));
    const list = [q, ...kept].slice(0, HISTORY_KEPT);
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list)); }
    catch { /* non-essential */ }
    setLocalHist(list);
  };

  /** Local recency first (it is this person's own trail), then the workspace
   * snapshots; deduplicated case-insensitively; narrowed by whatever is
   * already typed, minus the exact query already on screen. */
  const historyItems = useMemo(() => {
    const needle = input.trim().toLowerCase();
    const seen = new Set<string>();
    const out: string[] = [];
    for (const s of [...localHist, ...serverHist]) {
      const t = s.trim();
      const key = t.toLowerCase();
      if (!t || seen.has(key) || key === needle) continue;
      if (needle && !key.includes(needle)) continue;
      seen.add(key);
      out.push(t);
      if (out.length === HISTORY_SHOWN) break;
    }
    return out;
  }, [input, localHist, serverHist]);

  // A shorter list can strand the highlight past the end; a changed list
  // makes the old position meaningless either way.
  useEffect(() => { setHistIdx(-1); }, [historyItems.length]);

  const applyHistory = (q: string) => {
    setInput(q);
    setParams({ q, page: "" });   // immediate — a chosen entry shouldn't wait out the debounce
    setHistOpen(false);
    setHistIdx(-1);
  };

  const onSearchFocus = () => {
    setHistOpen(true);
    setHistIdx(-1);
    if (histFetched.current) return;
    histFetched.current = true;
    // The workspace trail arrives lazily, on first focus: the dropdown is the
    // only reader, and most gallery visits never open it.
    api.listActivity(100).then((events) => {
      const qs: string[] = [];
      for (const ev of events) {
        if (ev.kind === "search_snapshot") {
          const q = snapshotQuery(ev.payload ?? {});
          if (q) qs.push(q);
        }
      }
      setServerHist(qs);
    }).catch(() => {});
  };

  const onSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown" && !histOpen && historyItems.length > 0) {
      e.preventDefault();
      setHistOpen(true);
      setHistIdx(0);
      return;
    }
    if (!histOpen || historyItems.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHistIdx((i) => (i + 1) % historyItems.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHistIdx((i) => (i <= 0 ? historyItems.length - 1 : i - 1));
    } else if (e.key === "Enter" && histIdx >= 0) {
      e.preventDefault();
      applyHistory(historyItems[histIdx]);
    } else if (e.key === "Escape") {
      setHistOpen(false);
      setHistIdx(-1);
    }
  };

  const filterKey = [query, mode, sort, searchParams.get("like") ?? "",
    searchParams.get("unlike") ?? "", JSON.stringify(selection.params)].join("|");

  /** How a ranking names itself in one line: the words, then the pictures it
   * was steered by. An exclusion-only search has no words at all, so without
   * this it would name itself nothing — the album it becomes and the row it
   * leaves in the trail both need the references spoken aloud. */
  const rankingLabel = () => [
    query.trim(),
    likeIds.length ? `like #${likeIds.join(" #")}` : "",
    unlikeIds.length ? `unlike #${unlikeIds.join(" #")}` : "",
  ].filter(Boolean).join(" · ");

  /** A committed search joins the same activity trail the album lifecycle
   * writes to. That trail is what the History drawer shows and what carries a
   * search past this browser's localStorage — until now the drawer could only
   * ever list tag approvals, because nothing wrote the search kinds it labels.
   * Unlike the local recency list, a written row cannot be taken back, so the
   * write waits for the typing to stop: each answered search cancels the
   * pending one, and only the query still on screen after the pause is kept.
   * Deduplicated by what the row would SAY, so re-ranking the same words —
   * another mode, another sort, the next page — never becomes a second row.
   * The query string travels with it, making a drawer row a link back into the
   * exact view (the URL is the shareable state). */
  const TRAIL_SETTLE_MS = 2500;
  const loggedLabel = useRef("");
  const trailTimer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(trailTimer.current), []);

  const recordSearchActivity = () => {
    const q = query.trim();
    const kind = composed ? "composed_search" : q ? "search_snapshot" : null;
    const label = rankingLabel();
    if (!kind || loggedLabel.current === label) return;
    const params = searchParams.toString();
    window.clearTimeout(trailTimer.current);
    trailTimer.current = window.setTimeout(() => {
      loggedLabel.current = label;
      api.recordActivity(kind, {
        // A plain search is its words; a composed one is its words AND its
        // reference pictures, so it carries the fuller label instead.
        ...(composed ? { label, like: likeIds, unlike: unlikeIds } : { q }),
        query_string: params,
      }).catch(() => {});  // a trail that fails to record must not fail the search
    }, TRAIL_SETTLE_MS);
  };

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
        if (composed) {
          // Every composed shape goes to the server, exclusion-only included \u2014
          // the backend answers that one with a ranking pushed away from the
          // excluded images and says so in `message`, rendered as the notice.
          try {
            const res = await api.composedSearch({
              text: query || undefined,
              positive_ids: likeIds, negative_ids: unlikeIds,
              top_k: PER_PAGE, offset: (p - 1) * PER_PAGE,
              ...(selection.params as object),
            }, ctrl.signal);
            setNotice(res.message ?? null);
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
        // Only a query that answered joins the history — a search that threw
        // never reaches this line.
        if (query.trim()) rememberSearch(query.trim());
        recordSearchActivity();
        setLoading(false);
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") return; // superseded
        setError(e instanceof Error ? e.message : String(e));
        // Drop the previous answer. Leaving it on screen was worse than an
        // empty grid, because the heading and the cards came from different
        // queries: a failed request once kept the earlier hybrid results,
        // relabelled them with the new mode, and left every card wearing a
        // score badge the failed path never produced. The error was visible
        // and still the page read as an answer.
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

  /** The whole ranking, kept: an album named after the query, from the ids on
   * screen. Capped at 200 — an album is a curated set a person will actually
   * revisit, not an export; the full fusion depth stays available as the
   * rail's csv/jsonl/json downloads. The name is editable later in the album
   * header, like any album's. */
  const [savingAlbum, setSavingAlbum] = useState(false);
  /** Album names are unique, so keeping the same ranking twice — a perfectly
   * reasonable thing to do after re-running a query — would otherwise collide
   * and surface the raw 409. The copy numbers itself instead. */
  const createAlbumForRanking = async (base: string) => {
    for (let n = 1; n <= 20; n++) {
      const name = n === 1 ? base : `${base} (${n})`;
      try {
        return { album: await api.createAlbum(name), name };
      } catch (e) {
        if (!(e instanceof Error && e.message.startsWith("409"))) throw e;
      }
    }
    throw new Error(`“${base}” already exists 20 times over — rename one first`);
  };

  const saveResultsAsAlbum = () => {
    const base = rankingLabel();
    const ids = items.slice(0, 200).map((s) => s.id);
    if (!base || ids.length === 0) return;
    setSavingAlbum(true);
    createAlbumForRanking(base)
      .then(({ album, name }) => api.addToAlbum(album.id, ids)
        .then((r) => ({ id: album.id, added: r.added, name })))
      .then(({ id, added, name }) => {
        albumsChanged();
        showToast(`Saved ${added} to “${name}”`);
        // Land inside the album, like every other path that makes one.
        setParams({ album: String(id), q: "", like: "", unlike: "", page: "" });
      })
      .catch((e) => showToast(e instanceof Error ? e.message : "Could not save album"))
      .finally(() => setSavingAlbum(false));
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
            role="combobox"
            aria-expanded={histOpen && historyItems.length > 0}
            aria-controls="search-history-list"
            aria-autocomplete="list"
            aria-activedescendant={histIdx >= 0 ? `search-hist-${histIdx}` : undefined}
            placeholder='Search images… e.g. "dog jumping into water", "crowded market at night"'
            value={input}
            onChange={(e) => { setInput(e.target.value); setHistOpen(true); }}
            onFocus={onSearchFocus}
            onBlur={() => { setHistOpen(false); setHistIdx(-1); }}
            onKeyDown={onSearchKeyDown}
          />
          {/* The image affordance lives where the query goes: dropping a
              picture on the results or pasting one is the primary path, and
              this small control both says so and offers the file picker. */}
          <button type="button" className="by-image-hint" disabled={imageBusy}
                  aria-label="Search by image"
                  title="Search by image — drop a picture anywhere on the results, paste a copied image, or click to choose a file"
                  onClick={() => fileRef.current?.click()}>
            {imageBusy
              ? <span className="busy-dot" aria-hidden="true" />
              : (
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"
                     strokeLinejoin="round" aria-hidden="true">
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <path d="m21 15-5-5L5 21" />
                </svg>
              )}
          </button>
          <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
                 onChange={(e) => {
                   const f = e.target.files?.[0];
                   if (f) runImageSearch(f);
                   e.target.value = "";   // the same file, picked twice, still fires
                 }} />
          {histOpen && historyItems.length > 0 && (
            <ul className="search-history" id="search-history-list" role="listbox"
                aria-label="Recent searches">
              {historyItems.map((s, i) => (
                <li key={s} id={`search-hist-${i}`} role="option"
                    aria-selected={i === histIdx}
                    className={i === histIdx ? "active" : ""}
                    // preventDefault keeps the input focused through the
                    // press, so blur cannot close the list before click lands.
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => applyHistory(s)}
                    onMouseEnter={() => setHistIdx(i)}>
                  {s}
                </li>
              ))}
            </ul>
          )}
        </div>
        <SearchSettings
          mode={mode} sort={sort} density={density}
          densities={Object.keys(DENSITY)} providerName={providerName}
          onMode={(m) => setParams({ mode: m, page: "" })}
          onSort={(s) => setParams({ sort: s, page: "" })}
          onDensity={setDensity}
        />
      </div>

      {/* The tray exists only while something is picked — a tray of zero
          would be furniture. Fixed to the viewport bottom so the running
          count and the album action stay in reach however deep the scroll,
          which is exactly when hand-picking happens. */}
      {picked.size > 0 && (
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
          {/* Clearing the set is also how the tray leaves: with no mode to
              exit, an emptied selection has nothing left to dismiss. */}
          <button className="ghost" onClick={() => setPicked(new Set())}>
            Clear
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
              agents that show their work — built for hunting the long tail:
              rare scenes, coverage gaps, and the captions that don&rsquo;t
              hold up.
            </p>
          </div>
          <div className="hero-flows">
            <button type="button" className="flow-card" onClick={() => searchRef.current?.focus()}>
              <span className="flow-no">01</span>
              <span className="flow-name">Find</span>
              <span className="flow-hint">Words, an image, or both — every score labelled.</span>
            </button>
            <button type="button" className="flow-card"
                    onClick={() => {
                      window.scrollTo({ top: 0 });
                      showToast("Hover any image and click its ✓ to start picking");
                    }}>
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
                         selected={picked.has(s.id)}
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
          {(query || composed) && items.length > 0 && (
            <>
              {" · "}
              <button className="link-btn" onClick={saveResultsAsAlbum} disabled={savingAlbum}
                      title={`Save the top ${Math.min(items.length, 200)} ranked results `
                             + "as an album named after this query — rename it in the album header"}>
                {savingAlbum ? "saving…" : "save as album"}
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
                     selected={picked.has(s.id)}
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
