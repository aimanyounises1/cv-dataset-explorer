import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import {
  AlbumSummary, SampleCard, ScenarioGroup, SearchMode, TermStat,
} from "../api/types";
import AxisLegend from "../components/AxisLegend";
import SearchSettings from "../components/SearchSettings";
import { albumsChanged } from "../components/AlbumShelf";
import AlbumHeader from "../components/AlbumHeader";
import ImageCard from "../components/ImageCard";
import ScoreDistribution, { COSINE_BASES } from "../components/ScoreDistribution";
import ScenarioGroups, {
  GROUP_MIN_RESULTS, GROUP_THUMBS, GroupsAnswer, readGroupsAnswer,
} from "../components/ScenarioGroups";
import SelectionTray from "../components/SelectionTray";
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
   * temporary until "Save as album" makes one durable. The whole answer is
   * kept — including the degraded flag and the server's own sentence — so a
   * grouping that could not be made says why instead of rendering as "no
   * groups", which is indistinguishable from "your results have no structure". */
  const [scenarios, setScenarios] = useState<GroupsAnswer | null>(null);
  const [scenBusy, setScenBusy] = useState(false);
  /* Ranked results and grouped exploration are two views of ONE result set,
   * not two pages: the switch is in the result bar, the ranking is the
   * default, and moving between them costs one click and no request. The view
   * is deliberately not in the URL — the shareable artifact of a group is the
   * `?ids=` slice it opens, and a proposal is not a place. */
  const [view, setView] = useState<"ranked" | "grouped">("ranked");
  const [groupThumbs, setGroupThumbs] = useState<Record<number, string>>({});
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
  /* The corpus's own similarity floor, read once per mount: `undefined` until
   * the answer arrives, `null` when the active index publishes none. No
   * constant stands in for it — a number nobody measured must not be drawn as
   * if somebody had. */
  const [simFloor, setSimFloor] = useState<number | null | undefined>(undefined);
  useEffect(() => {
    api.overview()
      .then((o) => setSimFloor(typeof o.sim_floor === "number" ? o.sim_floor : null))
      .catch(() => setSimFloor(null));
  }, []);
  /* Where the reader has put the dim line, or null for "wherever the floor
   * is". Deliberately not a URL param: this is how one person is reading one
   * result set, not a filter that changed which images are in it. */
  const [dimBelow, setDimBelow] = useState<number | null>(null);

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
  // Proposals die with the ranking they described — a group over yesterday's
  // result set, shown beside today's, would be a lie with a thumbnail on it.
  useEffect(() => {
    setImageQuery(null);
    setScenarios(null);
    setView("ranked");
  }, [filterKey, page]);

  /* A new ranking is a new set of scores, so the reader's line goes back to
   * the measured floor. "Load more" deliberately does not reset it: the same
   * ranking, read deeper, is still the same reading. */
  useEffect(() => { setDimBelow(null); }, [filterKey]);

  /** Show the grouped view, fetching the proposal once per result set. The
   * switch flips immediately — the view is the user's choice, not the
   * request's — and the panel below reports whichever state the answer is in:
   * grouping, grouped, refused with a reason, or unavailable. */
  const showGroups = () => {
    setView("grouped");
    if (scenarios !== null || scenBusy) return;
    setScenBusy(true);
    api.scenarioGroups({
      text: query || undefined,
      positive_ids: likeIds, negative_ids: unlikeIds,
      ...(selection.params as object),
    })
      .then((r) => setScenarios(readGroupsAnswer(r)))
      .catch((e) => setScenarios({
        groups: [], basis: "", degraded: true,
        // A capability that is absent names the thing that would enable it;
        // any other failure is reported verbatim. Neither is ever silence.
        message: e instanceof Error && e.message.startsWith("404")
          ? "Grouping is not available on this backend — POST /api/search/scenarios "
            + "is not mounted. Restart the API after updating it."
          : `Could not group these results — ${e instanceof Error ? e.message : String(e)}`,
      }))
      .finally(() => setScenBusy(false));
  };

  /* Faces for the group strips, in one request for the whole panel: a label
   * and a count describe a group, but only the pictures show whether the
   * grouping is any good. Ids the server already ranked, so nothing here
   * re-ranks — this is a thumbnail lookup. */
  useEffect(() => {
    const groups = scenarios?.groups ?? [];
    if (view !== "grouped" || groups.length === 0) return;
    const want = groups.flatMap((g) => g.sample_ids.slice(0, GROUP_THUMBS));
    const missing = [...new Set(want)].filter((id) => !(id in groupThumbs));
    if (missing.length === 0) return;
    const ctrl = new AbortController();
    api.listSamples({ ids: missing.join(","), per_page: missing.length }, ctrl.signal)
      .then((r) => setGroupThumbs((prev) => {
        const next = { ...prev };
        for (const s of r.items) next[s.id] = s.thumb_url;
        return next;
      }))
      .catch(() => { /* the strip falls back to placeholders */ });
    return () => ctrl.abort();
    // groupThumbs is a cache read inside, never a trigger: depending on it
    // would re-run this effect with every arriving thumbnail.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, scenarios]);

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

  /* Groups go through the same collision-safe creation as a saved ranking:
   * the same query proposes the same labels, so keeping one twice is ordinary
   * and must not surface the unique-name 409. */
  const saveGroupAsAlbum = (gr: ScenarioGroup) => {
    createAlbumForRanking(gr.label)
      .then(({ album, name }) => api.addToAlbum(album.id, gr.sample_ids)
        .then(() => ({ album, name })))
      .then(({ album, name }) => {
        albumsChanged();
        showToast(`Saved “${name}” (${gr.sample_ids.length})`);
        setParams({ album: String(album.id), q: "", like: "", unlike: "", page: "" });
      })
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

  /* The distribution is offered only where the number under it is a
   * similarity. Two identical scores are not a distribution, so a flat result
   * set gets no panel either. */
  const distScores = useMemo(
    () => (meta.basis && COSINE_BASES.has(meta.basis)
      ? items.map((s) => s.score).filter((v): v is number => typeof v === "number")
      : []),
    [items, meta.basis]);
  const showDist = distScores.length > 1
    && Math.max(...distScores) > Math.min(...distScores);
  /* The line: wherever the reader put it, else the measured floor when it
   * falls inside these scores, else the bottom of the range — which dims
   * nothing, because the corpus never said where the line belongs. */
  const dimLine = (() => {
    if (!showDist) return 0;
    const lo = Math.min(...distScores);
    const hi = Math.max(...distScores);
    const fallback = simFloor != null && simFloor > lo && simFloor < hi ? simFloor : lo;
    return Math.min(hi, Math.max(lo, dimBelow ?? fallback));
  })();


  return (
    <div ref={pageRef}
         onDragOver={(e) => { e.preventDefault(); }}
         onDrop={(e) => {
           const f = e.dataTransfer.files?.[0];
           if (f && f.type.startsWith("image/")) { e.preventDefault(); runImageSearch(f); }
         }}>
      {/* The hero used to carry this page's only h1. The workspace does not
          want a headline above its search field, but the document still needs
          a heading — so it keeps one for screen readers, the same way the
          sample and chat pages do. */}
      <h1 className="sr-only">Image gallery</h1>
      <div className="controls">
        <div className="search-box">
          <input
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

      {/* The picked set's own surface, which knows nothing about searching.
          It renders itself away when nothing is picked. */}
      <SelectionTray picked={picked} albums={albums}
                     albumName={albumName} onAlbumName={setAlbumName}
                     busy={albumBusy} onSave={saveAlbum}
                     onCompare={(a, b) => navigate(`/compare?a=${a}&b=${b}`)}
                     onClear={() => setPicked(new Set())} />


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
        {/* Two views of one result set, and which one you are in. The ranking
            is the default; grouping is a proposal over the same ids. */}
        <div className="result-bar-right">
          {(query || composed) && items.length > 0 && (
            <div className="view-switch" role="group" aria-label="Result view">
              <button type="button" aria-pressed={view === "ranked"}
                      className={view === "ranked" ? "on" : ""}
                      title="The ranking, in score order"
                      onClick={() => setView("ranked")}>
                Ranked
              </button>
              <button type="button" aria-pressed={view === "grouped"}
                      className={view === "grouped" ? "on" : ""}
                      disabled={items.length < GROUP_MIN_RESULTS}
                      title={items.length < GROUP_MIN_RESULTS
                        ? `Grouping needs at least ${GROUP_MIN_RESULTS} results — `
                          + `${items.length} here`
                        : "The same results, clustered into at most three "
                          + "explainable groups"}
                      onClick={showGroups}>
                {scenBusy ? "Grouping…" : "Grouped"}
              </button>
            </div>
          )}
          {/* The key for the sparkline every card carries. Only shown when the
              cards actually have axes — a legend for an absent encoding is
              noise, and axes are absent until `python -m app.analyze` has run. */}
          {items.some((s) => s.axes) && view === "ranked" && <AxisLegend />}
        </div>
      </div>

      {view === "grouped" && (
        <ScenarioGroups resultCount={items.length} hasMore={hasMore}
                        answer={scenarios} busy={scenBusy} thumbs={groupThumbs}
                        onBack={() => setView("ranked")}
                        onSaveGroup={saveGroupAsAlbum} />
      )}

      {/* Only where the score is a similarity. Hybrid results are fused by
          rank and keyword results are BM25 — a "0.42" drawn on either scale
          would be a number pretending to mean something. */}
      {view === "ranked" && showDist && (
        <ScoreDistribution scores={distScores} basis={meta.basis ?? ""}
                           floor={simFloor} threshold={dimLine}
                           onThreshold={setDimBelow} />
      )}

      {view === "ranked" && (
      <div className="grid"
           style={{ "--frame-min": DENSITY[density] } as React.CSSProperties}>
        {/* Each card links with the query, mode and score that put it here.
            `items` accumulates every page loaded so far, so the array index is
            already the rank within the whole result set: the first card on
            page 3 is rank 121, not rank 1. */}
        {items.map((s, i) => {
          const card = (
            <ImageCard key={s.id} sample={s} scoreBasis={meta.basis}
                       query={query} mode={mode} rank={i + 1}
                       selected={picked.has(s.id)}
                       onToggleSelect={togglePick} getDragIds={getDragIds}
                       onLike={(id) => editRef(id, "like", true)}
                       onExclude={(id) => editRef(id, "unlike", true)} />
          );
          // Below the line: greyed in place, never moved and never removed.
          // Rank order is the answer the ranking gave, so dimming must not
          // rewrite it. The one-cell `grid dim` wrapper is what carries the
          // existing similarity-floor dim rule (`.grid.dim .card`, which also
          // restores full opacity on hover) down to a single card without
          // touching its column: an auto-fill track is by construction under
          // two frames wide, so the nested grid resolves to exactly one column.
          if (!showDist || typeof s.score !== "number" || s.score >= dimLine) return card;
          return <div className="grid dim" key={s.id}>{card}</div>;
        })}
      </div>
      )}

      {loading && view === "ranked" && <div className="loading">Loading…</div>}
      {!loading && view === "ranked" && items.length === 0 && (
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
      {!hasMore && query && meta.depthReached && view === "ranked" && (
        <div className="meta-line" style={{ textAlign: "center", marginTop: 18 }}>
          End of the ranked results — this query ranked its top{" "}
          {meta.depthLimit?.toLocaleString()}. More images match; narrow the query
          or add a filter to bring them into range.
        </div>
      )}

      {hasMore && view === "ranked" && (
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
