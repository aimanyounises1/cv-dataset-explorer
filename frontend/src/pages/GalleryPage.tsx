import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ApiError, api, isStatus } from "../api/client";
import {
  AlbumSummary, SampleCard, ScenarioGroup, SearchMode, TermStat,
} from "../api/types";
import AxisLegend from "../components/AxisLegend";
import SearchSettings from "../components/SearchSettings";
import { ALBUMS_CHANGED, albumsChanged } from "../components/AlbumShelf";
import AlbumHeader from "../components/AlbumHeader";
import ErrorState from "../components/ErrorState";
import FindingsRow from "../components/FindingsRow";
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
import { useGridCursor } from "../hooks/useGridCursor";
import { usePickedSet } from "../hooks/usePickedSet";
import { useScrollRestore } from "../hooks/useScrollRestore";
import { useSearchHistory } from "../hooks/useSearchHistory";

interface SearchMeta {
  basis?: string | null;
  /** The mode the ANSWER was ranked by, as the server named it — not the mode
   * the URL asked for. The two differ whenever reference chips are set: the
   * request goes to /search/composed, which has no mode at all and replies
   * `mode_used: "composed"`. Reading the URL instead let the bar assert
   * "(keyword)" over a ranking keyword never touched, so a mode comparison
   * that never ran read as one where the modes agreed. A ranking now names
   * itself the same way a score never travels without its basis. */
  modeUsed?: string | null;
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
const SIGNAL_KEY = "cvde-grid-signal";
const LOUPE_KEY = "cvde-loupe";

/** All search/filter state — including pagination depth — lives in the URL:
 * shareable links, working back-button, and "Load more" depth survives
 * navigating to a sample and back. */
/* A rejected composed request: the page shows the validator's own sentence,
 * never the wire payload. `ApiError` has already read `detail` out of the
 * envelope, so all that is left is Pydantic's own prefix on a custom
 * validator's message, which is machinery rather than meaning. */
const composedProblem = (e: unknown): string => {
  const msg = e instanceof Error ? e.message : String(e);
  return msg.replace(/^Value error, /, "") || "The composed query was rejected";
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
  /* S, not M. This is a triage surface: the first question asked of 8,000
     images is "which of these is worth opening", and that is answered by seeing
     many at once. M shows 16 frames at 1440x900; S with no caption block shows
     48, measured. A returning user's own choice still wins. */
  const [density, setDensity] = useState<string>(
    () => localStorage.getItem(DENSITY_KEY) ?? "S");
  /* Which measurement the frame edges draw. Kept out of the URL deliberately,
     beside density: `useSelection` names the map's colour mode as the precedent
     — how a set is drawn is the reader's own lens, not part of the set, and a
     shared link should arrive showing the recipient's preference, not the
     sender's. Empty string = no edge. */
  const [signal, setSignal] = useState<string>(
    () => localStorage.getItem(SIGNAL_KEY) ?? "difficulty");
  /* The loupe is a mode you enter, not something hover summons. The rail is the
     third grid column and it is absent when empty, so a panel that appeared on
     hover would widen the rail, reflow the grid, and move the cards out from
     under the pointer that opened it. Off by default: the grid at full width is
     the scanning tool, and this trades two columns for the ability to judge
     without navigating. */
  const [loupe, setLoupe] = useState<boolean>(
    () => localStorage.getItem(LOUPE_KEY) === "1");
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
   * remain labels; converting one into an album is an explicit act elsewhere.
   * Its whole life — durability, the shift-extend anchor, the drag payload —
   * is `usePickedSet` (hooks/usePickedSet); the page only reads it and hands
   * it to the tray. */
  const { picked, setPicked, toggle: togglePick, dragIds: getDragIds }
    = usePickedSet(items);
  const [albumName, setAlbumName] = useState("");
  const [albumBusy, setAlbumBusy] = useState(false);
  const [albums, setAlbums] = useState<AlbumSummary[]>([]);
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
        .catch(() => {
          /* A thumbnail that cannot load leaves the chip as id text; the
           * reference still ranks, so there is nothing here to act on. */
        });
    });
    // refThumbs is a cache read, not a trigger: depending on it would re-run
    // this after every thumbnail that lands.
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
    // Mount-once on purpose: the handler only forwards the pasted file to
    // runImageSearch, whose own reads are setters and the api client — a fresh
    // subscription per render would change nothing but the listener count.
  }, []);

  useEffect(() => {
    try { localStorage.setItem(DENSITY_KEY, density); } catch { /* non-essential */ }
  }, [density]);

  useEffect(() => {
    try { localStorage.setItem(SIGNAL_KEY, signal); } catch { /* non-essential */ }
  }, [signal]);

  useEffect(() => {
    try { localStorage.setItem(LOUPE_KEY, loupe ? "1" : "0"); } catch { /* non-essential */ }
  }, [loupe]);

  // Publishes the card under the cursor to the rail. Attached to both grids.
  const { gridProps, available: loupeAvailable } = useGridCursor(loupe);

  // Scroll restore for back-nav, keyed by the full query string. The hook owns
  // the whole of it, including the `isConnected` guard the repo's known-traps
  // note depends on.
  const pageRef = useRef<HTMLDivElement | null>(null);
  useScrollRestore(pageRef, `cvde-scroll:${searchParams.toString()}`,
                   !loading && items.length > 0);

  // Lets the fetch effect see the current list without depending on it.
  const itemsRef = useRef<SampleCard[]>([]);
  useEffect(() => { itemsRef.current = items; }, [items]);

  // Publish the result order for keyboard triage on the sample page.
  useEffect(() => { saveResultOrder(items.map((s) => s.id)); }, [items]);

  /* Mirrors `useSelection.set`, including its history rule: a discrete choice
   * (a mode, a sort, an album) pushes so Back undoes it, while controls that
   * emit a stream — the debounced query, "load more" — pass replace. Every
   * write used to replace, which left the first Back press exiting the app. */
  const setParams = (
    updates: Record<string, string>,
    options?: { replace?: boolean },
  ) => {
    setSearchParams((prev) => {
      const p = new URLSearchParams(prev);
      for (const [k, v] of Object.entries(updates)) {
        if (v) p.set(k, v);
        else p.delete(k);
      }
      return p;
    }, { replace: options?.replace ?? false });
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
    // Starting a search pushes, so Back returns to the unsearched view.
    // Refining one replaces, because the box commits every 400 ms and
    // "dog", "dog on", "dog on beach" are one search being typed, not three.
    if (debouncedInput.trim() !== query) {
      setParams({ q: debouncedInput.trim(), page: "" }, { replace: Boolean(query) });
    }
    // Runs only when the debounce settles. Depending on `query` too would fire
    // this again when the URL echoes the trimmed value back mid-typing — the
    // exact "dogon beach" failure the sync effect above guards against.
  }, [debouncedInput]);

  /* The recent-queries dropdown. Choosing an entry writes the URL immediately —
   * a chosen entry shouldn't wait out the debounce — which is the one thing the
   * hook cannot know how to do, so the page passes it in. */
  const hist = useSearchHistory(input, (q) => {
    setInput(q);
    setParams({ q, page: "" });
  });

  /* An album's membership can change while its grid is on screen — the member
   * strip removes an image, a drop adds one — and the grid is a view of that
   * membership, so it has to re-read it. The album mutations already announce
   * themselves on the shared event for the shelf's benefit; the gallery now
   * listens too, and only re-fetches when an album is what it is showing. */
  const [albumTick, setAlbumTick] = useState(0);
  useEffect(() => {
    if (!albumId) return;
    const onChanged = () => setAlbumTick((n) => n + 1);
    window.addEventListener(ALBUMS_CHANGED, onChanged);
    return () => window.removeEventListener(ALBUMS_CHANGED, onChanged);
  }, [albumId]);

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
            setMeta({ basis: res.score_basis, modeUsed: res.mode_used,
                      rrfK: null, terms: [],
                      depthLimit: res.depth_limit, depthReached: res.depth_reached });
            return { items: res.items, total: null as number | null, more: res.has_more };
          } catch (e) {
            if (isStatus(e, 404)) {
              // The endpoint ships in a parallel lane: degrade honestly to the
              // text ranking rather than a blank page, and say so.
              fallbackMsg = "Composed search is not available on this backend yet — "
                          + "showing the unsteered ranking; reference chips are kept.";
            } else if (e instanceof ApiError && e.status >= 400 && e.status < 500) {
              // A rejected request is explained in a sentence, never rendered
              // as the wire payload, and the page still shows a ranking.
              fallbackMsg = composedProblem(e)
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
          setMeta({ basis: res.score_basis, modeUsed: res.mode_used,
                    rrfK: res.rrf_k, terms: res.term_stats ?? [],
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
        if (query.trim()) hist.remember(query.trim());
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
    // Everything this fetch reads (query, mode, selection params, sort) is
    // serialized into filterKey — the key IS the dependency list, so listing
    // the parts as well would only duplicate it less reliably.
  }, [filterKey, page, albumTick]);

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
        message: isStatus(e, 404)
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
        if (!isStatus(e, 409)) throw e;
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

  /** What the bar, and the grouping panel's own caveat, are allowed to call
   * this ranking: the server's word for the answer it actually gave, falling
   * back to the requested mode only before the first answer arrives. */
  const modeUsed = meta.modeUsed ?? mode;

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
   * nothing, because the corpus never said where the line belongs.
   *
   * The floor may only seed a RAW-cosine ranking. `cosine_adj` is that cosine
   * minus a query-bank hubness penalty, so the corpus's image-to-image floor
   * measures a different quantity and would start the line at a number that
   * means nothing on this axis. cos* therefore opens at its own minimum, which
   * dims nothing until the reader moves it. */
  const dimLine = (() => {
    if (!showDist) return 0;
    const lo = Math.min(...distScores);
    const hi = Math.max(...distScores);
    const floorSeeds = meta.basis === "cosine"
      && simFloor != null && simFloor > lo && simFloor < hi;
    return Math.min(hi, Math.max(lo, dimBelow ?? (floorSeeds ? (simFloor as number) : lo)));
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
            aria-expanded={hist.open && hist.items.length > 0}
            aria-controls="search-history-list"
            aria-autocomplete="list"
            aria-activedescendant={
              hist.activeIndex >= 0 ? `search-hist-${hist.activeIndex}` : undefined}
            placeholder='Search images… e.g. "dog jumping into water", "crowded market at night"'
            value={input}
            onChange={(e) => { setInput(e.target.value); hist.show(); }}
            onFocus={hist.onFocus}
            onBlur={hist.close}
            onKeyDown={hist.onKeyDown}
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
          {hist.open && hist.items.length > 0 && (
            <ul className="search-history" id="search-history-list" role="listbox"
                aria-label="Recent searches">
              {hist.items.map((s, i) => (
                <li key={s} id={`search-hist-${i}`} role="option"
                    aria-selected={i === hist.activeIndex}
                    className={i === hist.activeIndex ? "active" : ""}
                    // preventDefault keeps the input focused through the
                    // press, so blur cannot close the list before click lands.
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => hist.pick(s)}
                    onMouseEnter={() => hist.highlight(i)}>
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
          modeLockedReason={composed
            ? "Reference chips steer this ranking (composed basis) — clear them to change mode"
            : null}
        />
      </div>

      {/* The picked set's own surface, which knows nothing about searching.
          It renders itself away when nothing is picked. */}
      <SelectionTray picked={picked} albums={albums}
                     albumName={albumName} onAlbumName={setAlbumName}
                     busy={albumBusy} onSave={saveAlbum}
                     onCompare={(a, b) => navigate(`/compare?a=${a}&b=${b}`)}
                     onClear={() => {
                       // Clear is the one remaining way to lose a set that now
                       // survives navigation, and it asks nothing before doing
                       // it. The restore is a copy of client state, so unlike
                       // the album Undo there is nothing to prove and no writer
                       // to race — it either comes back exactly, or the toast
                       // is never offered.
                       const prev = picked;
                       setPicked(new Set());
                       if (prev.size) {
                         showToast(`Cleared ${prev.size} picked`,
                                   { label: "Undo", run: () => setPicked(prev) });
                       }
                     }} />


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
            {/* The mode pills stay clickable while chips are set and the URL
                still records the click, but the request goes to composed
                retrieval, which has no mode — so the choice changes nothing.
                Unsaid, a researcher can read "semantic and keyword agree
                perfectly" off a comparison that never ran. (Disabling the
                three buttons belongs in SearchSettings, not here.) */}
            {" · "}
            <span title={"Composed retrieval ranks by these reference images "
                       + "(score basis “composed”). Search settings → Mode applies to "
                       + "text-only searches; clear the references to compare hybrid, "
                       + "semantic and keyword on the same query."}>
              search-mode dial does not apply while references are set
            </span>
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
      {!query && !selection.active && <FindingsRow />}

      {notice && <div className="notice">{notice}</div>}
      {error && <ErrorState error={error} />}

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
          <div className="grid" data-signal={signal || undefined} {...gridProps}
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
            ? `${items.length}${hasMore ? "+" : ""} result${items.length === 1 ? "" : "s"} for “${query}” (${modeUsed})`
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
          {items.some((s) => s.axes) && view === "ranked" && (
            <AxisLegend signal={signal} onSignal={setSignal} />
          )}
          {/* A mode, because it costs the rail's 300px and therefore two grid
              columns. Making that trade on hover would reflow the grid out from
              under the pointer that asked for it. Absent below the three-column
              breakpoint, where the rail stacks under the grid and the trade is
              not one worth offering. */}
          {loupeAvailable && <button type="button" className={`loupe-toggle${loupe ? " on" : ""}`}
                  aria-pressed={loupe} onClick={() => setLoupe((v) => !v)}
                  title={loupe
                    ? "Close the inspector and give the grid its columns back"
                    : "Show the image under the cursor at full size, with its "
                      + "five captions and their measured agreement, without "
                      + "leaving the grid"}>
            Inspector
          </button>}
        </div>
      </div>

      {view === "grouped" && (
        <ScenarioGroups resultCount={items.length} hasMore={hasMore}
                        mode={modeUsed} answer={scenarios} busy={scenBusy}
                        thumbs={groupThumbs}
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
      <div className="grid" data-signal={signal || undefined} {...gridProps}
           style={{ "--frame-min": DENSITY[density] } as React.CSSProperties}>
        {/* Each card links with the query, mode and score that put it here.
            `items` accumulates every page loaded so far, so the array index is
            already the rank within the whole result set: the first card on
            page 3 is rank 121, not rank 1.

            The mode is passed only when the answer was ranked by the mode the
            URL asked for. Otherwise it is omitted, and the sample page's
            provenance banner reads "Surfaced by search “…” — composed 0.765"
            rather than naming a mode that never ran — the same lie the result
            bar used to tell, one navigation further on. Omitting is right and
            substituting is not: ProvenanceBanner voids itself on a mode it
            does not recognise, and the score's own basis already travels. */}
        {items.map((s, i) => {
          const card = (
            <ImageCard key={s.id} sample={s} scoreBasis={meta.basis}
                       query={query} mode={modeUsed === mode ? mode : undefined}
                       rank={sort ? undefined : i + 1}
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
          {/* Replaces: deepening the same ranking is not a new view, and a
              reader who pressed it five times should not press Back five
              times to leave. The depth still rides in the URL. */}
          <button className="primary"
                  onClick={() => setParams({ page: String(page + 1) }, { replace: true })}
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
