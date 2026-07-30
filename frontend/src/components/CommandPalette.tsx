import { ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { AttributeGroup, TagInfo } from "../api/types";
import { useHotkey } from "../hooks/useHotkey";
import { buildExportParams } from "../hooks/useSelection";

type NavigateFn = ReturnType<typeof useNavigate>;

interface SavedView {
  name: string;
  query_string: string;
  created_at: string;
}

// ---------------------------------------------------------------- fetching --
// Tag/attribute/saved-view lists are fetched once and shared across every
// open of the palette (the component itself lives for the app's lifetime, so
// this is really just "don't refetch on every keystroke or reopen"). Module
// scope on top of that means a second mounted instance — unlikely, but not
// this file's business to assume — would not re-fetch either.

interface SuggestionData {
  tags: TagInfo[];
  vlmTags: TagInfo[];
  attrGroups: AttributeGroup[];
}

let suggestionPromise: Promise<SuggestionData> | null = null;

function loadSuggestions(): Promise<SuggestionData> {
  if (!suggestionPromise) {
    suggestionPromise = Promise.all([api.tags(), api.vlmTags(), api.coverage()])
      .then(([tags, vlmTags, attrGroups]) => ({ tags, vlmTags, attrGroups }))
      .catch((err: unknown) => {
        suggestionPromise = null; // let the next open retry
        throw err;
      });
  }
  return suggestionPromise;
}

// Written by three call sites that must agree: the option element's id, the
// input's aria-activedescendant, and the scroll-into-view lookup.
const optionId = (index: number) => `cmdk-option-${index}`;

let savedViewsPromise: Promise<SavedView[] | null> | null = null; // resolves null when the router is absent (404)

function loadSavedViews(): Promise<SavedView[] | null> {
  if (!savedViewsPromise) {
    savedViewsPromise = fetch("/api/views")
      .then((res) => {
        if (res.status === 404) return null;
        if (!res.ok) throw new Error(String(res.status));
        return res.json() as Promise<SavedView[]>;
      })
      .catch((err: unknown) => {
        savedViewsPromise = null;
        throw err;
      });
  }
  return savedViewsPromise;
}

// ------------------------------------------------------------ fuzzy match ---

interface FuzzyMatch {
  score: number;
  indices: number[];
}

/** Characters after which a match counts as starting a new "word", for the
 * word-start tier below. */
const WORD_BOUNDARY = /[\s/_\-:.,()"'|]/;

function sequentialIndices(start: number, end: number): number[] {
  return Array.from({ length: end - start }, (_, i) => start + i);
}

/**
 * Subsequence fuzzy match with three ranking tiers, encoded as non-overlapping
 * score bands so a single numeric sort produces prefix > word-start >
 * subsequence: whole-string prefix (3000s), match starting right after a
 * word-boundary character (2000s), and a plain in-order subsequence
 * elsewhere (bounded well under 2000, with fewer gaps and an earlier start
 * scoring higher — so a contiguous substring naturally outranks a scattered
 * one within that tier). Returns null when `query` is not a subsequence of
 * `text` at all.
 */
function fuzzyMatch(query: string, text: string): FuzzyMatch | null {
  const q = query.trim().toLowerCase();
  if (!q) return { score: 0, indices: [] };
  const t = text.toLowerCase();

  if (t.startsWith(q)) {
    return { score: 3000 - t.length, indices: sequentialIndices(0, q.length) };
  }

  for (let i = 1; i <= t.length - q.length; i++) {
    if (WORD_BOUNDARY.test(t[i - 1]) && t.slice(i, i + q.length) === q) {
      return { score: 2000 - i - t.length * 0.1, indices: sequentialIndices(i, i + q.length) };
    }
  }

  const indices: number[] = [];
  let cursor = 0;
  for (const ch of q) {
    const idx = t.indexOf(ch, cursor);
    if (idx === -1) return null;
    indices.push(idx);
    cursor = idx + 1;
  }
  const span = indices[indices.length - 1] - indices[0] + 1;
  const gaps = span - q.length;
  return { score: 1000 - gaps * 5 - indices[0] - t.length * 0.05, indices };
}

/** Renders `text` with the characters at `indices` wrapped in the app's
 * existing `mark` style, grouping consecutive indices into one run each. */
function renderHighlighted(text: string, indices: number[]): ReactNode {
  if (indices.length === 0) return text;
  const sorted = [...indices].sort((a, b) => a - b);
  const parts: ReactNode[] = [];
  let cursor = 0;
  let i = 0;
  while (i < sorted.length) {
    const runStart = sorted[i];
    let runEnd = runStart + 1;
    while (i + 1 < sorted.length && sorted[i + 1] === runEnd) {
      runEnd++;
      i++;
    }
    if (runStart > cursor) parts.push(text.slice(cursor, runStart));
    parts.push(<mark key={runStart}>{text.slice(runStart, runEnd)}</mark>);
    cursor = runEnd;
    i++;
  }
  if (cursor < text.length) parts.push(text.slice(cursor));
  return parts;
}

// -------------------------------------------------------------- item model --

interface PaletteItem {
  id: string;
  label: string;
  indices: number[];
  sublabel?: string;
  hint: string;
  score: number;
  perform: () => void;
}

interface PaletteGroup {
  name: string;
  items: PaletteItem[];
  total: number;
}

const GROUP = {
  NAV: "Navigation",
  SAMPLES: "Samples",
  TAGS: "Tags",
  VLM: "VLM tags",
  ATTR: "Attribute slices",
  VIEWS: "Saved views",
  ACTIONS: "Actions",
} as const;

const CAP: Record<string, number> = {
  [GROUP.NAV]: 8,
  [GROUP.SAMPLES]: 2,
  [GROUP.TAGS]: 6,
  [GROUP.VLM]: 6,
  [GROUP.ATTR]: 6,
  [GROUP.VIEWS]: 5,
  [GROUP.ACTIONS]: 2,
};

const ROUTES: { label: string; path: string; keywords: string }[] = [
  { label: "Gallery", path: "/", keywords: "home browse grid contact sheet search" },
  { label: "Map", path: "/map", keywords: "scatter embedding cluster projection" },
  { label: "Statistics", path: "/stats", keywords: "stats overview charts captions histogram" },
  { label: "Quality", path: "/quality", keywords: "qa agreement suspect captions consistency" },
  { label: "Benchmark", path: "/eval", keywords: "eval retrieval recall mrr evaluation" },
  { label: "Assistant", path: "/chat", keywords: "chat agent llm ask" },
  { label: "Compare", path: "/compare", keywords: "compare loupe side by side regions diff" },
];

function shortDate(iso: string): string {
  const t = Date.parse(iso);
  return Number.isNaN(t) ? "" : new Date(t).toLocaleDateString();
}

/** Jump to the gallery with `patch` applied. Merges into the current URL when
 * already on the gallery (so a second filter from the palette narrows what's
 * on screen rather than discarding it); jumps in fresh from anywhere else. */
function goToFilteredGallery(
  navigate: NavigateFn, pathname: string, search: string,
  patch: Record<string, string>, append = false,
): void {
  const params = pathname === "/" ? new URLSearchParams(search) : new URLSearchParams();
  for (const [k, v] of Object.entries(patch)) {
    if (append && !params.getAll(k).includes(v)) params.append(k, v);
    else if (!append) params.set(k, v);
  }
  params.delete("page");
  navigate({ pathname: "/", search: `?${params.toString()}` });
}

function downloadCurrentView(search: string): void {
  // Every route, not just the gallery. The filter vocabulary lives in the URL
  // wherever you are — `/map?split=test` narrows the same corpus the gallery
  // does — so starting from an empty set off `/` made "Export current view"
  // hand back all 8,000 rows while the rail beside it said 1,001. (The same
  // ternary in `goToFilteredGallery` above is deliberate and stays: *navigating*
  // in from elsewhere should arrive fresh rather than inherit. Exporting
  // describes where you already are.) Serialization lives in
  // `buildExportParams` — the same builder the rail's export pills use, so the
  // two paths cannot drift.
  const a = document.createElement("a");
  a.href = `/api/export?${buildExportParams(search, "csv").toString()}`;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** A bare id resolves without a network round trip; a filename does not, so
 * it is looked up against `/api/samples?ids=` (which matches ids *or*
 * filenames) only when the row is actually activated — never per keystroke. */
async function resolveFilename(filename: string, navigate: NavigateFn): Promise<void> {
  try {
    const res = await api.listSamples({ ids: filename, per_page: 1 });
    if (res.items.length === 1) {
      navigate(`/samples/${res.items[0].id}`);
      return;
    }
  } catch {
    // Fall through to the filtered gallery, which shows its own empty state.
  }
  navigate(`/?ids=${encodeURIComponent(filename)}`);
}

/** Matches a route's label first (so the highlight lines up with what's shown);
 * a route whose label doesn't match but whose keywords do still appears, just
 * without highlighting and at a discounted score. */
function matchRoute(query: string, r: { label: string; keywords: string }): FuzzyMatch | null {
  // fuzzyMatch("", …) always matches (score 0), so an empty query short-circuits
  // here and every route is shown — the keyword fallback below only matters
  // once there is text that the label itself didn't match.
  const labelMatch = fuzzyMatch(query, r.label);
  if (labelMatch) return labelMatch;
  const kwMatch = fuzzyMatch(query, r.keywords);
  return kwMatch ? { score: kwMatch.score - 500, indices: [] } : null;
}

export default function CommandPalette(): JSX.Element | null {
  const navigate = useNavigate();
  const location = useLocation();
  const { pathname, search } = location;

  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);

  const [suggestions, setSuggestions] = useState<SuggestionData | null>(null);
  const [suggestionsFailed, setSuggestionsFailed] = useState(false);
  const [savedViews, setSavedViews] = useState<SavedView[] | null | undefined>(undefined);
  // Distinct from `savedViews === null`, which means the views router is absent
  // (404) and the group is correctly hidden. A 500 or a dropped connection is a
  // different fact and must not render as "you have no saved views" — a silently
  // empty group is indistinguishable from a working one with nothing in it.
  const [savedViewsFailed, setSavedViewsFailed] = useState(false);

  const previouslyFocused = useRef<HTMLElement | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const closePalette = useCallback(() => {
    setOpen(false);
    setQuery("");
    requestAnimationFrame(() => previouslyFocused.current?.focus());
  }, []);

  const toggle = useCallback(() => {
    setOpen((o) => {
      const next = !o;
      if (next) {
        previouslyFocused.current = document.activeElement as HTMLElement | null;
      } else {
        setQuery("");
        requestAnimationFrame(() => previouslyFocused.current?.focus());
      }
      return next;
    });
  }, []);

  // The only global hotkey this component owns. It fires even while focus is
  // inside some other <input> or <textarea> — that is deliberate: Cmd/Ctrl+K
  // is a modifier chord no text field types as a character, so intercepting
  // it everywhere is what makes the palette "global" without ever eating a
  // letter a user is actually typing. No other key is bound globally; once
  // open, Arrow/Enter/Escape/Tab are handled by onKeyDown on the dialog
  // itself (ordinary bubbling from the focused search input), scoped to the
  // dialog only.
  useHotkey(
    (e) => !e.repeat && !e.altKey && e.key.toLowerCase() === "k" && (e.metaKey || e.ctrlKey),
    toggle,
  );

  useEffect(() => {
    if (!open) return;
    if (suggestions === null && !suggestionsFailed) {
      loadSuggestions().then(setSuggestions).catch(() => setSuggestionsFailed(true));
    }
    if (savedViews === undefined && !savedViewsFailed) {
      loadSavedViews()
        .then(setSavedViews)
        .catch(() => { setSavedViews(null); setSavedViewsFailed(true); });
    }
  }, [open, suggestions, suggestionsFailed, savedViews, savedViewsFailed]);

  useEffect(() => {
    if (!open) return undefined;
    const t = setTimeout(() => inputRef.current?.focus(), 0);
    return () => clearTimeout(t);
  }, [open]);

  useEffect(() => {
    setActiveIndex(0);
  }, [query, open]);

  const groups = useMemo<PaletteGroup[]>(() => {
    const q = query.trim();

    const nav: PaletteItem[] = ROUTES.map((r) => ({ r, m: matchRoute(query, r) }))
      .filter((x): x is { r: typeof ROUTES[number]; m: FuzzyMatch } => x.m !== null)
      .map(({ r, m }) => ({
        id: `nav:${r.path}`, label: r.label, indices: m.indices, hint: "Go to page",
        score: m.score,
        perform: () => { closePalette(); navigate(r.path); },
      }));

    const samples: PaletteItem[] = [];
    if (/^\d+$/.test(q)) {
      samples.push({
        id: "sample:id", label: `Open sample #${q}`, indices: [], hint: "Sample id",
        score: 5000, perform: () => { closePalette(); navigate(`/samples/${q}`); },
      });
    } else if (/^[\w-]+\.[a-z0-9]{2,5}$/i.test(q)) {
      samples.push({
        id: "sample:filename", label: `Open sample "${q}"`, indices: [], hint: "By filename",
        score: 5000, perform: () => { closePalette(); void resolveFilename(q, navigate); },
      });
    }

    const tagLike = (list: TagInfo[] | null, groupName: string, param: "tag" | "vlm_tag"): PaletteItem[] => {
      if (!list) return [];
      const candidates: { t: TagInfo; m: FuzzyMatch }[] = q
        ? list
            .map((t) => ({ t, m: fuzzyMatch(q, t.name) }))
            .filter((x): x is { t: TagInfo; m: FuzzyMatch } => x.m !== null)
        : [...list].sort((a, b) => b.count - a.count).slice(0, 5).map((t) => ({ t, m: { score: 0, indices: [] } }));
      return candidates.map(({ t, m }) => ({
        id: `${param}:${t.name}`, label: t.name, sublabel: `${t.count.toLocaleString()} images`,
        indices: m.indices, hint: groupName === GROUP.TAGS ? "Filter by tag" : "Filter by VLM tag",
        score: m.score,
        perform: () => { closePalette(); goToFilteredGallery(navigate, pathname, search, { [param]: t.name }); },
      }));
    };
    const tags = tagLike(suggestions?.tags ?? null, GROUP.TAGS, "tag");
    const vlm = tagLike(suggestions?.vlmTags ?? null, GROUP.VLM, "vlm_tag");

    const flatAttrs = (suggestions?.attrGroups ?? []).flatMap((g) =>
      g.labels.map((l) => ({ grp: g.grp, label: l.label, count: l.count })));
    const attrCandidates = q
      ? flatAttrs
          .map((f) => ({ f, m: fuzzyMatch(q, `${f.grp.replace(/_/g, " ")}: ${f.label}`) }))
          .filter((x): x is { f: typeof flatAttrs[number]; m: FuzzyMatch } => x.m !== null)
      : [...flatAttrs].sort((a, b) => b.count - a.count).slice(0, 5)
          .map((f) => ({ f, m: { score: 0, indices: [] } }));
    const attrs: PaletteItem[] = attrCandidates.map(({ f, m }) => ({
      id: `attr:${f.grp}:${f.label}`, label: `${f.grp.replace(/_/g, " ")}: ${f.label}`,
      sublabel: `${f.count.toLocaleString()} images`, indices: m.indices, hint: "Filter by attribute",
      score: m.score,
      perform: () => {
        closePalette();
        goToFilteredGallery(navigate, pathname, search,
                            { attr: `${f.grp}:${f.label}` }, true);
      },
    }));

    const viewList = savedViews ?? [];
    const viewCandidates = q
      ? viewList
          .map((v) => ({ v, m: fuzzyMatch(q, v.name) }))
          .filter((x): x is { v: SavedView; m: FuzzyMatch } => x.m !== null)
      : [...viewList].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)).slice(0, 5)
          .map((v) => ({ v, m: { score: 0, indices: [] } }));
    const views: PaletteItem[] = viewCandidates.map(({ v, m }) => ({
      id: `view:${v.name}`, label: v.name, sublabel: shortDate(v.created_at),
      indices: m.indices, hint: "Restore saved view", score: m.score,
      perform: () => {
        closePalette();
        navigate({ pathname: "/", search: v.query_string ? `?${v.query_string}` : "" });
      },
    }));

    const actions: PaletteItem[] = [];
    if (q) {
      actions.push({
        id: "action:search", label: `Search the dataset for "${q}"`, indices: [], hint: "Search",
        score: 4000,
        perform: () => { closePalette(); navigate({ pathname: "/", search: `?q=${encodeURIComponent(q)}` }); },
      });
    }
    actions.push({
      id: "action:export", label: "Export current view (CSV)", indices: [], hint: "Download",
      score: 3900,
      perform: () => { closePalette(); downloadCurrentView(search); },
    });

    const order: [string, PaletteItem[]][] = [
      [GROUP.NAV, nav], [GROUP.SAMPLES, samples], [GROUP.TAGS, tags], [GROUP.VLM, vlm],
      [GROUP.ATTR, attrs], [GROUP.VIEWS, views], [GROUP.ACTIONS, actions],
    ];
    return order
      .map(([name, items]) => {
        const sorted = [...items].sort((a, b) => b.score - a.score);
        return { name, items: sorted.slice(0, CAP[name]), total: sorted.length };
      })
      .filter((g) => g.items.length > 0);
  }, [query, suggestions, savedViews, navigate, pathname, search, closePalette]);

  const flatItems = useMemo(() => groups.flatMap((g) => g.items), [groups]);

  useEffect(() => {
    if (activeIndex > flatItems.length - 1) setActiveIndex(Math.max(0, flatItems.length - 1));
  }, [flatItems.length, activeIndex]);

  // The list scrolls, so the keyboard's idea of "active" has to stay the eye's
  // idea of it: aria-activedescendant would otherwise announce a row that a
  // sighted user cannot see, and Enter would fire it blind. "nearest" scrolls
  // only when the row is actually out of view, which is what keeps mouse-hover
  // selection (onMouseEnter, below) from yanking the list under the cursor.
  // Re-run on the result set too, not just the index: a query change resets the
  // index to 0, and when it was already 0 that is not a change this could see.
  useEffect(() => {
    document.getElementById(optionId(activeIndex))?.scrollIntoView({ block: "nearest" });
  }, [activeIndex, flatItems]);

  const moveActive = (delta: number) => {
    setActiveIndex((i) => {
      if (flatItems.length === 0) return 0;
      return (i + delta + flatItems.length) % flatItems.length;
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        moveActive(1);
        break;
      case "ArrowUp":
        e.preventDefault();
        moveActive(-1);
        break;
      case "Enter":
        e.preventDefault();
        flatItems[activeIndex]?.perform();
        break;
      case "Escape":
        e.preventDefault();
        closePalette();
        break;
      case "Tab":
        // The dialog holds exactly one focusable control (the search input),
        // so trapping Tab here is enough to keep focus from ever escaping.
        e.preventDefault();
        break;
      default:
        break;
    }
  };

  const loadingSuggestions = suggestions === null && !suggestionsFailed;

  return (
    <>
      <button
        className="cmdk-mobile-trigger"
        type="button"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls="cmdk-dialog"
        onClick={toggle}
      >
        Search
      </button>
      {open && (
        <div className="cmdk-backdrop" onClick={(e) => { if (e.target === e.currentTarget) closePalette(); }}>
          <div
            id="cmdk-dialog"
            className="cmdk-dialog"
            role="dialog"
            aria-modal="true"
            aria-label="Command palette"
            onKeyDown={handleKeyDown}
          >
        <div className="cmdk-input-row">
          <span className="cmdk-input-icon" aria-hidden="true">⌘K</span>
          <input
            ref={inputRef}
            className="cmdk-input"
            type="text"
            role="combobox"
            aria-expanded="true"
            aria-controls="cmdk-listbox"
            aria-haspopup="listbox"
            aria-autocomplete="list"
            aria-activedescendant={flatItems.length > 0 ? optionId(activeIndex) : undefined}
            aria-label="Search commands, samples, tags, and attributes"
            placeholder="Search or jump to…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>

        {/* Named individually rather than as one blanket "suggestions failed":
            which group is missing tells the reader whether the thing they came
            here for is gone, or something unrelated is. Both subjects are
            plural, so the verb never changes. */}
        {(suggestionsFailed || savedViewsFailed) && (
          <div className="notice cmdk-notice">
            {[suggestionsFailed && "Tag and attribute suggestions",
              savedViewsFailed && "saved views"]
              .filter(Boolean)
              .join(" and ")
              .replace(/^s/, "S")}{" "}
            are unavailable right now — navigation and search still work.
          </div>
        )}
        {loadingSuggestions && <div className="loading cmdk-loading">Loading suggestions…</div>}

        <div className="cmdk-list" id="cmdk-listbox" role="listbox" aria-label="Command palette results">
          {flatItems.length === 0 ? (
            <div className="empty cmdk-empty">
              No matches for "{query}". Try a different word, or press Enter to search the dataset for it.
            </div>
          ) : (
            groups.map((g) => (
              <div key={g.name} className="cmdk-group">
                <div className="eyebrow cmdk-group-label">
                  {g.name}
                  {g.total > g.items.length && (
                    <span className="pill cmdk-group-count">{g.items.length} of {g.total}</span>
                  )}
                </div>
                {g.items.map((item) => {
                  const idx = flatItems.indexOf(item);
                  return (
                    <div
                      key={item.id}
                      id={optionId(idx)}
                      role="option"
                      aria-selected={idx === activeIndex}
                      className={idx === activeIndex ? "cmdk-item active" : "cmdk-item"}
                      onMouseEnter={() => setActiveIndex(idx)}
                      onClick={() => item.perform()}
                    >
                      <span className="cmdk-item-label">{renderHighlighted(item.label, item.indices)}</span>
                      {item.sublabel && <span className="cmdk-item-sub mono">{item.sublabel}</span>}
                      <span className="cmdk-item-hint">{item.hint}</span>
                    </div>
                  );
                })}
              </div>
            ))
          )}
        </div>

        <div className="cmdk-footer">
          <span><span className="pill">↑↓</span>navigate</span>
          <span><span className="pill">↵</span>select</span>
          <span><span className="pill">esc</span>close</span>
        </div>
          </div>
        </div>
      )}
    </>
  );
}
