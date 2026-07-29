import { Fragment, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import { createPortal } from "react-dom";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { api } from "../../api/client";
import type { ActivityEvent, ChatStatus, StatsOverview } from "../../api/types";
import FilterPanel from "./FilterPanel";
import { useSelection } from "../../hooks/useSelection";
import AlbumShelf from "../AlbumShelf";
import SavedViews from "../SavedViews";

/**
 * Navigation grouped by the job you came to do, not by the table behind it.
 *
 * The seven routes were a flat list of peers, which made a first-time reader
 * ask "is Quality a kind of Statistics?" — a question the old top bar had no way
 * to answer. Grouping says what each destination is *for*.
 *
 * **Every path is unchanged.** Only labels and grouping move, so saved views
 * (which store bare query strings), bookmarks and shared links all keep working.
 * `/samples/:id` is deliberately absent: it is a child of Browse, reached by
 * clicking a frame, not a destination you navigate to cold.
 */

/** A view of a destination: same page, different addressable slice. */
interface View { to: string; label: string; end?: boolean }
/** `keycap` is what the collapsed strip shows in place of the label. It is
 * written per item rather than sliced off `label[0]`, because a substring
 * cannot know that "Compare two" and "Caption quality" both start with C —
 * which is exactly what the strip used to render, twice, in two different
 * groups, with the difference living only in a hover tooltip. The keycap is a
 * named product decision like the label beside it. */
interface Item { to: string; label: string; keycap: string; end?: boolean;
                 hint: string; views?: View[] }
interface Group { title: string; items: Item[] }

/** The two routes that read the same selection: the gallery and the map both
 * call useSelection and filter on the same params. Moving between them used to
 * drop the filter on the floor — you would narrow the corpus to 1,595 images,
 * click "Embedding map", and land on all 8,000 with nothing saying why. So the
 * selection travels, but only between these two, and only when leaving one of
 * them: carrying a gallery's filter onto the benchmark or the assistant would
 * be noise, and carrying it out of a sample page would resurrect a filter the
 * reader had already left behind.
 *
 * `page` is dropped because it belongs to one view's pagination, not to the
 * set — the map has no pages, and coming back to a page that no longer exists
 * is how a filter change lands on an empty grid. */
const SELECTION_ROUTES = new Set(["/", "/map"]);

function railTo(to: string, here: string, search: string): string {
  if (!search || !SELECTION_ROUTES.has(to) || !SELECTION_ROUTES.has(here)) return to;
  const params = new URLSearchParams(search);
  params.delete("page");
  const qs = params.toString();
  return qs ? `${to}?${qs}` : to;
}

const GROUPS: Group[] = [
  {
    title: "Find",
    items: [
      { to: "/", label: "Browse", keycap: "B", end: true,
        hint: "Search and filter the corpus" },
      { to: "/map", label: "Embedding map", keycap: "E",
        hint: "Lasso a region of embedding space" },
      // Reachable by name, not only by picking two cards. It is the only place
      // a region can be saved as an annotation, and a destination you can get
      // to solely by a side effect of another screen is one most people never
      // find. It opens empty and says how to fill itself.
      { to: "/compare", label: "Compare two", keycap: "Cmp",
        hint: "Two frames under one loupe: synced zoom, shared/different, regions" },
    ],
  },
  {
    title: "Audit",
    items: [
      { to: "/quality", label: "Caption quality", keycap: "Q",
        hint: "Rank captions least supported by their image" },
    ],
  },
  {
    title: "Trust",
    items: [
      { to: "/stats", label: "Dataset profile", keycap: "D",
        hint: "Composition, coverage, duplicates and leakage",
        /* The profile's five views belong to the rail, not to a second left
           column beside it: two stacked navigations on one edge read as one
           broken navigation. They appear indented under the profile while it
           is the open section, so the rail shows where you are without
           carrying five more rows everywhere else. */
        views: [
          { to: "/stats", label: "Overview", end: true },
          { to: "/stats?view=integrity", label: "Split integrity" },
          { to: "/stats?view=coverage", label: "Prompt slices" },
          { to: "/stats?view=captions", label: "Caption health" },
          { to: "/stats?view=provenance", label: "Provenance" },
        ] },
      { to: "/eval", label: "Retrieval benchmark", keycap: "R",
        hint: "The tool measuring its own search accuracy" },
    ],
  },
  {
    title: "Ask",
    items: [
      { to: "/chat", label: "Assistant", keycap: "A",
        hint: "Multi-agent answers rendered as live charts" },
    ],
  },
];

/** Fetched once for the whole session; the rail is on every route and this must
 * not become a request per navigation. Failure is silent — a missing corpus
 * count is not worth an error state in the furniture. The same response feeds
 * the models card below, so status costs no extra request. */
let overviewPromise: Promise<StatsOverview | null> | null = null;
function overviewOnce() {
  if (!overviewPromise) {
    overviewPromise = api.overview().catch(() => null);
  }
  return overviewPromise;
}

/** Same once-per-session rule for the assistant probe: Ollama starting or
 * stopping is a reload-the-tab event, not a navigation event. */
let chatStatusPromise: Promise<ChatStatus | null> | null = null;
function chatStatusOnce() {
  if (!chatStatusPromise) {
    chatStatusPromise = api.chatStatus().catch(() => null);
  }
  return chatStatusPromise;
}

/** "org/name" → "name": the card cites the model; the hub prefix is
 * addressing, not identity. */
function shortModel(name?: string | null): string | null {
  if (!name) return null;
  const tail = name.split("/").pop()?.trim();
  return tail || null;
}

/** Retrieval counts as active unless the server said keyword-only
 * (`embed_provider: null`), the older flag says no embeddings, or the index
 * for the active provider is not on disk. */
function retrievalActive(ov: StatsOverview | null): boolean {
  if (!ov) return false;
  if (ov.embed_provider === null) return false;
  if (ov.embed_provider === undefined && !ov.embeddings_available) return false;
  return ov.embed_index_ready !== false;
}

const RAIL_KEY = "cvde-rail";
/** The width at which the app stops being three columns — it mirrors the
 *  `@media (max-width: 1100px)` block in `index.css` that collapses `.app` to a
 *  single column and stacks both rails. Exported because anything whose value
 *  depends on there *being* a side column has to agree with that block, and a
 *  second literal would be a second breakpoint waiting to drift. */
export const COMPACT_RAIL_QUERY = "(max-width: 1100px)";

/** WCAG 2.2 SC 2.5.8 puts the floor for a pointer target at 24×24 CSS px and
 * SC 2.5.5 puts the comfortable target at 44×44 — the same number Apple's HIG
 * uses, and roughly the contact patch of a fingertip. The two dismiss buttons
 * this file renders inherit `.hist-close`, measured at 21×20; the strip's
 * History button inherits `.rail-history-btn`, measured at 61×21. Both are
 * under the floor, so the rail asks for the target on the controls it owns. */
const TARGET_MIN_PX = 44;

/** One geometry for "close this dialog", so the drawer and the tools surface
 * cannot disagree about how big a dismiss target is. */
const DISMISS_TARGET = {
  minWidth: TARGET_MIN_PX,
  minHeight: TARGET_MIN_PX,
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
} as const;

/** Everything a modal must put out of reach. The two dialogs sit in different
 * places in the tree — the tools surface is a child of the rail, the history
 * drawer is portalled to <body> beside it — so the background is named as a
 * flat list of the shell's regions and filtered against whichever dialog is
 * open, rather than assuming one fixed shape. */
const DIALOG_BACKGROUND = [
  ".pane",
  ".rail-r",
  ".rail-brand",
  ".rail-mobile-tools-btn",
  ".rail-groups",
  ".rail-foot",
  ".rail-tools-surface",
  ".cmdk-mobile-trigger",
].join(",");

const DIALOG_FOCUSABLE = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

/**
 * The modal contract, written once for both dialogs this rail owns.
 *
 * Saying `role="dialog"` promises five things: focus starts inside, Tab cannot
 * leave, Escape closes, the page behind is inert and does not scroll, and
 * focus returns where it came from. The tools surface kept that promise and
 * the history drawer did not — Tab walked straight into the gallery it was
 * covering — which is worse than not claiming the role at all, because a
 * screen reader announces a modal and then hands you the page underneath.
 * The behaviour lives here so the two cannot drift apart again.
 *
 * The opener is read from the document rather than passed in: whatever had
 * focus when the dialog opened is the honest place to land on close, and
 * neither caller has to name its own trigger.
 */
function useDialogFocus(
  dialogRef: RefObject<HTMLElement | null>,
  open: boolean,
  onClose: () => void,
) {
  /* Held in a ref so a parent re-render cannot re-run the effect below and
     re-take focus mid-interaction: the effect's subject is the dialog, not the
     identity of the callback. */
  const closeRef = useRef(onClose);
  useEffect(() => { closeRef.current = onClose; });

  useEffect(() => {
    const dialog = open ? dialogRef.current : null;
    if (!dialog) return undefined;

    const focusable = () => Array.from(
      dialog.querySelectorAll<HTMLElement>(DIALOG_FOCUSABLE),
    ).filter((element) => element.getClientRects().length > 0);

    const background = Array.from(
      document.querySelectorAll<HTMLElement>(DIALOG_BACKGROUND),
    ).filter((element) => !dialog.contains(element) && !element.contains(dialog));
    const priorInert = background.map((element) => element.hasAttribute("inert"));
    background.forEach((element) => element.setAttribute("inert", ""));

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const opener = document.activeElement as HTMLElement | null;
    const focusFrame = requestAnimationFrame(() => focusable()[0]?.focus());

    const containFocus = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (event.key === "Escape") {
        event.preventDefault();
        closeRef.current();
        return;
      }
      if (event.key !== "Tab") return;

      const items = focusable();
      if (items.length === 0) {
        event.preventDefault();
        dialog.focus();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", containFocus);

    return () => {
      cancelAnimationFrame(focusFrame);
      window.removeEventListener("keydown", containFocus);
      background.forEach((element, index) => {
        if (!priorInert[index]) element.removeAttribute("inert");
      });
      document.body.style.overflow = previousOverflow;
      /* Only back to something still on screen: growing past the compact
         breakpoint closes the tools surface by hiding its own trigger, and
         focusing a display:none button drops focus onto <body>. */
      if (opener?.isConnected && opener.getClientRects().length > 0) {
        opener.focus();
      }
    };
  }, [dialogRef, open]);
}

function compactRailMatches(): boolean {
  return typeof window !== "undefined"
    && typeof window.matchMedia === "function"
    && window.matchMedia(COMPACT_RAIL_QUERY).matches;
}

export default function LeftRail() {
  const [overview, setOverview] = useState<StatsOverview | null>(null);
  const [chat, setChat] = useState<ChatStatus | null>(null);
  const [histOpen, setHistOpen] = useState(false);
  const [mobileToolsOpen, setMobileToolsOpen] = useState(false);
  const [compactRail, setCompactRail] = useState(compactRailMatches);
  const toolsTriggerRef = useRef<HTMLButtonElement>(null);
  const toolsDialogRef = useRef<HTMLElement>(null);
  const location = useLocation();
  const navigate = useNavigate();
  // Read-only: the chip count on the collapsed strip's filter key, so
  // collapsing the rail never silently disowns an active filter set.
  const sel = useSelection();
  /* Collapsed is workspace geometry, not shareable state — localStorage like
   * the gallery density, never the URL. The state is mirrored onto the root
   * element so the app grid's column width follows without prop-drilling
   * through App. */
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(RAIL_KEY) === "min");
  useEffect(() => {
    try { localStorage.setItem(RAIL_KEY, collapsed ? "min" : "full"); }
    catch { /* non-essential */ }
    if (collapsed) document.documentElement.dataset.rail = "min";
    else delete document.documentElement.dataset.rail;
  }, [collapsed]);

  useEffect(() => {
    let live = true;
    overviewOnce().then((o) => { if (live && o) setOverview(o); });
    chatStatusOnce().then((c) => { if (live && c) setChat(c); });
    return () => { live = false; };
  }, []);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return undefined;
    const media = window.matchMedia(COMPACT_RAIL_QUERY);
    const onChange = (event: MediaQueryListEvent) => {
      setCompactRail(event.matches);
      if (!event.matches) {
        setMobileToolsOpen(false);
        requestAnimationFrame(() => {
          document.querySelector<HTMLElement>(".rail-link.active")?.focus();
        });
      }
    };
    setCompactRail(media.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  /* Closing is only a state change: the hook restores focus to whatever
     opened the surface, so the trigger is not named twice. */
  const closeMobileTools = () => setMobileToolsOpen(false);
  useDialogFocus(toolsDialogRef, compactRail && mobileToolsOpen, closeMobileTools);

  const restoreView = (query: string) => {
    navigate(query ? `/?${query}` : "/");
  };

  return (
    <nav className={`rail-l${collapsed ? " collapsed" : ""}`} aria-label="Sections">
      <div className="rail-brand">
        <span className="brand-mark" aria-hidden="true">◈</span>
        <span className="rail-brand-name">CV Dataset Explorer</span>
        <button className="rail-toggle" aria-expanded={!collapsed}
                title={collapsed ? "Expand the library rail" : "Collapse the library rail"}
                onClick={() => setCollapsed(!collapsed)}>
          {collapsed ? "»" : "«"}
        </button>
      </div>

      <button
        className="rail-mobile-tools-btn"
        type="button"
        ref={toolsTriggerRef}
        aria-expanded={mobileToolsOpen}
        aria-controls="rail-mobile-tools"
        onClick={() => setMobileToolsOpen(true)}
      >
        Filters{sel.chips.length > 0 ? ` · ${sel.chips.length}` : ""}
      </button>

      <div className="rail-scroll">
        <div className="rail-groups">
          {GROUPS.map((g) => (
            <div className="rail-group" key={g.title}>
              <div className="eyebrow rail-group-title">{g.title}</div>
              {g.items.map((it) => (
                <Fragment key={it.to}>
                  <NavLink to={railTo(it.to, location.pathname, location.search)}
                           end={it.end}
                           title={collapsed ? `${it.label} — ${it.hint}` : it.hint}
                           /* Collapsed, the only rendered text is an
                              aria-hidden keycap and the label is display:none
                              — so without this the link has no accessible name
                              at all. Expanded, the label speaks for itself. */
                           aria-label={collapsed ? it.label : undefined}
                           className={({ isActive }) =>
                             `rail-link${isActive ? " active" : ""}`}>
                    {/* Both spans always render; CSS swaps them, so collapsing
                        never remounts the nav or loses keyboard focus. */}
                    <span className="rail-link-label">{it.label}</span>
                    <span className="rail-link-min" aria-hidden="true">{it.keycap}</span>
                  </NavLink>
                  {/* A destination's own views, shown only while you are there —
                      and never in the collapsed strip, which has room for one
                      letter per row. */}
                  {it.views && !collapsed && location.pathname === it.to && (
                    <div className="rail-views">
                      {it.views.map((v) => {
                        const active = location.pathname + location.search === v.to
                          || (v.end && location.pathname === it.to && !location.search);
                        return (
                          <Link key={v.to} to={v.to}
                                aria-current={active ? "page" : undefined}
                                className={`rail-view${active ? " active" : ""}`}>
                            {v.label}
                          </Link>
                        );
                      })}
                    </div>
                  )}
                </Fragment>
              ))}
            </div>
          ))}

          {/* The collapsed strip keeps one way back into filters and the
              library. The count is the active-chip count — an amber badge says
              "this strip is hiding live constraints", which is exactly the state
              that must never be silent. */}
          {collapsed && location.pathname === "/" && (
            <button className="rail-filter-key" onClick={() => setCollapsed(false)}
                    title={"Filters and library — expand the rail"
                           + (sel.chips.length ? ` (${sel.chips.length} active)` : "")}>
              ≡
              {sel.chips.length > 0 && (
                <span className="rail-filter-count">{sel.chips.length}</span>
              )}
            </button>
          )}

          {/* Collapsing hides the whole footer (`.rail-l.collapsed .rail-foot`
              is display:none) and History went with it — the one state that
              removed an entry point instead
              of narrowing it, which teaches that collapsing is lossy in ways
              you only find by looking. It comes back wearing the strip's
              existing keycap chrome rather than a second one, and a glyph
              instead of a letter says it opens a drawer rather than routing. */}
          {collapsed && (
            <button className="rail-filter-key" onClick={() => setHistOpen(true)}
                    aria-label="History: recent searches and album changes"
                    title="History — recent searches and album changes">
              ⟲
            </button>
          )}

          {/* The models card folded to one dot: the collapsed strip hides the
              whole footer, but "retrieval is degraded" must never be one of the
              things collapsing hides. Expanding is the way to read the detail. */}
          {collapsed && (
            <button className="rail-models-key" onClick={() => setCollapsed(false)}
                    title={"Models — retrieval "
                           + (retrievalActive(overview) ? "active" : "keyword only")
                           + ", assistant "
                           + (chat?.available ? "ready" : "unavailable")
                           + ". Expand the rail for detail."}>
              <span className={"models-dot "
                               + (retrievalActive(overview) && chat?.available
                                  ? "ok" : "warn")}
                    aria-hidden="true" />
            </button>
          )}
        </div>

        {/* One mounted copy serves both layouts. At compact widths this same
            stateful tree becomes a modal surface; it is never duplicated behind
            the drawer, so saved views and API-backed filters cannot diverge. */}
        <section
          id="rail-mobile-tools"
          ref={toolsDialogRef}
          className={`rail-tools-surface${mobileToolsOpen ? " open" : ""}`}
          role={compactRail && mobileToolsOpen ? "dialog" : undefined}
          aria-modal={compactRail && mobileToolsOpen ? "true" : undefined}
          aria-label={compactRail && mobileToolsOpen ? "Filters and library" : undefined}
          tabIndex={compactRail && mobileToolsOpen ? -1 : undefined}
        >
          <header className="rail-tools-head">
            <div>
              <div className="eyebrow">Browse tools</div>
              <strong>Filters and library</strong>
            </div>
            <button
              className="hist-close"
              type="button"
              style={DISMISS_TARGET}
              aria-label="Close filters and library"
              onClick={() => closeMobileTools()}
            >
              ×
            </button>
          </header>
          <div className="rail-tools-body">
            {/* The library lives with navigation, not with the selection: a
                saved view is how you come back. Restoring from the compact
                surface also closes it; desktop restoration keeps the rail. */}
            <LibraryAndFilters
              current={location.pathname === "/" ? location.search : ""}
              onRestore={(query) => {
                restoreView(query);
                if (compactRail) closeMobileTools();
              }}
            />
          </div>
        </section>
      </div>

      <div className="rail-foot">
        <div className="rail-corpus">
          {overview ? `${overview.total_samples.toLocaleString()} images` : "Flickr8k"}
        </div>
        {/* The palette is the fast path for people who know what they want; the
            rail is the map for people who do not. Neither replaces the other, so
            the hint lives here rather than the palette growing a nav. */}
        <div className="rail-kbd"><kbd>⌘K</kbd> search anything</div>
        {/* On the phone strip this pill is the only way back into a search you
            already ran (there is no keyboard for ⌘K), and its stylesheet size
            is 61×21 — a thumb target three times too short. It grows only
            where it is touched; the desktop rail keeps its quiet footer. */}
        <button className="rail-history-btn" onClick={() => setHistOpen(true)}
                style={compactRail ? { minHeight: TARGET_MIN_PX } : undefined}
                title="Recent searches and album changes">
          History
        </button>
        <ModelsCard overview={overview} chat={chat} />
      </div>

      {/* Portalled to <body>: the rail is a stacking context (sticky on
          desktop, masked on the phone strip), so as a child of this <nav> the
          drawer's z-index only ranked it against its rail siblings — the page
          painted over it and the backdrop never dimmed anything. */}
      {histOpen && createPortal(
        <HistoryDrawer onClose={() => setHistOpen(false)} />, document.body)}
      {compactRail && mobileToolsOpen && createPortal(
        <div
          className="rail-tools-backdrop"
          onClick={(event) => {
            if (event.target === event.currentTarget) closeMobileTools();
          }}
          aria-hidden="true"
        />,
        document.body,
      )}
    </nav>
  );
}

function LibraryAndFilters({
  current,
  onRestore,
}: {
  current: string;
  onRestore: (query: string) => void;
}) {
  return (
    <>
      <div className="rail-group rail-library">
        <div className="eyebrow rail-group-title">Library</div>
        <AlbumShelf />
        <SavedViews current={current} onRestore={onRestore} />
      </div>
      <FilterPanel />
    </>
  );
}

/** Which models are doing the thinking, in the register of the corpus count
 * above it: status is furniture until it is bad news, and bad news arrives as
 * amber plus a collapsed "why", never as layout. Every provenance field is
 * optional — a backend predating them yields a quieter card, not a broken
 * one, because this card is itself a QA surface. */
function ModelsCard({ overview: ov, chat }:
                    { overview: StatsOverview | null; chat: ChatStatus | null }) {
  const keywordOnly = !!ov && (ov.embed_provider === null
    || (ov.embed_provider === undefined && !ov.embeddings_available));
  const fallback = !!(ov?.embed_provider && ov.embed_preferred
    && ov.embed_provider !== ov.embed_preferred);
  const retrievalName = keywordOnly ? "keyword only"
    : shortModel(ov?.embed_model) ?? (ov ? "semantic index" : "…");
  const retrievalOk = retrievalActive(ov);
  const retrievalWhy = ov?.embed_fallback_reason ?? null;

  const assistantOk = chat?.available === true;
  const assistantName = ov?.chat_model ?? chat?.model ?? "…";

  const total = ov?.total_samples ?? 0;
  const tagged = ov?.vlm_enriched ?? 0;
  const pct = total > 0 ? Math.round((100 * tagged) / total) : 0;
  // vlm_ready = is the model pulled NOW; vlm_enriched = past work. A corpus
  // can be tagged while the model is gone, or untagged while it waits.
  const vlmReady = ov?.vlm_ready === true;
  const enrichState = tagged === 0
    ? (ov && !vlmReady ? "model not pulled" : "not run")
    : pct === 0 ? "<1% tagged" : `${pct}% tagged`;
  const enrichHint = ov && !vlmReady
    ? `ollama pull ${ov.vlm_model ?? "qwen2.5vl:7b"} — nothing downloads without you`
    : undefined;

  return (
    <div className="models-card">
      <div className="models-row">
        <span className="models-role">Retrieval</span>
        <span className="models-name" title={ov?.embed_model ?? undefined}>
          {retrievalName}
        </span>
        {fallback
          ? <span className="models-chip">fallback</span>
          : <span className={`models-dot ${retrievalOk ? "ok" : "warn"}`}
                  aria-hidden="true" />}
      </div>
      {(fallback || keywordOnly) && retrievalWhy && <ModelsWhy text={retrievalWhy} />}
      <div className="models-row">
        <span className="models-role">Assistant</span>
        <span className="models-name">{assistantName}</span>
        <span className={`models-dot ${assistantOk ? "ok" : "warn"}`}
              aria-hidden="true" />
      </div>
      {!assistantOk && chat?.reason && <ModelsWhy text={chat.reason} />}
      <div className="models-row">
        <span className="models-role">Enrichment</span>
        <span className="models-name" title={ov?.vlm_model ?? undefined}>
          {shortModel(ov?.vlm_model) ?? "VLM tags"}
        </span>
        <span className="models-state" title={enrichHint}>{enrichState}</span>
      </div>
      {/* Verbatim by requirement: the tool's one-sentence privacy statement. */}
      <p className="models-foot">
        {"All models local. Ollama serves the language models only — image embeddings are computed in-process."}
      </p>
    </div>
  );
}

/** The reason is developer-authored plain text (an enabling command, a path).
 * It waits behind a collapsed disclosure: the row states the state; the why
 * is for the person who intends to change it. */
function ModelsWhy({ text }: { text: string }) {
  return (
    <details className="models-why">
      <summary>why</summary>
      <p>{text}</p>
    </details>
  );
}

function kindLabel(kind: string): string {
  if (kind === "search_snapshot") return "Search";
  if (kind === "image_search") return "Image search";
  if (kind === "composed_search") return "Composed search";
  if (kind.startsWith("album_")) return "Album";
  return kind.replace(/_/g, " ");
}

/** Best human line the opaque payload offers: a query text or a name, else a
 * `q` recovered from a stored query string. Empty is allowed — the kind label
 * and timestamp still make the row a record. */
function eventSummary(payload: ActivityEvent["payload"]): string {
  for (const key of ["q", "query", "text", "name", "label"]) {
    const v = payload[key];
    if (typeof v === "string" && v.trim()) return v.trim();
  }
  const qs = payload["query_string"];
  if (typeof qs === "string" && qs) {
    const q = new URLSearchParams(qs.replace(/^\?/, "")).get("q");
    if (q?.trim()) return q.trim();
  }
  return "";
}

function relTime(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  const d = Math.floor(s / 86400);
  return d === 1 ? "yesterday" : `${d}d ago`;
}

/** The activity trail as a glance, not a destination: fetched on every open
 * rather than cached, because activity is precisely the data that changed
 * since you last looked — reopening IS the refresh. Escape and the backdrop
 * both close; a glance must cost nothing to leave. Rows that stored a
 * query string travel (the URL is the state doctrine's shareable artifact);
 * rows without one are the record only. */
const HIST_TITLE_ID = "hist-drawer-title";

function HistoryDrawer({ onClose }: { onClose: () => void }) {
  const navigate = useNavigate();
  const [events, setEvents] = useState<ActivityEvent[] | null>(null);
  const drawerRef = useRef<HTMLElement>(null);

  useEffect(() => {
    let live = true;
    api.listActivity(100)
      .then((ev) => { if (live) setEvents(ev); })
      .catch(() => { if (live) setEvents([]); });
    return () => { live = false; };
  }, []);

  /* The drawer exists only while open, so the contract is on from mount:
     focus lands inside, Tab stays inside, Escape closes and hands focus back
     to the History button. */
  useDialogFocus(drawerRef, true, onClose);

  return (
    <>
      <div className="hist-backdrop" onClick={onClose} />
      <aside className="hist-drawer" ref={drawerRef} role="dialog"
             aria-modal="true" aria-labelledby={HIST_TITLE_ID} tabIndex={-1}>
        <div className="hist-head">
          {/* A real heading, not a styled span: it is what a screen reader
              reads on entering the dialog and the only landmark inside it.
              `.eyebrow` carries the small-caps; the flush margin keeps it
              sitting in the head's flex row exactly where the span did. */}
          <h2 id={HIST_TITLE_ID} className="eyebrow" style={{ margin: 0 }}>
            History
          </h2>
          <button className="hist-close" onClick={onClose} style={DISMISS_TARGET}
                  aria-label="Close history" title="Close (Esc)">
            ×
          </button>
        </div>
        <div className="hist-list">
          {events === null && <p className="hist-empty">Loading…</p>}
          {events?.length === 0 && (
            <p className="hist-empty">
              No activity yet — searches and album changes will appear here.
            </p>
          )}
          {events?.map((e) => {
            const raw = e.payload["query_string"];
            const qs = typeof raw === "string" && raw.trim()
              ? raw.replace(/^\?/, "") : null;
            const body = (
              <>
                <span className="hist-kind">{kindLabel(e.kind)}</span>
                <span className="hist-summary">{eventSummary(e.payload) || "—"}</span>
                <span className="hist-time">{relTime(e.created_at)}</span>
              </>
            );
            return qs ? (
              <button key={e.id} className="hist-row link"
                      onClick={() => { navigate(`/?${qs}`); onClose(); }}>
                {body}
              </button>
            ) : (
              <div key={e.id} className="hist-row">{body}</div>
            );
          })}
        </div>
      </aside>
    </>
  );
}
