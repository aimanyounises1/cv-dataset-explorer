import { useEffect, useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
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

interface Item { to: string; label: string; end?: boolean; hint: string }
interface Group { title: string; items: Item[] }

const GROUPS: Group[] = [
  {
    title: "Find",
    items: [
      { to: "/", label: "Browse", end: true,
        hint: "Search and filter the corpus" },
      { to: "/map", label: "Embedding map",
        hint: "Lasso a region of embedding space" },
    ],
  },
  {
    title: "Audit",
    items: [
      { to: "/quality", label: "Caption quality",
        hint: "Rank captions least supported by their image" },
    ],
  },
  {
    title: "Trust",
    items: [
      { to: "/stats", label: "Dataset profile",
        hint: "Composition, coverage, duplicates and leakage" },
      { to: "/eval", label: "Retrieval benchmark",
        hint: "The tool measuring its own search accuracy" },
    ],
  },
  {
    title: "Ask",
    items: [
      { to: "/chat", label: "Assistant",
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

export default function LeftRail() {
  const [overview, setOverview] = useState<StatsOverview | null>(null);
  const [chat, setChat] = useState<ChatStatus | null>(null);
  const [histOpen, setHistOpen] = useState(false);
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

      <div className="rail-groups">
        {GROUPS.map((g) => (
          <div className="rail-group" key={g.title}>
            <div className="eyebrow rail-group-title">{g.title}</div>
            {g.items.map((it) => (
              <NavLink key={it.to} to={it.to} end={it.end}
                       title={collapsed ? `${it.label} — ${it.hint}` : it.hint}
                       className={({ isActive }) =>
                         `rail-link${isActive ? " active" : ""}`}>
                {/* Both spans always render; CSS swaps them, so collapsing
                    never remounts the nav or loses keyboard focus. */}
                <span className="rail-link-label">{it.label}</span>
                <span className="rail-link-min" aria-hidden="true">{it.label[0]}</span>
              </NavLink>
            ))}
          </div>
        ))}

        {/* The library lives with navigation, not with the selection: a saved
            view is how you COME BACK, so it must be reachable when nothing is
            selected yet — which is exactly when the selection rail is absent.
            Saving captures the gallery's query string (the only route whose
            search params are a filter set); restoring navigates there. */}
        <div className="rail-group rail-library">
          <div className="eyebrow rail-group-title">Library</div>
          {/* Albums before saved views: an album is a destination you return
              to daily; a saved view is a filter recipe. Both are the library —
              where you come back to — so both live with navigation. */}
          <AlbumShelf />
          <SavedViews
            current={location.pathname === "/" ? location.search : ""}
            onRestore={(qs) => navigate(qs ? `/?${qs}` : "/")}
          />
        </div>

        <FilterPanel />

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

      <div className="rail-foot">
        <div className="rail-corpus">
          {overview ? `${overview.total_samples.toLocaleString()} images` : "Flickr8k"}
        </div>
        {/* The palette is the fast path for people who know what they want; the
            rail is the map for people who do not. Neither replaces the other, so
            the hint lives here rather than the palette growing a nav. */}
        <div className="rail-kbd"><kbd>⌘K</kbd> search anything</div>
        <button className="rail-history-btn" onClick={() => setHistOpen(true)}
                title="Recent searches and album changes">
          History
        </button>
        <ModelsCard overview={overview} chat={chat} />
      </div>

      {histOpen && <HistoryDrawer onClose={() => setHistOpen(false)} />}
    </nav>
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
  const enrichState = tagged === 0 ? "not run"
    : pct === 0 ? "<1% tagged" : `${pct}% tagged`;

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
        <span className="models-state">{enrichState}</span>
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
function HistoryDrawer({ onClose }: { onClose: () => void }) {
  const navigate = useNavigate();
  const [events, setEvents] = useState<ActivityEvent[] | null>(null);

  useEffect(() => {
    let live = true;
    api.listActivity(100)
      .then((ev) => { if (live) setEvents(ev); })
      .catch(() => { if (live) setEvents([]); });
    return () => { live = false; };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <>
      <div className="hist-backdrop" onClick={onClose} />
      <aside className="hist-drawer" role="dialog" aria-label="Activity history">
        <div className="hist-head">
          <span className="eyebrow">History</span>
          <button className="hist-close" onClick={onClose} title="Close (Esc)">
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
