import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { api } from "../api/client";

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

/** The recent-queries dropdown: a local recency list merged with the workspace
 * trail's search snapshots, narrowed by whatever is typed. Both sources are
 * reads — the hook never writes the URL. Choosing an entry calls `onPick`,
 * which is the page's business: this hook does not know what a query does.
 * Returns a closed interface (no setters) so the only ways the list can move
 * are the gestures it defines. */
export function useSearchHistory(input: string, onPick: (q: string) => void) {
  const [localHist, setLocalHist] = useState<string[]>(readLocalHistory);
  const [serverHist, setServerHist] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const fetched = useRef(false);

  /** Local recency first (it is this person's own trail), then the workspace
   * snapshots; deduplicated case-insensitively; narrowed by whatever is
   * already typed, minus the exact query already on screen. */
  const items = useMemo(() => {
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
  useEffect(() => { setActiveIndex(-1); }, [items.length]);

  const show = () => setOpen(true);
  const close = () => { setOpen(false); setActiveIndex(-1); };
  const highlight = (i: number) => setActiveIndex(i);
  const pick = (q: string) => { close(); onPick(q); };

  const onFocus = () => {
    setOpen(true);
    setActiveIndex(-1);
    if (fetched.current) return;
    fetched.current = true;
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
    }).catch(() => {
      /* Optional enrichment, deliberately silent: the dropdown still serves
       * local recency, and there is nothing for the reader to act on — the
       * same degradation contract as the findings cards. */
    });
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown" && !open && items.length > 0) {
      e.preventDefault();
      setOpen(true);
      setActiveIndex(0);
      return;
    }
    if (!open || items.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => (i + 1) % items.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => (i <= 0 ? items.length - 1 : i - 1));
    } else if (e.key === "Enter" && activeIndex >= 0) {
      e.preventDefault();
      pick(items[activeIndex]);
    } else if (e.key === "Escape") {
      close();
    }
  };

  /** Typing is not a sequence of searches. The box commits every 400 ms, so
   * "dog on beach" arrives as "dog", "dog on", "dog on beach" — each one a real
   * ranking, none of them a thing anyone meant to keep. A new entry therefore
   * absorbs the prefixes it grew out of, and only the phrase the person stopped
   * on survives in the list. */
  const remember = (q: string) => {
    const key = q.toLowerCase();
    const kept = readLocalHistory()
      .filter((s) => s.toLowerCase() !== key && !key.startsWith(s.toLowerCase()));
    const list = [q, ...kept].slice(0, HISTORY_KEPT);
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list)); }
    catch { /* non-essential */ }
    setLocalHist(list);
  };

  return { items, open, activeIndex, show, close, highlight, pick,
           onFocus, onKeyDown, remember };
}
