import { useEffect, useRef, useState } from "react";
import { SampleCard } from "../api/types";

/* Three concerns that used to sit loose in the gallery page body — the picked
 * set, scroll restore, and the search-history dropdown — are hooks, each owning
 * one thing end to end, beside the ones the repo already had (useSelection,
 * useResultOrder). Nothing about their behaviour changed; what changed is that
 * each is now readable whole. The page was 1193 lines holding ten unrelated
 * concerns, which is how a result bar came to read its search mode from the URL
 * 950 lines away from the branch that made the request. */

/** The hand-picked set, per tab. sessionStorage, not local: a basket is a
 * session's work, and a new window should start empty. */
const PICKED_KEY = "cvde-picked";
/** The card a shift-extend measures from. Stored beside the set because the
 * two are one thought: the set now survives navigation, and an anchor that did
 * not would make the first shift-click after coming back behave as a plain
 * toggle — the feature silently absent exactly when a large set is being
 * rebuilt. Kept as its own key rather than folded into the set, because the
 * failure modes differ: a stale anchor is harmless (an id absent from the
 * current ranking already degrades to a plain toggle) while a corrupt set is
 * not. */
const PICK_ANCHOR_KEY = "cvde-pick-anchor";

/** The picked set plus the two gestures that build it, over whichever ranking
 * is on screen (the range a shift-extend covers is measured in the order the
 * ranking gave, so the current items are the hook's one input).
 *
 * Kept in sessionStorage, on one tab-wide key rather than one per query.
 * Hand-picking is the most expensive thing a person does here, and it used
 * to be the least durable: `picked` was component state, the gallery is its
 * own route, so opening a card or pressing the tray's own Compare unmounted
 * it and took the whole set. The QA sweep had encoded that loss as a
 * workaround — it re-picked two cards after going back, guarded by "if the
 * tray is gone". Two picks cost two clicks to rebuild; two hundred are
 * simply gone.
 *
 * Tab-wide and not keyed by query, because curating across several searches
 * into one album is the point — a set that reset on every filter change
 * would be a set you could not assemble. This is the same class as scroll
 * position and the benchmark result: ephemeral working state the URL
 * deliberately does not own, because a pasted link should reproduce the
 * view, not somebody else's half-finished basket. Nothing is hidden by it:
 * the tray is visible whenever the set is non-empty. */
export function usePickedSet(items: SampleCard[]) {
  const [picked, setPicked] = useState<Set<number>>(() => {
    try {
      const raw = sessionStorage.getItem(PICKED_KEY);
      const ids: unknown = raw ? JSON.parse(raw) : null;
      return Array.isArray(ids)
        ? new Set(ids.filter((n): n is number => Number.isInteger(n)))
        : new Set();
    } catch { return new Set(); }
  });

  /* Where a shift-extend measures from: the last card whose check was clicked,
   * not the last member of the set — after a range, extending again continues
   * from where the hand last was. Rehydrated, because the gallery is its own
   * route and returning from Compare or a sample page remounts it. */
  const anchorRef = useRef<number | null>(
    (() => {
      const raw = Number(sessionStorage.getItem(PICK_ANCHOR_KEY));
      return Number.isInteger(raw) && raw > 0 ? raw : null;
    })());

  // The ranking's own order, read at click time so a range always spans what
  // is on screen now — never a list captured when the handler was made.
  const orderRef = useRef<number[]>([]);
  useEffect(() => { orderRef.current = items.map((s) => s.id); }, [items]);

  useEffect(() => {
    try {
      if (picked.size) {
        sessionStorage.setItem(PICKED_KEY, JSON.stringify([...picked]));
      } else {
        // No set, no anchor: an extend measured from a card nobody picked is
        // a range out of nowhere.
        sessionStorage.removeItem(PICKED_KEY);
        sessionStorage.removeItem(PICK_ANCHOR_KEY);
        anchorRef.current = null;
      }
    } catch { /* non-essential */ }
  }, [picked]);

  const toggle = (id: number, extend = false) => {
    // Read the anchor BEFORE queueing the update, and move it in the same
    // breath. A functional updater does not run at call time, so a ref written
    // straight after `setPicked` is already the card just clicked by the time
    // the updater reads it — `from === id`, every range collapses to a plain
    // toggle, and the feature silently does nothing. Measured exactly that:
    // click #0 then shift-click #20 gave 2 picked instead of 21.
    const from = anchorRef.current;
    anchorRef.current = id;
    try { sessionStorage.setItem(PICK_ANCHOR_KEY, String(id)); }
    catch { /* non-essential */ }
    setPicked((prev) => {
      const next = new Set(prev);
      const order = orderRef.current;
      const a = from == null ? -1 : order.indexOf(from);
      const b = order.indexOf(id);
      if (extend && a !== -1 && b !== -1 && a !== b) {
        // A range only ever ADDS. Making it mirror the anchor's state would
        // mean a stray shift-click could silently drop dozens of picks, and
        // this set is now durable enough that losing it quietly is the worst
        // thing the control could do. Un-picking stays one deliberate click.
        for (let i = Math.min(a, b); i <= Math.max(a, b); i++) next.add(order[i]);
        return next;
      }
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  /** A drag from a picked card carries the whole picked set — a selection is
   * one object, and dragging it should feel like moving that object. An
   * unpicked card drags alone. */
  const dragIds = (id: number) =>
    picked.has(id) && picked.size > 0 ? [...picked] : [id];

  return { picked, setPicked, toggle, dragIds };
}
