import { useEffect, useMemo } from "react";
import { setInspected } from "../lib/inspected";

/**
 * Turns a grid of cards into a cursor: whichever card the reader is on gets
 * published to the inspector rail.
 *
 * Delegated, not per-card. One listener on the container reads
 * `closest("[data-sample-id]")` off the event target, so 48 tiles cost two
 * handlers rather than 96, and a card that is added by the next page of results
 * is covered without re-binding anything.
 *
 * Focus is the real cursor and hover is only a preview:
 *
 * - `onFocusCapture` fires for keyboard and click alike, works on touch, and is
 *   the one that survives into assistive tech. Capture phase because focus does
 *   not bubble.
 * - `onPointerOver` is gated on `(hover: hover)`. On a touch screen the browser
 *   synthesises a pointerover at the tap point, which would make every tap
 *   inspect the card before navigating to it. The gallery already uses this
 *   media query to decide what may be hover-only.
 *
 * Returns nothing when inactive, so the handlers are not attached at all rather
 * than attached and returning early — the grid stays exactly as cheap as it was
 * before the loupe existed.
 */
export function useGridCursor(active: boolean) {
  const canHover = useMemo(
    () => typeof window !== "undefined"
      && !!window.matchMedia?.("(hover: hover)").matches,
    []);

  // The rail reads a module store, so a cursor left behind outlives the grid
  // that set it: switching to the map would show an inspector for a card that
  // is no longer on screen. Clear on deactivate and on unmount both.
  useEffect(() => {
    if (!active) setInspected(null);
    return () => setInspected(null);
  }, [active]);

  return useMemo(() => {
    if (!active) return {};
    const read = (e: { target: EventTarget | null }) => {
      const el = (e.target as HTMLElement | null)?.closest?.("[data-sample-id]");
      const id = el?.getAttribute("data-sample-id");
      if (id) setInspected(Number(id));
    };
    return {
      onFocusCapture: read,
      onPointerOver: canHover ? read : undefined,
    };
  }, [active, canHover]);
}
