import { useEffect, useMemo, useState } from "react";
import { setInspected, setInspecting } from "../lib/inspected";
import { COMPACT_RAIL_QUERY } from "../components/shell/LeftRail";

function matches(query: string): boolean {
  return typeof window !== "undefined"
    && typeof window.matchMedia === "function"
    && window.matchMedia(query).matches;
}

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
export function useGridCursor(wanted: boolean) {
  const canHover = useMemo(() => matches("(hover: hover)"), []);

  /* Below the app's three-column breakpoint both rails stack, so the inspector
     would land underneath the whole grid — where it is strictly worse than
     opening the sample page, because you would scroll past every result to
     reach it. The mode is therefore *unavailable* rather than merely awkward
     there, and the toggle that offers it goes away with it. The reader's stored
     preference is untouched: widening the window brings it back on. */
  const [roomy, setRoomy] = useState(() => !matches(COMPACT_RAIL_QUERY));
  useEffect(() => {
    if (typeof window === "undefined"
        || typeof window.matchMedia !== "function") return undefined;
    const media = window.matchMedia(COMPACT_RAIL_QUERY);
    const onChange = () => setRoomy(!media.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  const active = wanted && roomy;

  // The rail reads a module store, so a mode left open outlives the grid that
  // opened it: switching to the map would show an inspector for a card that is
  // no longer on screen. Close on deactivate and on unmount both.
  useEffect(() => {
    setInspecting(active);
    return () => setInspecting(false);
  }, [active]);

  const gridProps = useMemo(() => {
    const read = (e: { target: EventTarget | null }) => {
      const el = (e.target as HTMLElement | null)?.closest?.("[data-sample-id]");
      const id = el?.getAttribute("data-sample-id");
      if (id) setInspected(Number(id));
    };
    return {
      // Always bound. Walking a grid with the arrow keys is worth having on its
      // own; the inspector is what makes it fast, not what makes it legal.
      onKeyDown: moveCursor,
      onFocusCapture: active ? read : undefined,
      onPointerOver: active && canHover ? read : undefined,
    };
  }, [active, canHover]);

  return { gridProps, available: roomy };
}

/** Rows are laid out by `auto-fill`, so the column count is a fact about the
 *  rendered box and not about `--frame-min`: read it back off the DOM by
 *  counting the cards that share the first one's top edge. Any density, any
 *  viewport, any future change to the grid CSS stays correct for free. */
function columnsIn(cards: HTMLElement[]): number {
  if (cards.length < 2) return 1;
  const top = cards[0].offsetTop;
  let n = 1;
  while (n < cards.length && cards[n].offsetTop === top) n++;
  return n;
}

/**
 * Arrow-key movement over the grid.
 *
 * Follows the W3C ARIA APG grid pattern (https://www.w3.org/WAI/ARIA/apg/
 * patterns/grid/) for `Home`/`End`, `Ctrl+Home`/`Ctrl+End`, and for arrows not
 * running off the ends — with one deliberate divergence. APG's grid is tabular,
 * where the right-most cell of a row is a wall. This grid is a *ranked list*
 * that happens to wrap: the card after the last one in a row is the next rank,
 * and stopping there would make ← and → mean something different depending on
 * where the browser chose to break the row. So ← and → walk the ranking, and
 * ↑ and ↓ walk rows and stop at the ends, which is also what photo grids do.
 *
 * Focus itself is the cursor — no roving `tabindex`. The cards are already
 * links, so this adds a fast path to a grid that is keyboard-reachable today
 * rather than restructuring its tab order. Demoting 48 tiles to one tab stop is
 * the APG-complete version and a larger change: each card also holds a pick
 * control, which would have to join the same roving scheme to be reached at all.
 */
function moveCursor(e: React.KeyboardEvent<HTMLElement>): void {
  if (e.metaKey || e.altKey || e.defaultPrevented) return;
  const target = e.target as HTMLElement | null;
  const here = target?.closest?.("[data-sample-id]") as HTMLElement | null;
  if (!here) return;

  const cards = Array.from(
    e.currentTarget.querySelectorAll<HTMLElement>("[data-sample-id]"));
  const i = cards.indexOf(here);
  if (i < 0) return;

  const cols = columnsIn(cards);
  const last = cards.length - 1;
  // A screenful of rows, from the height the cards actually render at.
  const rows = Math.max(
    1, Math.floor(window.innerHeight / (here.offsetHeight || window.innerHeight)));

  let next: number;
  switch (e.key) {
    case "ArrowRight": next = i + 1; break;
    case "ArrowLeft":  next = i - 1; break;
    case "ArrowDown":  next = i + cols; break;
    case "ArrowUp":    next = i - cols; break;
    case "PageDown":   next = Math.min(i + cols * rows, last); break;
    case "PageUp":     next = Math.max(i - cols * rows, 0); break;
    case "Home":
      next = e.ctrlKey ? 0 : i - (i % cols);
      break;
    case "End":
      next = e.ctrlKey ? last : Math.min(i - (i % cols) + cols - 1, last);
      break;
    default: return;
  }

  if (next < 0 || next > last) return;   // the ends are walls, not wraps
  e.preventDefault();                    // only once we know we are moving
  cards[next].focus();
}
