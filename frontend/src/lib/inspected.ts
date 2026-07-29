import { useSyncExternalStore } from "react";

/**
 * Which sample the cursor is resting on.
 *
 * The grid renders inside `<main>`; the rail that shows the inspected sample is
 * a *sibling* of it (`App.tsx`), so no common ancestor holds both without
 * lifting gallery state into the shell. This is the third thing in the app that
 * needs to speak across that boundary, and it follows what the other two
 * already do rather than introducing a fourth mechanism: `Toast.tsx` keeps a
 * module-level `push`, `AlbumShelf.tsx` uses a window event. A module store
 * read through `useSyncExternalStore` is the same idea with a subscription
 * React can tear-check.
 *
 * Deliberately not in the URL. `useSelection` owns everything shareable, and a
 * cursor is not: it changes on every arrow key, and a link to "the image I was
 * hovering" is a link to nothing. It is not in `sessionStorage` either — unlike
 * scroll position there is no returning-reader case, so it dies with the page.
 */

/** The mode and the cursor travel together, and the mode is the load-bearing
 *  half. The inspector costs the rail's width, and the rail is a grid column
 *  that collapses when empty — so if the rail only appeared once a card was
 *  under the cursor, the grid would reflow on the first hover and carry the
 *  cards out from under the pointer that summoned it. Opening the mode claims
 *  the space immediately; from then on the layout is still and only the
 *  contents change. */
export interface Cursor { active: boolean; id: number | null }

const IDLE: Cursor = { active: false, id: null };
let current: Cursor = IDLE;
const listeners = new Set<() => void>();

function publish(next: Cursor): void {
  if (next.active === current.active && next.id === current.id) return;
  current = next;
  listeners.forEach((fn) => fn());
}

/** Open or close the inspector. Closing also drops the cursor: reopening should
 *  show what you are pointing at now, not what you left behind. */
export function setInspecting(active: boolean): void {
  publish(active ? { active: true, id: current.id } : IDLE);
}

/** Publish the card under the cursor. Ignored while closed, so a stray hover
 *  cannot open a panel nobody asked for. */
export function setInspected(id: number | null): void {
  if (!current.active) return;
  publish({ active: true, id });
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}

export function useCursor(): Cursor {
  // The server snapshot is idle: there is no cursor before there is a pointer.
  return useSyncExternalStore(subscribe, () => current, () => IDLE);
}
