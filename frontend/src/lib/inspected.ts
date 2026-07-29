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

let current: number | null = null;
const listeners = new Set<() => void>();

/** Publish the cursor. Passing null clears it — do that on unmount, or the
 *  rail keeps a sample from a page that is no longer mounted. */
export function setInspected(id: number | null): void {
  if (id === current) return;
  current = id;
  listeners.forEach((fn) => fn());
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}

export function useInspected(): number | null {
  // The server snapshot is null: nothing is under a cursor before there is one.
  return useSyncExternalStore(subscribe, () => current, () => null);
}
