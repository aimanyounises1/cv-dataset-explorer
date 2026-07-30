import { useEffect, useRef } from "react";
import type { RefObject } from "react";

/** Restore the scroll position for back-nav, keyed by the full query string in
 * sessionStorage: scan position is ephemeral scan-order state like the result
 * order, not something a pasted link should reproduce — the URL deliberately
 * does not own it. Restored once per mount, once `ready` says the first result
 * set has rendered, because before that the page has no height to scroll into.
 *
 * `pageRef` is the page's own root, and it is load-bearing rather than
 * decorative: see the listener below. */
export function useScrollRestore(
  pageRef: RefObject<HTMLElement | null>, key: string, ready: boolean,
) {
  const keyRef = useRef(key);
  useEffect(() => { keyRef.current = key; }, [key]);

  useEffect(() => {
    const save = () => {
      // Navigating away fires one last scroll event: the next page's shorter
      // document clamps the position to 0 before this listener is removed
      // (passive-effect cleanup runs after paint). Recording that 0 would
      // overwrite the very position this key exists to keep, so a scroll only
      // counts while the gallery's own DOM is still attached. Measured: the
      // clamp event arrives with the root already disconnected.
      if (!pageRef.current || !pageRef.current.isConnected) return;
      try { sessionStorage.setItem(keyRef.current, String(window.scrollY)); }
      catch { /* non-essential */ }
    };
    window.addEventListener("scroll", save, { passive: true });
    return () => window.removeEventListener("scroll", save);
  }, [pageRef]);

  const restored = useRef(false);
  useEffect(() => {
    if (restored.current || !ready) return;
    restored.current = true;
    const saved = Number(sessionStorage.getItem(key) ?? NaN);
    if (Number.isFinite(saved) && saved > 0) window.scrollTo(0, saved);
  }, [ready, key]);
}
