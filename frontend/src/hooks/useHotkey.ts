import { useEffect, useRef } from "react";

/**
 * Binds a single global key combo for the life of the component.
 *
 * `match`/`callback` are read from refs on every keydown rather than being
 * effect dependencies, so callers can pass fresh inline closures each render
 * without re-subscribing the window listener.
 *
 * This hook always calls `preventDefault()` on a match and never checks where
 * focus currently is. That makes it correct only for chords a text field would
 * never otherwise receive (Cmd/Ctrl+letter, function keys) — never bind a bare
 * letter or Enter/Escape/Arrow with this, or it will eat those keys while the
 * user is typing in an unrelated `<input>` elsewhere in the app. The caller
 * (CommandPalette) relies on exactly this to open on Cmd/Ctrl+K even while
 * focus is inside some other input or textarea.
 */
export function useHotkey(
  match: (e: KeyboardEvent) => boolean,
  callback: (e: KeyboardEvent) => void,
): void {
  const matchRef = useRef(match);
  matchRef.current = match;
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (matchRef.current(e)) {
        e.preventDefault();
        callbackRef.current(e);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
}
