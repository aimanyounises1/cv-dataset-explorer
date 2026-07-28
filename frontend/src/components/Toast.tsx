import { useEffect, useState } from "react";

/** Toasts exist for undo, not narration.
 *
 * A drag-and-drop mutation happens away from any panel that could carry its
 * result, so the confirmation and its one recovery action need a home that is
 * always on screen. Everything else this app confirms, it confirms in place —
 * a toast that merely repeats what the user just watched happen is noise, so
 * the API deliberately takes one optional action and no severity levels. */

interface ToastMsg {
  id: number;
  text: string;
  action?: { label: string; run: () => void };
}

let push: ((t: Omit<ToastMsg, "id">) => void) | null = null;
let seq = 0;

export function showToast(text: string, action?: { label: string; run: () => void }) {
  push?.({ text, action });
}

export default function Toasts() {
  const [items, setItems] = useState<ToastMsg[]>([]);
  useEffect(() => {
    push = (t) => {
      const id = ++seq;
      // Keep at most three: an undo stack this is not — the newest actions
      // are the ones still worth reverting.
      setItems((prev) => [...prev.slice(-2), { ...t, id }]);
      window.setTimeout(
        () => setItems((prev) => prev.filter((x) => x.id !== id)), 6000);
    };
    return () => { push = null; };
  }, []);

  if (items.length === 0) return null;
  return (
    <div className="toasts" role="status" aria-live="polite">
      {items.map((t) => (
        <div className="toast" key={t.id}>
          <span>{t.text}</span>
          {t.action && (
            <button className="ghost" onClick={() => {
              t.action!.run();
              setItems((prev) => prev.filter((x) => x.id !== t.id));
            }}>
              {t.action.label}
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
