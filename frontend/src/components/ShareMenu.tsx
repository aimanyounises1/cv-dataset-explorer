import { useEffect, useRef, useState } from "react";
import type { AlbumDetail } from "../api/types";

/**
 * Local-first sharing. Every entry hands over a pointer to THIS machine —
 * the ?album= URL or the export endpoint — because the tool runs against a
 * local corpus and holds no credentials. Nothing here uploads anything; the
 * Teams entry only opens a compose window with the link pasted in, and the
 * footer says so out loud rather than letting "Share" imply a cloud.
 */
export default function ShareMenu({ album }: { album: AlbumDetail }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  const url = `${window.location.origin}/?album=${album.id}`;
  const blurb = `${album.name} — ${album.item_count} image${
    album.item_count === 1 ? "" : "s"} in CV Dataset Explorer: ${url}`;

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const copy = () => {
    navigator.clipboard.writeText(url).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    }).catch(() => {});
  };

  const mailto = `mailto:?subject=${
    encodeURIComponent(`${album.name} — CV Dataset Explorer`)}&body=${
    encodeURIComponent(blurb)}`;
  const teams = `https://teams.microsoft.com/l/chat/0/0?users=&message=${
    encodeURIComponent(blurb)}`;

  return (
    <div className="share-menu" ref={rootRef}>
      <button type="button" className="ghost" aria-haspopup="menu"
              aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        Share
      </button>
      {open && (
        <div className="share-pop" role="menu" aria-label={`Share “${album.name}”`}>
          <button type="button" role="menuitem" className="share-item" onClick={copy}>
            Copy link
            {copied && <span className="share-ok" role="status">Copied</span>}
          </button>
          {typeof navigator.share === "function" && (
            <button type="button" role="menuitem" className="share-item"
                    onClick={() => {
                      void navigator.share({ title: album.name, url }).catch(() => {});
                    }}>
              Share…
            </button>
          )}
          <a role="menuitem" className="share-item" href={mailto}>Email</a>
          <a role="menuitem" className="share-item" href={teams}
             target="_blank" rel="noopener noreferrer">
            Microsoft Teams
            <span className="share-sub">opens a Teams compose — nothing is uploaded</span>
          </a>
          <div className="share-item share-downloads">
            <span>Download</span>
            {(["json", "csv", "jsonl"] as const).map((f) => (
              <a key={f} href={`/api/export?album=${album.id}&format=${f}`}>{f}</a>
            ))}
          </div>
          <p className="share-foot">
            Everything stays local — links point at this machine.
          </p>
        </div>
      )}
    </div>
  );
}
