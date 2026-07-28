import { useCallback, useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { AlbumSummary } from "../api/types";
import { showToast } from "./Toast";

/** Every album surface announces its mutations here and refetches on hearing
 * one — the shelf, the tray and (later) the agent's proposals stay in step
 * without a store. */
export const ALBUMS_CHANGED = "cvde-albums-changed";
export function albumsChanged() {
  window.dispatchEvent(new Event(ALBUMS_CHANGED));
}

/** The MIME type cards put on the drag: a JSON array of sample ids. Custom
 * type rather than text/plain so a stray text drop into an input can never
 * paste an id list. */
export const DRAG_IDS = "application/x-cvde-ids";

/**
 * The album shelf: ordered collections as destinations.
 *
 * Rows are drop targets — drag a card (or a picked bundle) from the gallery
 * onto a row to file it. The keyboard path to the same mutation is select
 * mode's tray, and every drop that added something offers Undo. The add
 * response only carries a count, so membership is read and diffed BEFORE the
 * add: Undo removes exactly the ids this drop introduced, never a member
 * that was already there.
 */
export default function AlbumShelf() {
  const [albums, setAlbums] = useState<AlbumSummary[]>([]);
  const [dragOver, setDragOver] = useState<number | null>(null);
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const active = Number(params.get("album")) || null;

  const load = useCallback(() => {
    // Rail furniture fails silent, like the corpus count: a missing shelf is
    // not worth an error state on every route.
    api.listAlbums().then(setAlbums).catch(() => {});
  }, []);
  useEffect(() => {
    load();
    window.addEventListener(ALBUMS_CHANGED, load);
    return () => window.removeEventListener(ALBUMS_CHANGED, load);
  }, [load]);

  const onDrop = (album: AlbumSummary, e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(null);
    const raw = e.dataTransfer.getData(DRAG_IDS);
    if (!raw) return;
    let ids: number[] = [];
    try { ids = JSON.parse(raw) as number[]; } catch { return; }
    if (ids.length === 0) return;
    // Read membership first and diff: `fresh` is the set this drop is about
    // to introduce, and the only set Undo may remove. The dropped ids alone
    // cannot say which were already members, and the add response is a count.
    api.albumDetail(album.id)
      .then((detail) => {
        const members = new Set(detail.items.map((s) => s.id));
        const fresh = ids.filter((id) => !members.has(id));
        return api.addToAlbum(album.id, ids).then((r) => ({ r, fresh }));
      })
      .then(({ r, fresh }) => {
        albumsChanged();
        const skipped = ids.length - r.added;
        // A count that disagrees with the diff means another writer moved the
        // membership between our read and our add — an Undo built on that
        // would be guessing, so it is withheld.
        const provable = r.added > 0 && r.added === fresh.length;
        showToast(
          skipped === 0
            ? `Added ${r.added} to “${album.name}”`
            : `Added ${r.added} to “${album.name}” — ${skipped} already there`,
          provable
            ? {
                label: "Undo",
                run: () => {
                  void Promise.allSettled(
                    fresh.map((sid) => api.removeFromAlbum(album.id, sid)),
                  ).then((results) => {
                    albumsChanged();
                    // A swallowed failure would leave the count quietly
                    // wrong — the one thing an undo must never do.
                    const failed = results.filter(
                      (x) => x.status === "rejected").length;
                    if (failed > 0) {
                      showToast(`Undo removed ${fresh.length - failed} of ${
                        fresh.length} — ${failed} could not be removed`);
                    }
                  });
                },
              }
            : undefined,
        );
      })
      .catch((err) =>
        showToast(err instanceof Error ? err.message : "Could not add to album"));
  };

  return (
    <div className="album-shelf">
      {albums.map((a) => (
        <button
          type="button"
          key={a.id}
          className={`album-row${active === a.id ? " active" : ""}${dragOver === a.id ? " drop" : ""}`}
          title={`${a.name} — ${a.item_count} image${a.item_count === 1 ? "" : "s"}`
                 + (a.origin !== "manual" ? ` · from ${a.origin}` : "")
                 + ". Click to open; drop images here to add."}
          onClick={() => navigate(`/?album=${a.id}`)}
          onDragOver={(e) => { e.preventDefault(); setDragOver(a.id); }}
          onDragLeave={() => setDragOver((d) => (d === a.id ? null : d))}
          onDrop={(e) => onDrop(a, e)}
        >
          <span className="album-stack" aria-hidden="true">
            {a.cover
              ? <img src={a.cover} alt="" loading="lazy" />
              : <span className="album-stack-empty" />}
          </span>
          <span className="album-name">{a.name}</span>
          <span className="album-count">{a.item_count}</span>
        </button>
      ))}
      {albums.length === 0 && (
        <p className="album-empty">
          No albums yet — pick images in the gallery and name the set, or drag
          a card here once one exists.
        </p>
      )}
    </div>
  );
}
