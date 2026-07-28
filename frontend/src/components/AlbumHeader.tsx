import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { AlbumAnalysis, AlbumDetail } from "../api/types";
import { api } from "../api/client";
import { ALBUMS_CHANGED, albumsChanged, DRAG_IDS } from "./AlbumShelf";
import ShareMenu from "./ShareMenu";

/**
 * The album, editable where it is read. Renders above the grid whenever
 * `?album=` is active: inline-renamable name, a details disclosure with the
 * editable summary/category/notes (PATCH — the same fields the analysis
 * drafts into), provenance, and the Analyze panel.
 *
 * The analysis panel keeps the two epistemic halves visually apart: measured
 * signals in the product's own voice, the generated draft in agent indigo,
 * named after its model, editable, and saved only by the user's click.
 */
export default function AlbumHeader({ albumId, onGone }:
    { albumId: number; onGone: () => void }) {
  const [album, setAlbum] = useState<AlbumDetail | null>(null);
  const [missing, setMissing] = useState(false);
  const [editingName, setEditingName] = useState(false);
  const [draft, setDraft] = useState({ summary: "", category: "", notes: "" });
  const [open, setOpen] = useState(false);
  const [analysis, setAnalysis] = useState<AlbumAnalysis | null>(null);
  const [analysisBusy, setAnalysisBusy] = useState(false);
  const [genBusy, setGenBusy] = useState(false);
  const [genDraft, setGenDraft] = useState<string | null>(null);
  const [genModel, setGenModel] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  // Member-strip reorder: which thumb is being dragged, which one it hovers.
  const [dragSid, setDragSid] = useState<number | null>(null);
  const [overSid, setOverSid] = useState<number | null>(null);
  const stripRef = useRef<HTMLDivElement | null>(null);
  // An arrow-key move re-renders the strip in the new order, which would drop
  // keyboard focus on the floor; this remembers whom to give it back to.
  const focusSid = useRef<number | null>(null);

  const load = useCallback(() => {
    api.albumDetail(albumId)
      .then((a) => {
        setAlbum(a);
        setMissing(false);
        setDraft({ summary: a.summary ?? "", category: a.category ?? "",
                   notes: a.notes ?? "" });
      })
      .catch(() => setMissing(true));
  }, [albumId]);

  useEffect(() => {
    load();
    window.addEventListener(ALBUMS_CHANGED, load);
    return () => window.removeEventListener(ALBUMS_CHANGED, load);
  }, [load]);

  useEffect(() => {
    const sid = focusSid.current;
    if (sid == null || !stripRef.current) return;
    focusSid.current = null;
    stripRef.current
      .querySelector<HTMLElement>(`[data-member="${sid}"]`)?.focus();
  }, [album]);

  if (missing) {
    return (
      <div className="album-header">
        <p className="ah-missing">This album no longer exists.{" "}
          <button className="link-btn" onClick={onGone}>Back to the gallery</button></p>
      </div>
    );
  }
  if (!album) return null;

  const patch = (body: Parameters<typeof api.updateAlbum>[1], done?: string) =>
    api.updateAlbum(album.id, body)
      .then((a) => { setAlbum(a); albumsChanged(); if (done) setNote(done); })
      .catch((e) => setNote(e instanceof Error ? e.message : String(e)));

  /** The order endpoint demands exactness — the new order must be the whole
   * current membership, nothing added, nothing missing — so every commit
   * sends the full id list, never a delta. */
  const commitOrder = (orderedIds: number[]) =>
    fetch(`/api/albums/${album.id}/items/order`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_ids: orderedIds }),
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
        albumsChanged();
      })
      .catch((e) => {
        setNote(e instanceof Error ? e.message : String(e));
        load();                       // put the strip back to the server's order
      });

  /** Apply a new order optimistically (arrow keys and drops should feel
   * instant), then commit; ALBUMS_CHANGED reconciles with the server. */
  const applyOrder = (ids: number[]) => {
    const byId = new Map(album.items.map((s) => [s.id, s]));
    setAlbum({ ...album, items: ids.map((id) => byId.get(id)!) });
    void commitOrder(ids);
  };

  const moveMember = (sid: number, delta: -1 | 1) => {
    const ids = album.items.map((s) => s.id);
    const i = ids.indexOf(sid);
    const j = i + delta;
    if (i < 0 || j < 0 || j >= ids.length) return;
    [ids[i], ids[j]] = [ids[j], ids[i]];
    focusSid.current = sid;
    applyOrder(ids);
  };

  const dropReorder = (fromSid: number, toSid: number) => {
    setOverSid(null);
    setDragSid(null);
    if (fromSid === toSid) return;
    const ids = album.items.map((s) => s.id);
    const from = ids.indexOf(fromSid);
    const to = ids.indexOf(toSid);
    if (from < 0 || to < 0) return;
    ids.splice(from, 1);
    // Dragging rightward lands after the target, leftward before it — the
    // thumb settles on the side it was pulled toward.
    const at = ids.indexOf(toSid);
    ids.splice(from < to ? at + 1 : at, 0, fromSid);
    applyOrder(ids);
  };

  const removeMember = (sid: number) =>
    api.removeFromAlbum(album.id, sid)
      .then(() => albumsChanged())
      .catch((e) => setNote(e instanceof Error ? e.message : String(e)));

  const runAnalysis = () => {
    setAnalysisBusy(true);
    api.albumAnalysis(album.id)
      .then(setAnalysis)
      .catch((e) => setNote(e instanceof Error ? e.message : String(e)))
      .finally(() => setAnalysisBusy(false));
  };

  const generate = () => {
    setGenBusy(true);
    api.generateAlbumSummary(album.id)
      .then((r) => { setGenDraft(r.summary); setGenModel(r.model); })
      .catch((e) => setNote(e instanceof Error
        ? e.message.replace(/^\d+:\s*/, "").replace(/^"|"$/g, "") : String(e)))
      .finally(() => setGenBusy(false));
  };

  const thumbOf = (id: number) =>
    album.items.find((s) => s.id === id)?.thumb_url ?? null;

  return (
    <div className="album-header">
      <div className="ah-row">
        {editingName ? (
          <input className="ah-name-input" defaultValue={album.name} autoFocus
                 onKeyDown={(e) => {
                   if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                   if (e.key === "Escape") setEditingName(false);
                 }}
                 onBlur={(e) => {
                   const v = e.target.value.trim();
                   setEditingName(false);
                   if (v && v !== album.name) patch({ name: v }, "Renamed.");
                 }} />
        ) : (
          <h2 className="ah-name" title="Click to rename"
              onClick={() => setEditingName(true)}>{album.name}</h2>
        )}
        <span className="ah-meta">
          {album.item_count} image{album.item_count === 1 ? "" : "s"}
          {" · "}{album.origin}
          {" · "}created {album.created_at.slice(0, 10)}
        </span>
        <span className="ah-spacer" />
        <ShareMenu album={album} />
        {/* Named for what is behind it. Removing an image, setting the cover
            and reordering all live in this panel, and "Details & analysis"
            promises reading rather than editing — so nobody looking to drop a
            frame from the album had a reason to open it. */}
        <button className="ghost" onClick={() => setOpen((o) => !o)}
                title={open ? "Hide the member strip and analysis"
                            : "Edit membership — remove, set the cover, reorder — and analyse the set"}>
          {open ? "Close members"
                : `Members & analysis${album.item_count ? ` (${album.item_count})` : ""}`}
        </button>
        {confirmDelete ? (
          <button className="danger"
                  onClick={() => api.deleteAlbum(album.id)
                    .then(() => { albumsChanged(); onGone(); })}>
            Really delete?
          </button>
        ) : (
          <button className="ghost" onClick={() => {
            setConfirmDelete(true);
            window.setTimeout(() => setConfirmDelete(false), 4000);
          }}>Delete</button>
        )}
      </div>
      {album.summary && !open && <p className="ah-summary">{album.summary}</p>}
      {note && <p className="ah-note" onAnimationEnd={() => setNote(null)}>{note}</p>}

      {open && (
        <div className="ah-details">
          {album.items.length > 0 && (
            <div className="ah-members-wrap">
              <span className="ah-members-label">
                Members, in album order — drag or use ←/→ to reorder
              </span>
              <div className="ah-members" role="list" ref={stripRef}
                   aria-label={`${album.item_count} album member${
                     album.item_count === 1 ? "" : "s"}, reorderable`}>
                {album.items.map((s, i) => {
                  const isCover = album.cover_sample_id === s.id;
                  return (
                    <div key={s.id} role="listitem" tabIndex={0}
                         data-member={s.id}
                         className={`ah-member${isCover ? " is-cover" : ""}${
                           overSid === s.id ? " drop" : ""}${
                           dragSid === s.id ? " dragging" : ""}`}
                         aria-label={`Image ${s.id}, position ${i + 1} of ${
                           album.items.length}${isCover ? ", cover" : ""}.`
                           + " Arrow keys move it."}
                         draggable
                         onDragStart={(e) => {
                           setDragSid(s.id);
                           // The payload a card would carry, so a member thumb
                           // can also be filed onto another shelf row.
                           e.dataTransfer.setData(DRAG_IDS, JSON.stringify([s.id]));
                           e.dataTransfer.effectAllowed = "move";
                         }}
                         onDragEnd={() => { setDragSid(null); setOverSid(null); }}
                         onDragOver={(e) => {
                           if (dragSid != null && dragSid !== s.id) {
                             e.preventDefault();
                             setOverSid(s.id);
                           }
                         }}
                         onDragLeave={() =>
                           setOverSid((o) => (o === s.id ? null : o))}
                         onDrop={(e) => {
                           if (dragSid == null) return;
                           e.preventDefault();
                           e.stopPropagation();
                           dropReorder(dragSid, s.id);
                         }}
                         onKeyDown={(e) => {
                           if (e.key === "ArrowLeft") {
                             e.preventDefault(); moveMember(s.id, -1);
                           } else if (e.key === "ArrowRight") {
                             e.preventDefault(); moveMember(s.id, 1);
                           }
                         }}>
                      <img src={s.thumb_url} alt={`sample ${s.id}`} />
                      <button type="button" className="ah-member-x"
                              aria-label={`Remove image ${s.id} from this album`}
                              onClick={() => void removeMember(s.id)}>
                        ✕
                      </button>
                      <button type="button"
                              className={`ah-member-star${isCover ? " on" : ""}`}
                              aria-label={isCover
                                ? `Image ${s.id} is the cover`
                                : `Set image ${s.id} as the album cover`}
                              aria-pressed={isCover}
                              onClick={() => {
                                if (!isCover) {
                                  void patch({ cover_sample_id: s.id }, "Cover set.");
                                }
                              }}>
                        {isCover ? "★" : "☆"}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          <label>Summary
            <textarea value={draft.summary} rows={2}
                      onChange={(e) => setDraft({ ...draft, summary: e.target.value })} />
          </label>
          <div className="ah-two">
            <label>Category
              <input value={draft.category}
                     onChange={(e) => setDraft({ ...draft, category: e.target.value })} />
            </label>
            <label>Notes
              <input value={draft.notes}
                     onChange={(e) => setDraft({ ...draft, notes: e.target.value })} />
            </label>
          </div>
          <div className="ah-actions">
            <button className="primary" onClick={() => patch({
              summary: draft.summary || null, category: draft.category || null,
              notes: draft.notes || null,
            }, "Saved.")}>Save details</button>
            <button className="ghost" onClick={runAnalysis} disabled={analysisBusy}>
              {analysisBusy ? "Analyzing…" : "Analyze"}
            </button>
          </div>

          {analysis && (
            <div className="ah-analysis">
              <div className="ah-measured">
                <div className="ah-panel-title">
                  Measured from the dataset
                  {analysis.measured.coherence != null && (
                    <span className="chiplet" title={`basis: ${analysis.measured.score_basis}`}>
                      coherence {analysis.measured.coherence}
                    </span>
                  )}
                </div>
                {analysis.measured.common.length > 0 && (
                  <p><strong>Common:</strong>{" "}
                    {analysis.measured.common.map((c) => (
                      <span className="chiplet" key={`${c.kind}:${c.label}`}>
                        {c.label} {Math.round(c.share * 100)}%
                      </span>
                    ))}</p>
                )}
                {analysis.measured.different.length > 0 && (
                  <p><strong>Splits:</strong>{" "}
                    {analysis.measured.different.map((d) => (
                      <span className="chiplet" key={d.grp}>
                        {d.grp}: {d.top.map((t) =>
                          `${t.label} ${Math.round(t.share * 100)}%`).join(" vs ")}
                      </span>
                    ))}</p>
                )}
                {analysis.measured.outliers.length > 0 && (
                  <p className="ah-outliers"><strong>Outliers:</strong>{" "}
                    {analysis.measured.outliers.map((o) => (
                      <Link key={o.id} to={`/samples/${o.id}`}
                            title={`cosine to album centroid ${o.score}`}>
                        {thumbOf(o.id)
                          ? <img src={thumbOf(o.id)!} alt={`outlier ${o.id}`} />
                          : `#${o.id}`}
                      </Link>
                    ))}</p>
                )}
                {analysis.measured.note && (
                  <p className="ah-panel-note">{analysis.measured.note}</p>
                )}
              </div>
              <div className="ah-generated">
                <div className="ah-panel-title ai">
                  AI observation — {analysis.generated.model}, from captions and
                  measured signals (no pixels)
                </div>
                {genDraft != null ? (
                  <>
                    <textarea value={genDraft} rows={3}
                              onChange={(e) => setGenDraft(e.target.value)} />
                    <div className="ah-actions">
                      <button className="primary" onClick={() => {
                        patch({ summary: genDraft }, "Saved as album summary.");
                        setDraft({ ...draft, summary: genDraft });
                      }}>Save as album summary</button>
                      <span className="ah-panel-note">
                        drafted by {genModel} — yours after you save it
                      </span>
                    </div>
                  </>
                ) : analysis.generated.available ? (
                  <button className="ai-btn" onClick={generate} disabled={genBusy}>
                    {genBusy ? "Drafting…" : `Draft a summary (${analysis.generated.model})`}
                  </button>
                ) : (
                  <p className="ah-panel-note">{analysis.generated.message}</p>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
