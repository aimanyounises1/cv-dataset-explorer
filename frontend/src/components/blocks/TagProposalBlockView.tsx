import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import type { TagProposalBlock } from "../../api/blocks";
import { api } from "../../api/client";
import type { SampleCard } from "../../api/types";

/**
 * The approval boundary, rendered. The assistant proposed a tag; nothing has
 * been written. Approve performs the bulk-tag call FROM THIS BROWSER — the
 * mutation is the user's click, which is why the button wears the product's
 * sage, not the assistant's indigo — and records a `tag_approval` activity
 * event. Reject touches no network at all.
 *
 * Approval is per image. A proposal is a suggestion about many samples, and a
 * reviewer who agrees with most of it should not have to choose between
 * accepting the ones they disagree with and dropping the whole thing. Every
 * member is therefore listed and individually toggleable, and only the chosen
 * ids are written.
 *
 * Nothing is tagged that was not on screen to be judged: the list renders EVERY
 * proposed id, not a preview. Thumbnails come in one bounded request, so beyond
 * that ceiling members appear as id tiles — still listed, still selectable,
 * never silently included.
 */

/** One request's worth of thumbnails; the API caps per_page at 200. */
const THUMB_CAP = 200;

export default function TagProposalBlockView({ block }: { block: TagProposalBlock }) {
  const [thumbs, setThumbs] = useState<SampleCard[] | null>(null);
  const [picked, setPicked] = useState<Set<number>>(() => new Set(block.sample_ids));
  const [state, setState] = useState<
    { phase: "open" } | { phase: "busy" }
    | { phase: "approved"; tagged: number; of: number } | { phase: "rejected" }
    | { phase: "failed"; message: string }
  >({ phase: "open" });

  const ids = block.sample_ids;
  const idKey = ids.join(",");
  useEffect(() => {
    if (ids.length === 0) { setThumbs([]); return; }
    const ctrl = new AbortController();
    api.listSamples({ ids: ids.slice(0, THUMB_CAP).join(","), per_page: THUMB_CAP },
                    ctrl.signal)
      .then((list) => setThumbs(list.items))
      // A failed thumbnail fetch must not hide a member: the list falls back to
      // id tiles below, so the proposal stays reviewable without pictures.
      .catch(() => { if (!ctrl.signal.aborted) setThumbs([]); });
    return () => ctrl.abort();
    // `ids` is a fresh array each render; `idKey` is its identity, so the
    // fetch re-runs only when the membership actually changes.
  }, [idKey]);

  // A proposal that changes underneath is a different proposal. Keyed by
  // idKey for the same identity reason as the thumbnail fetch above.
  useEffect(() => { setPicked(new Set(ids)); setState({ phase: "open" });
  }, [idKey]);

  const byId = useMemo(() => {
    const m = new Map<number, SampleCard>();
    for (const s of thumbs ?? []) m.set(s.id, s);
    return m;
  }, [thumbs]);

  const toggle = (id: number) => setPicked((prev) => {
    const next = new Set(prev);
    if (!next.delete(id)) next.add(id);
    return next;
  });

  const approve = async () => {
    const chosen = ids.filter((id) => picked.has(id));   // proposal order, not click order
    if (chosen.length === 0) return;
    setState({ phase: "busy" });
    try {
      const res = await api.bulkTag(chosen, block.tag);
      // The trail's witness for a client-side mutation is the client. It records
      // what was offered as well as what was taken, so a partial approval is
      // legible later as a decision rather than a smaller proposal.
      api.recordActivity("tag_approval",
        { tag: res.tag, tagged: res.tagged, selected: chosen.length,
          proposed: ids.length }).catch(() => {});
      setState({ phase: "approved", tagged: res.tagged, of: ids.length });
    } catch (e) {
      setState({ phase: "failed", message: e instanceof Error ? e.message : String(e) });
    }
  };

  const open = state.phase === "open" || state.phase === "busy";
  const all = picked.size === ids.length;

  return (
    <div className="vblock-body tag-proposal">
      {block.reason && <p className="tp-reason">{block.reason}</p>}

      {open && ids.length > 0 && (
        <div className="tp-bulk">
          <span className="tp-count" aria-live="polite">
            {picked.size} of {ids.length} chosen
          </span>
          <button type="button" className="link-btn" disabled={all}
                  onClick={() => setPicked(new Set(ids))}>Select all</button>
          <button type="button" className="link-btn" disabled={picked.size === 0}
                  onClick={() => setPicked(new Set())}>Clear</button>
        </div>
      )}

      <div className={`tp-thumbs${open ? " picking" : ""}`}>
        {ids.map((id) => {
          const s = byId.get(id);
          const on = picked.has(id);
          if (!open) {
            return s ? (
              <Link key={id} to={`/samples/${id}`} title={`#${id}`}>
                <img src={s.thumb_url} alt={`sample ${id}`} loading="lazy" />
              </Link>
            ) : <span key={id} className="tp-idtile">#{id}</span>;
          }
          return (
            <button key={id} type="button"
                    className={`tp-item${on ? " on" : ""}${s ? "" : " bare"}`}
                    aria-pressed={on}
                    aria-label={`${on ? "Remove" : "Include"} sample ${id}`}
                    title={s ? `#${id} — click to ${on ? "exclude" : "include"}`
                             : `#${id} (no thumbnail) — click to ${on ? "exclude" : "include"}`}
                    onClick={() => toggle(id)}>
              {s ? <img src={s.thumb_url} alt="" loading="lazy" />
                 : <span className="tp-idtile">#{id}</span>}
              <span className="tp-pick" aria-hidden="true">{on ? "✓" : ""}</span>
            </button>
          );
        })}
      </div>

      {thumbs === null && ids.length > 0 && <p className="tp-note">Loading the proposal…</p>}
      {ids.length > THUMB_CAP && (
        <p className="tp-note">
          Thumbnails load for the first {THUMB_CAP}; the rest are listed by id above
          and choose exactly the same way — nothing is tagged unlisted.
        </p>
      )}
      {block.missing && block.missing.length > 0 && (
        <p className="tp-missing">
          {block.missing.length} requested id{block.missing.length > 1 ? "s" : ""} do
          not exist and were dropped from this proposal.
        </p>
      )}

      {open ? (
        <div className="tp-actions">
          <button className="primary" onClick={approve}
                  disabled={state.phase === "busy" || picked.size === 0}
                  title={picked.size === 0
                    ? "Choose at least one image to tag"
                    : `Writes the tag to ${picked.size} sample${picked.size === 1 ? "" : "s"}`}>
            {state.phase === "busy" ? "Tagging…"
              : `Approve — tag ${picked.size} as ‘${block.tag}’`}
          </button>
          <button className="ghost" onClick={() => setState({ phase: "rejected" })}
                  disabled={state.phase === "busy"}>
            Reject
          </button>
        </div>
      ) : state.phase === "approved" ? (
        <p className="tp-done">
          Tagged {state.tagged} of {state.of} proposed sample{state.of === 1 ? "" : "s"} as
          ‘{block.tag}’.{" "}
          <Link to={`/?tag=${encodeURIComponent(block.tag)}`}>Open the slice</Link>
        </p>
      ) : state.phase === "rejected" ? (
        <p className="tp-done">Proposal dismissed — nothing was written.</p>
      ) : (
        <p className="tp-failed">Tagging failed: {state.message} — nothing may have
          been written; check the tag filter before retrying.</p>
      )}
    </div>
  );
}
