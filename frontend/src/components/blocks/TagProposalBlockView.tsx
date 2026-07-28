import { useEffect, useState } from "react";
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
 */
export default function TagProposalBlockView({ block }: { block: TagProposalBlock }) {
  const [thumbs, setThumbs] = useState<SampleCard[] | null>(null);
  const [state, setState] = useState<
    { phase: "open" } | { phase: "busy" }
    | { phase: "approved"; tagged: number } | { phase: "rejected" }
    | { phase: "failed"; message: string }
  >({ phase: "open" });

  const ids = block.sample_ids;
  const idKey = ids.join(",");
  useEffect(() => {
    if (ids.length === 0) { setThumbs([]); return; }
    const ctrl = new AbortController();
    api.listSamples({ ids: ids.slice(0, 12).join(","), per_page: 12 }, ctrl.signal)
      .then((list) => setThumbs(list.items))
      .catch(() => { if (!ctrl.signal.aborted) setThumbs([]); });
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idKey]);

  const approve = async () => {
    setState({ phase: "busy" });
    try {
      const res = await api.bulkTag(ids, block.tag);
      // The trail's witness for a client-side mutation is the client.
      api.recordActivity("tag_approval",
        { tag: res.tag, tagged: res.tagged, proposed: ids.length }).catch(() => {});
      setState({ phase: "approved", tagged: res.tagged });
    } catch (e) {
      setState({ phase: "failed", message: e instanceof Error ? e.message : String(e) });
    }
  };

  return (
    <div className="vblock-body tag-proposal">
      {block.reason && <p className="tp-reason">{block.reason}</p>}
      <div className="tp-thumbs">
        {(thumbs ?? []).map((s) => (
          <Link key={s.id} to={`/samples/${s.id}`} title={`#${s.id}`}>
            <img src={s.thumb_url} alt={`sample ${s.id}`} loading="lazy" />
          </Link>
        ))}
        {ids.length > 12 && <span className="tp-more">+{ids.length - 12} more</span>}
      </div>
      {block.missing && block.missing.length > 0 && (
        <p className="tp-missing">
          {block.missing.length} requested id{block.missing.length > 1 ? "s" : ""} do
          not exist and were dropped from this proposal.
        </p>
      )}
      {state.phase === "open" || state.phase === "busy" ? (
        <div className="tp-actions">
          <button className="primary" onClick={approve} disabled={state.phase === "busy"}>
            {state.phase === "busy" ? "Tagging…" : `Approve — tag ${ids.length} as ‘${block.tag}’`}
          </button>
          <button className="ghost" onClick={() => setState({ phase: "rejected" })}
                  disabled={state.phase === "busy"}>
            Reject
          </button>
        </div>
      ) : state.phase === "approved" ? (
        <p className="tp-done">
          Tagged {state.tagged} sample{state.tagged === 1 ? "" : "s"} as ‘{block.tag}’.{" "}
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
