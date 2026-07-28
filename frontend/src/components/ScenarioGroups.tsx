import { Link } from "react-router-dom";
import { ScenarioGroup, ScenarioResponse } from "../api/types";

/** How many members of a group get a face in the strip. Enough to see what the
 * group IS at a glance; the rest are counted, and "Open N" shows them all.
 * Seven, so the faces plus the "+N" chip fill the strip's 4-column grid
 * exactly rather than leaving a ragged last row. */
export const GROUP_THUMBS = 7;

/** The server refuses to group fewer than this — grouping is only offered when
 * the groups would average eight members (backend `SCENARIO_MIN_RESULTS`). */
export const GROUP_MIN_RESULTS = 8;

/** One grouping answer, kept whole. `degraded`/`message` are optional on the
 * wire (an older backend omits them), so they are read defensively rather than
 * assumed — and a missing capability is reported, never rendered as emptiness. */
export interface GroupsAnswer {
  groups: ScenarioGroup[];
  basis: string;
  degraded: boolean;
  message: string | null;
}

export const readGroupsAnswer = (r: ScenarioResponse): GroupsAnswer => {
  const raw = r as unknown as Record<string, unknown>;
  return {
    groups: r.groups ?? [],
    basis: r.basis ?? "",
    degraded: raw.degraded === true,
    message: typeof raw.message === "string" && raw.message ? raw.message : null,
  };
};

/**
 * Grouped exploration: the same result set, proposed as at most three
 * explainable groups.
 *
 * Proposals, not state — nothing here is kept until "Save as album" makes a
 * group durable. The panel reports whichever state the answer is in, and the
 * order of those states is the order of how badly they need saying: a failed
 * or unavailable grouping speaks in the server's own words, a refusal to group
 * states its rule, and only then are there groups. A swallowed failure
 * rendering as "no groups" would be indistinguishable from "your results have
 * no structure", which is the one thing this panel must never say by accident.
 */
export default function ScenarioGroups({
  resultCount, hasMore, answer, busy, thumbs, onBack, onSaveGroup,
}: {
  /** How many results are on screen — what the proposals are over. */
  resultCount: number;
  hasMore: boolean;
  /** null until the grouping request has answered. */
  answer: GroupsAnswer | null;
  busy: boolean;
  /** sample id → thumb url, filled in by the caller as the faces arrive. */
  thumbs: Record<number, string>;
  onBack: () => void;
  onSaveGroup: (gr: ScenarioGroup) => void;
}) {
  return (
    <section className="groups-panel" aria-label="Grouped exploration">
      <div className="groups-head">
        <span className="eyebrow">Proposed groups</span>
        {/* Says what was grouped, not what is visible. The server clusters the
            ranking to its own depth, so a group can be larger than the page —
            claiming "the results on screen" while showing a group of 118 over
            60 rendered cards was the panel contradicting itself. */}
        <span className="groups-note">
          Proposals over this ranking, deeper than the {resultCount}
          {hasMore ? "+" : ""} shown
          {answer?.basis ? ` · ${answer.basis}` : ""} · nothing is
          kept until you save a group as an album
        </span>
        <button className="link-btn groups-back" onClick={onBack}>
          ← back to ranked results
        </button>
      </div>

      {busy && <div className="loading">Grouping these results…</div>}

      {/* Honest states, in order of how badly they need saying: a failed or
          unavailable grouping speaks in the server's own words; a refusal
          to group states its rule; only then are there groups. */}
      {!busy && answer?.degraded && (
        <div className="notice">{answer.message
          ?? "Grouping is unavailable and gave no reason."}</div>
      )}
      {!busy && answer && !answer.degraded && answer.groups.length === 0 && (
        <div className="empty">
          {answer.message
            ?? "No group over these results shares an attribute, tag or "
               + "caption term with a majority of its members."}
        </div>
      )}

      {!busy && answer && answer.groups.length > 0 && (
        <div className="scenario-row">
          {/* Keyed by membership, not label: two groups can honestly both be
              called "mixed", and identical keys would make React reuse the
              wrong card's state. */}
          {answer.groups.map((gr) => (
            <div className="scenario-card"
                 key={`${gr.label}:${gr.sample_ids[0] ?? "empty"}`}>
              <div className="scenario-head">
                <strong>{gr.label}</strong>
                <span className="scenario-count">{gr.count}</span>
              </div>
              {/* The measured counts behind the label, verbatim from the
                  server — the label is a summary, this is the evidence. */}
              <div className="scenario-evidence">{gr.evidence}</div>
              <div className="scenario-thumbs" aria-hidden="true">
                {gr.sample_ids.slice(0, GROUP_THUMBS).map((id) => (
                  thumbs[id]
                    ? <img key={id} src={thumbs[id]} alt="" loading="lazy" />
                    : <span key={id} className="scenario-thumb-ph" />
                ))}
                {gr.count > GROUP_THUMBS && (
                  <span className="scenario-thumb-more">
                    +{gr.count - GROUP_THUMBS}
                  </span>
                )}
              </div>
              <div className="scenario-actions">
                {/* The existing id-list slice: a group opens as a set the
                    URL already knows how to carry. */}
                <Link className="open-all" to={`/?ids=${gr.sample_ids.join(",")}`}>
                  Open {gr.count} →
                </Link>
                <button className="ghost" onClick={() => onSaveGroup(gr)}>
                  Save as album
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
