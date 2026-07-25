import { Link } from "react-router-dom";
import { AXES, SampleCard } from "../api/types";
import { AXIS_META } from "./AxisFilters";
import Highlight from "./Highlight";

/** Short labels, so four axes fit under a thumbnail without wrapping. */
const AXIS_ABBR: Record<string, string> = {
  legibility: "leg", rarity: "rar", difficulty: "dif", clutter: "clt",
};

/** Colour by how much attention the score is asking for. Every axis runs
 * easy→hard, so one scale serves all four. */
function heat(v: number): string {
  if (v >= 9) return "hot";
  if (v >= 7) return "warm";
  if (v >= 4) return "mid";
  return "cool";
}

/** A score is only interpretable next to what produced it: a text-image cosine
 * and an RRF sum live on different scales and must never be read against each
 * other, so the value always carries its basis. */
const SCORE_LABEL: Record<string, string> = { cosine: "cos", rrf: "rrf" };

const SCORE_HELP: Record<string, string> = {
  cosine: "Cosine similarity in the embedding space — comparable with other cosines, and not a probability.",
  rrf: "Reciprocal-rank fusion weight — derived from ranks, not a similarity.",
};

const PATH_LABEL: Record<string, string> = { keyword: "kw", semantic: "sem" };

interface Props {
  sample: SampleCard;
  scoreBasis?: string | null;
}

export default function ImageCard({ sample, scoreBasis }: Props) {
  const caption = sample.match_caption ?? sample.caption ?? sample.filename;
  const basis = scoreBasis ?? undefined;

  /** Reasons worth showing: only for axes actually scoring hard, deduplicated,
   * because the same phrase can be earned on two axes at once. */
  const reasons: string[] = [];
  for (const axis of AXES) {
    const v = sample.axes?.[axis];
    const why = sample.axes?.detail?.[axis]?.why;
    if (v != null && v >= 7 && typeof why === "string") {
      for (const phrase of why.split(", ")) {
        if (!reasons.includes(phrase)) reasons.push(phrase);
      }
    }
  }
  return (
    <Link className="card" to={`/samples/${sample.id}`}>
      <div className="card-media">
        <img src={sample.thumb_url} alt={caption} loading="lazy" />
        {/* Frame number, as on a contact sheet — cite or find a sample without
            opening it. */}
        <span className="frame-no">{sample.id}</span>
      </div>
      <div className="card-body">
        <div className="card-caption" title={caption}>
          <Highlight text={caption} terms={sample.matched_terms} />
        </div>
        {/* Evidence strip: provenance, then the paths that retrieved this
            frame and where it placed in each, then the score. Same order
            every time, so it can be read at a glance across a grid. */}
        <div className="card-evidence">
          <span className="ev">{sample.split}</span>
          {sample.match_paths?.map((p) => (
            <span key={p.path} className="ev ev-path"
                  title={`Retrieved by ${p.path} search at rank ${p.rank}`}>
              {PATH_LABEL[p.path] ?? p.path} {p.rank}
            </span>
          ))}
          {sample.score != null && basis && (
            <span className="ev ev-score" title={SCORE_HELP[basis] ?? "Search relevance"}>
              {SCORE_LABEL[basis] ?? basis} {sample.score.toFixed(3)}
            </span>
          )}
        </div>
        {/* Difficulty axes. The strip above answers "why did the search return
            this?"; this row answers "why is this one worth my time?" */}
        {sample.axes && (
          <div className="axis-badges">
            {AXES.map((axis) => {
              const v = sample.axes?.[axis];
              if (v == null) return null;
              const meta = AXIS_META[axis];
              const comps = sample.axes?.detail?.[axis] ?? {};
              // Raw values in the tooltip: the score is a rank, and a rank you
              // cannot trace back to a measurement is just an assertion.
              const measured = Object.entries(comps)
                .filter(([k]) => k !== "why")
                .map(([k, val]) => `${k} ${val}`)
                .join(", ");
              return (
                <span key={axis} className={`axis-badge ${heat(v)}`}
                      title={`${meta.label} ${v}/10 — ${meta.low} → ${meta.high}. ${meta.hint}`
                             + (measured ? `\nMeasured: ${measured}` : "")}>
                  {AXIS_ABBR[axis]} <b>{v}</b>
                </span>
              );
            })}
          </div>
        )}
        {/* One templated line naming what actually makes this sample hard. */}
        {reasons.length > 0 && (
          <div className="axis-why" title="Templated from the measured components, not generated">
            {reasons.join(" · ")}
          </div>
        )}
      </div>
    </Link>
  );
}
