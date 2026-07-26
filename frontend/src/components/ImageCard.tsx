import { Link } from "react-router-dom";
import { AXES, Axis, SampleCard } from "../api/types";
import { heatBand } from "../lib/viz";
import AxisSparkline from "./AxisSparkline";
import { AXIS_META } from "./AxisFilters";
import Highlight from "./Highlight";

/** Short labels, so an axis name fits under a thumbnail without wrapping. */
const AXIS_ABBR: Record<string, string> = {
  legibility: "leg", rarity: "rar", difficulty: "dif", clutter: "clt",
};

/** A score is only interpretable next to what produced it: a text-image cosine
 * and an RRF sum live on different scales and must never be read against each
 * other, so the value always carries its basis. */
const SCORE_LABEL: Record<string, string> = {
  cosine: "cos", rrf: "rrf",
  // Starred rather than renamed: the number really is a plain cosine — the
  // hubness correction subtracts its penalty before ranking and the raw
  // similarity is what comes back — so it stays comparable with every other
  // `cos` in the app. Only the ordering differs, and the star is what says so.
  cosine_adj: "cos*",
  // Not a cosine at all — a log-likelihood under the image's trained speaker
  // model — so it gets its own label rather than borrowing one it would lie in.
  prism_ll: "fit",
};

const SCORE_HELP: Record<string, string> = {
  cosine: "Cosine similarity in the embedding space — comparable with other cosines, and not a probability.",
  cosine_adj: "Cosine similarity, ranked with the hubness correction. The number is the "
            + "raw cosine and is comparable with any other; the ordering is not, because "
            + "each image was scored against how close it sits to queries in general.",
  rrf: "Reciprocal-rank fusion weight — derived from ranks, not a similarity.",
  prism_ll: "Log-likelihood of the query under this image's trained speaker model "
          + "(PRISM, boosted mode). Higher is better and the ordering is what was "
          + "measured; the value is only comparable within this result list — it is "
          + "not a cosine and not a probability.",
};

const PATH_LABEL: Record<string, string> = { keyword: "kw", semantic: "sem", boosted: "boost" };

interface Props {
  sample: SampleCard;
  scoreBasis?: string | null;
}

export default function ImageCard({ sample, scoreBasis }: Props) {
  const caption = sample.match_caption ?? sample.caption ?? sample.filename;
  const basis = scoreBasis ?? undefined;

  /** The one axis worth naming in text. The sparkline shows all four; spelling
   * out the highest gives the eye somewhere to land without reading the chart. */
  let hardest: { axis: Axis; v: number } | null = null;
  for (const axis of AXES) {
    const v = sample.axes?.[axis];
    if (v != null && (hardest === null || v > hardest.v)) hardest = { axis, v };
  }
  if (hardest !== null && hardest.v < 7) hardest = null;   // nothing notable to lead with

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
            this?"; this row answers "why is this one worth my time?"
            One sparkline instead of four numbers: the profile is the signal, and
            exact values live in the tooltip and on the detail page. */}
        {sample.axes && (
          <div className="card-axis-row">
            <AxisSparkline axes={sample.axes} />
            {hardest && (
              <span className={`axis-lead ${heatBand(hardest.v)}`}
                    title={`Highest axis: ${AXIS_META[hardest.axis].label} ${hardest.v}/10 `
                           + `(${AXIS_META[hardest.axis].low} → ${AXIS_META[hardest.axis].high}). `
                           + AXIS_META[hardest.axis].hint}>
                {AXIS_ABBR[hardest.axis]} {hardest.v}
              </span>
            )}
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
