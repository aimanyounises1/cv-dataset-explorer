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
  // Spelled out, not abbreviated: a blended-query score has no everyday name,
  // and "cmp"-style shorthand would only pretend it has one.
  composed: "composed",
};

const SCORE_HELP: Record<string, string> = {
  cosine: "Cosine similarity in the embedding space — comparable with other cosines, and not a probability.",
  cosine_adj: "Cosine similarity, ranked with the hubness correction. The number is the "
            + "raw cosine and is comparable with any other; the ordering is not, because "
            + "each image was scored against how close it sits to queries in general.",
  rrf: "Reciprocal-rank fusion weight — derived from ranks, not a similarity.",
  composed: "Cosine to the blended query (steering text averaged with the positive "
          + "reference images), minus half the strongest negative-example cosine. "
          + "Only comparable within this result list.",
};

const PATH_LABEL: Record<string, string> = { keyword: "kw", semantic: "sem" };

interface Props {
  sample: SampleCard;
  scoreBasis?: string | null;
  /** The query and mode that produced this card, and its 1-based place in the
   * result set. The card cannot infer any of the three, and without them the
   * sample page cannot say why the researcher is looking at this frame. */
  query?: string;
  /** Gallery select-mode: clicking toggles membership in the picked set
   * instead of navigating. The card stays a link underneath, so middle-click
   * and "open in new tab" keep working even while picking. */
  selectMode?: boolean;
  selected?: boolean;
  onToggleSelect?: (id: number) => void;
  /** What a drag from this card carries. Defaults to just this sample; the
   * gallery passes the whole picked set when this card is part of it, so a
   * selection drags as one bundle. */
  getDragIds?: (id: number) => number[];
  /** Steering actions for composed search: add this frame as a positive
   * reference ("more like this") or a negative example. Only the gallery
   * passes them — a chat transcript or the similar strip steers nothing. */
  onLike?: (id: number) => void;
  onExclude?: (id: number) => void;
  mode?: string;
  rank?: number;
}

/** Query terms go into one parameter, separated by a pipe rather than a comma,
 * because a matched term is a slice of caption text and can legitimately
 * contain a comma; a pipe cannot appear in a term the tokenizer produced. */
const TERM_SEP = "|";

export default function ImageCard({ sample, scoreBasis, query, mode, rank,
                                    selectMode, selected, onToggleSelect,
                                    getDragIds, onLike, onExclude }: Props) {
  const caption = sample.match_caption ?? sample.caption ?? sample.filename;
  const basis = scoreBasis ?? undefined;

  /** Clicking a result used to throw away every reason it was on screen. The
   * link carries them instead, so the sample page can answer "why am I looking
   * at this?" from the URL alone and the answer survives a bookmark or a paste.
   * URLSearchParams does the escaping, which a query full of spaces and quotes
   * needs. Grids that are not result sets (the similar-images strip, a chat
   * transcript) pass nothing and keep a bare link: what they are is already
   * visible around them, and `rank` is what marks a caller as ranked. */
  const provenance = new URLSearchParams();
  if (rank != null) {
    if (query) {
      provenance.set("src", "search");
      provenance.set("q", query);
      if (mode) provenance.set("mode", mode);
      provenance.set("rank", String(rank));
      // The same three decimals the badge below shows, so a banner reading the
      // URL cannot contradict the card that was clicked.
      // The basis travels with the score, always. A bare `score=0.016` is
      // uninterpretable, and it cannot be recovered from `mode`: `semantic`
      // yields "cosine" or "cosine_adj" depending on whether the hubness
      // penalty loaded for that request -- runtime state, not mode -- and
      // `keyword` yields no basis at all. Deriving the label would relabel an
      // adjusted cosine as a plain one, which is the mistake SCORE_LABEL above
      // exists to prevent.
      if (sample.score != null && basis) {
        provenance.set("score", sample.score.toFixed(3));
        provenance.set("basis", basis);
      }
      const terms = sample.matched_terms?.filter((t) => t.length > 0) ?? [];
      if (terms.length > 0) provenance.set("terms", terms.join(TERM_SEP));
    } else {
      // No query means the ordering came from a filter or an id list, not a
      // ranking, so the search-only parameters would be lies. An empty `q=`
      // would read as "searched for nothing", which is worse than silence.
      provenance.set("src", "browse");
    }
  }
  const search = provenance.toString();

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
    <Link className={`card${selected ? " picked" : ""}`}
          to={search ? `/samples/${sample.id}?${search}` : `/samples/${sample.id}`}
          aria-pressed={selectMode ? Boolean(selected) : undefined}
          onClick={selectMode
            ? (e) => { e.preventDefault(); onToggleSelect?.(sample.id); }
            : undefined}
          /* An anchor drags its URL by default; overriding the payload with
             sample ids is what lets the album shelf receive a drop. The
             custom MIME type keeps a stray drop into a text field from
             pasting an id list. */
          onDragStart={(e) => {
            const ids = getDragIds ? getDragIds(sample.id) : [sample.id];
            e.dataTransfer.setData("application/x-cvde-ids", JSON.stringify(ids));
            e.dataTransfer.effectAllowed = "copy";
          }}>
      <div className="card-media">
        <img src={sample.thumb_url} alt={caption} loading="lazy" />
        {/* Frame number, as on a contact sheet — cite or find a sample without
            opening it. */}
        <span className="frame-no">{sample.id}</span>
        {selected && <span className="pick-mark" aria-hidden="true">✓</span>}
        {/* The images lead; the diagnostics wait. Everything below stays in
            the DOM (tests count it, keyboard focus reveals it) but paints
            only on hover — the mock's rule: hide secondary detail until it
            is requested. The sample page still shows all of it, always. */}
        <div className="card-details">
          {(onLike || onExclude) && (
            <div className="card-actions">
              {onLike && (
                <button type="button"
                        title="More like this — add as a positive reference and re-rank"
                        onClick={(e) => { e.preventDefault(); e.stopPropagation(); onLike(sample.id); }}>
                  ⊕ More like this
                </button>
              )}
              {onExclude && (
                <button type="button"
                        title="Exclude images like this — add as a negative example"
                        onClick={(e) => { e.preventDefault(); e.stopPropagation(); onExclude(sample.id); }}>
                  ⊖
                </button>
              )}
            </div>
          )}
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
          {reasons.length > 0 && (
            <div className="axis-why" title="Templated from the measured components, not generated">
              {reasons.join(" · ")}
            </div>
          )}
        </div>
      </div>
      <div className="card-body">
        <div className="card-caption" title={caption}>
          <Highlight text={caption} terms={sample.matched_terms} />
        </div>
      </div>
    </Link>
  );
}
