import { Link } from "react-router-dom";
import type { SampleCard } from "../api/types";
import Highlight from "./Highlight";

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
      </div>
    </Link>
  );
}
