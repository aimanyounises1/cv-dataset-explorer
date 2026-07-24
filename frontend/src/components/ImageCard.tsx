import { Link } from "react-router-dom";
import type { SampleCard } from "../api/types";
import Highlight from "./Highlight";

export default function ImageCard({ sample }: { sample: SampleCard }) {
  const caption = sample.match_caption ?? sample.caption ?? sample.filename;
  return (
    <Link className="card" to={`/samples/${sample.id}`}>
      <img src={sample.thumb_url} alt={caption} loading="lazy" />
      <div className="card-body">
        <div className="card-caption" title={caption}>
          <Highlight text={caption} terms={sample.matched_terms} />
        </div>
        <div className="card-meta">
          <span className="pill">{sample.split}</span>
          {sample.score != null && (
            <span className="pill score" title="Search relevance">
              {sample.score.toFixed(3)}
            </span>
          )}
        </div>
      </div>
    </Link>
  );
}
