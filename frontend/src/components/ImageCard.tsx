import { Link } from "react-router-dom";
import type { SampleCard } from "../api/types";

export default function ImageCard({ sample }: { sample: SampleCard }) {
  return (
    <Link className="card" to={`/samples/${sample.id}`}>
      <img src={sample.thumb_url} alt={sample.caption ?? sample.filename} loading="lazy" />
      <div className="card-body">
        <div className="card-caption">{sample.caption ?? sample.filename}</div>
        <div className="card-meta">
          <span className="pill">{sample.split}</span>
          {sample.score != null && (
            <span className="pill score">{sample.score.toFixed(3)}</span>
          )}
        </div>
      </div>
    </Link>
  );
}
