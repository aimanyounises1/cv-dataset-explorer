import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import type { SearchMode } from "../api/types";

/** What the score actually is. The same distinction the cards make: an RRF weight
 * and a cosine live on different scales and must never be read against each
 * other, so the figure always arrives carrying its basis.
 *
 * Keyed by the `basis` the card puts in the link, not by `mode`, because mode
 * does not determine basis: `semantic` comes back as `cosine` or `cosine_adj`
 * depending on whether the hubness penalty loaded for that request, which is
 * runtime state. Deriving the label from `semantic` alone relabelled a
 * hubness-adjusted score as a raw cosine — measured, the card read `cos* 0.091`
 * while this line read `cosine 0.091` for the same number, which are two
 * different quantities however alike they look.
 *
 * A Map for the same reason SOURCE below is one: the key arrives from a URL. */
const BASIS = new Map<string, string>([
  ["cosine", "cosine"],
  // Not a cosine. The index returns the value that ranked the result, penalty
  // included, so cos* is the cosine minus a query-bank penalty — the same
  // units measuring a different quantity, and not comparable with a plain
  // cosine in either value or ordering.
  ["cosine_adj", "cos*, hubness-adjusted (not a raw cosine)"],
  ["rrf", "RRF"],
  // The blended-query score. Missing from this map once, which silently
  // voided the WHOLE banner for composed click-throughs — an unknown basis
  // must never erase the provenance around it.
  ["composed", "composed"],
]);
/** Only used when no basis travelled with the score, which is a link built by
 * hand or by an older build. Mode is the weakest honest guess available. */
const BASIS_BY_MODE: Record<SearchMode, string> = {
  hybrid: "RRF", semantic: "cosine", keyword: "keyword score",
};
const MODES: readonly SearchMode[] = ["hybrid", "semantic", "keyword"];

/** Surfaces that can hand a sample over, phrased so the sentence reads. A Map
 * rather than an object literal because the key comes from a URL: `SOURCE.get`
 * cannot return `Object.prototype.constructor` where `SOURCE[src]` would. */
const SOURCE = new Map<string, string>([
  // The two values ImageCard actually writes. Other surfaces hand over bare
  // links on purpose — what they are is already visible around them.
  ["search", "search"],
  ["browse", "browsing"],
]);

const TOKEN = /^[a-z][a-z0-9_-]{0,23}$/;
const RANK = /^\d{1,7}(\/\d{1,7})?$/;
const NUMBER = /^-?\d+(\.\d+)?([eE][-+]?\d+)?$/;

const MAX_Q = 120;
const MAX_TERMS = 6;

/**
 * Why this frame is on screen, read from the URL query.
 *
 * The inspector is reached from several places and, once reached, looked
 * identical from all of them: the one thing a reviewer needs to know first —
 * "this is result 4 of 60 for the query I typed" — was thrown away by the
 * navigation. This puts it back.
 *
 * The provenance travels in the URL, not sessionStorage, because the artifact of
 * this page is a link pasted into a review thread, and a link has to arrive with
 * its own history. The cost is that the query string can say anything, which is
 * why every field is checked and why a single unparseable one voids the whole
 * banner: a provenance line that is half wrong is worse than no line at all.
 */
export default function ProvenanceBanner() {
  const [params] = useSearchParams();
  const [dismissed, setDismissed] = useState(false);

  const src = params.get("src");
  if (dismissed || src === null || !TOKEN.test(src)) return null;

  const modeParam = params.get("mode");
  if (modeParam !== null && !MODES.includes(modeParam as SearchMode)) return null;
  const mode = modeParam as SearchMode | null;

  const rank = params.get("rank");
  if (rank !== null && !RANK.test(rank)) return null;
  if (rank !== null && rank.includes("/")) {
    // A position past the end of its own list means the link was built wrong,
    // and this figure is the whole reason the line exists.
    const [pos, total] = rank.split("/").map(Number);
    if (pos < 1 || pos > total) return null;
  }

  const score = params.get("score");
  if (score !== null && !NUMBER.test(score)) return null;

  const basis = params.get("basis");
  if (basis !== null && !BASIS.has(basis)) return null;

  const raw = (params.get("q") ?? "").trim();
  const q = raw.length > MAX_Q ? `${raw.slice(0, MAX_Q - 1)}…` : raw;
  // Split on a pipe, which is what ImageCard joins with: a matched term is a
  // slice of caption text and can contain a comma, so splitting on a comma both
  // failed to split at all (the line printed `matched: dog|running` verbatim)
  // and would have torn one term in two the day it did.
  const terms = (params.get("terms") ?? "").split("|").map((t) => t.trim()).filter(Boolean);
  const shown = terms.slice(0, MAX_TERMS);

  const figures: string[] = [];
  if (rank !== null) figures.push(`rank ${rank}`);
  if (score !== null) {
    const label = (basis && BASIS.get(basis)) || (mode ? BASIS_BY_MODE[mode] : "score");
    figures.push(`${label} ${Number(score).toFixed(3)}`);
  }
  const where = SOURCE.get(src) ?? src.replace(/[_-]/g, " ");

  return (
    <div className="provenance">
      <span className="provenance-text">
        Surfaced by {q ? `${mode ? `${mode} ` : ""}${where} ` : where}
        {q && <span className="provenance-q">“{q}”</span>}
        {figures.length > 0 && (
          <span className="provenance-fig"> — {figures.join(" · ")}</span>
        )}
        {shown.length > 0 && (
          <span>
            {figures.length > 0 ? " · " : " — "}matched: {shown.join(", ")}
            {terms.length > shown.length && ` +${terms.length - shown.length} more`}
          </span>
        )}
      </span>
      <button className="provenance-x" onClick={() => setDismissed(true)}
              aria-label="Dismiss provenance" title="Dismiss">×</button>
    </div>
  );
}
