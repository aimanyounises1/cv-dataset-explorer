import { AXES, Axis, AxisScores } from "../api/types";
import { HEAT_COLOR, heatBand } from "../lib/viz";
import { AXIS_META } from "./AxisFilters";
import AxisSparkline from "./AxisSparkline";

/** Every raw component the scoring pass records, in the unit it was measured
 * in, with one line saying which direction is harder.
 *
 * Direction is the part a reader gets wrong: sharpness, brightness, agreement
 * and consistency are all *inverted* into their axis, so a high raw number
 * there describes an easy sample. Someone who assumes "higher is worse" holds
 * throughout reads half this panel backwards.
 *
 * Keys match analyze.compute_axes. A component that is not listed is skipped
 * rather than guessed at, so adding one server-side shows up as a missing row
 * here instead of an unlabelled number. */
const COMPONENT: Record<string, { label: string; reading: string }> = {
  blur: {
    label: "Sharpness",
    reading: "Variance of the Laplacian: strong second derivatives mean crisp edges. "
           + "Inverted, so a soft or out-of-focus frame is the one that scores hard.",
  },
  luma: {
    label: "Brightness",
    reading: "Mean grey level, 0 black to 1 white. Inverted too: a dark frame is "
           + "harder to see into, and neither end of the scale is punished for glare.",
  },
  word_rarity: {
    label: "Lexical rarity",
    reading: "Mean inverse document frequency of the three rarest words in the five "
           + "captions. Higher means vocabulary few other captions in this corpus use.",
  },
  isolation: {
    label: "Visual isolation",
    reading: "Mean cosine distance to the ten nearest images in embedding space. "
           + "Higher means nothing else in the dataset looks much like this one.",
  },
  agreement: {
    label: "Image-caption agreement",
    reading: "Mean similarity between the image and each of its captions. Inverted: "
           + "it is the low scores that mark a sample the model does not follow.",
  },
  consistency: {
    label: "Caption consistency",
    reading: "Mean similarity between the five captions themselves. Inverted: five "
           + "people describing different things is evidence the image is ambiguous.",
  },
  vocab: {
    label: "Things named",
    reading: "Distinct content words across the five captions. More things named, "
           + "more going on in the frame.",
  },
  length_sd: {
    label: "Length spread",
    reading: "Standard deviation of caption length in words. Uneven lengths mean the "
           + "annotators did not agree on how much there was to say.",
  },
};

/** The machinery behind each axis, named where its number is shown rather than
 * in a document nobody opens. A percentile with no stated provenance is a number
 * the reader has to take on faith, and this tool asks people to act on these. */
const MACHINERY: Record<Axis, string> = {
  legibility: "Sharpness: variance of the Laplacian on the stored 320px thumbnail. "
            + "Brightness: mean grey level. Both inverted, both converted to "
            + "dataset-relative percentiles, then averaged.",
  rarity: "Lexical rarity: caption inverse document frequency over this corpus. "
        + "Embedding isolation: SigLIP 2 kNN distance, k=10, dataset-relative "
        + "percentile. The axis is the mean of the two percentiles.",
  difficulty: "Image-caption agreement and caption consistency: SigLIP 2 cosines, "
            + "both inverted, both dataset-relative percentiles, then averaged.",
  clutter: "Counted from caption text alone — no detector ran — so this measures how "
         + "busy five annotators found the frame, not how many objects are in it.",
};

/** Caveats that are not about the machinery but about how far the number can be
 * trusted at all. Only two axes have earned one, and both are load-bearing. */
const CAVEAT: Partial<Record<Axis, string>> = {
  rarity: "The two components disagree routinely — rare wording over an ordinary "
        + "photograph, or a visually isolated frame described in plain words — and the "
        + "mean hides which one fired. Read both values, not the composite alone.",
  difficulty: "Image-caption agreement is a scorer's opinion (SigLIP 2), not ground "
            + "truth; related scorers align only moderately with human judgment "
            + "(CLIPScore family, tau_b about 0.34 on Flickr8k-CF).",
};

/** Raw values arrive rounded to four decimals server-side. Sub-unit quantities
 * keep all four, because blur separates samples in the third decimal; larger
 * ones get two, and counts stay integers. */
function formatRaw(v: number): string {
  if (Number.isInteger(v)) return String(v);
  return v < 1 ? v.toFixed(4) : v.toFixed(2);
}

function AxisFold({ axis, value, detail }: {
  axis: Axis;
  value?: number | null;
  detail?: Record<string, number | string>;
}) {
  const meta = AXIS_META[axis];
  const comps = Object.entries(detail ?? {}).filter(
    (e): e is [string, number] => typeof e[1] === "number" && e[0] in COMPONENT,
  );

  // Absent, not zero: an unscored axis is missing from these ranks rather than
  // sitting at the bottom of them, and those are different facts about a sample.
  if (value == null) {
    return (
      <div className="axis-fold flat">
        <span className="axis-fold-name">{meta.label}</span>
        <span className="axis-fold-value absent">not scored</span>
      </div>
    );
  }

  const head = (
    <>
      <span className="axis-fold-name">{meta.label}</span>
      {/* Same colour as the bar the sparkline just drew for this axis, because
          it is the same value; the band comes from one function so they cannot
          drift apart. */}
      <span className="axis-fold-value" style={{ color: HEAT_COLOR[heatBand(value)] }}>
        {value}/10
      </span>
    </>
  );

  // Nothing to expand into, so nothing that looks expandable. An empty
  // <details> that opens onto blank rows is worse than a plain line.
  if (comps.length === 0) return <div className="axis-fold flat">{head}</div>;

  return (
    <details className="axis-fold">
      <summary title={`${meta.hint} (${meta.low} → ${meta.high})`}>{head}</summary>
      <div className="axis-fold-body">
        <dl className="axis-comps">
          {comps.map(([key, raw]) => (
            <div key={key}>
              <dt>{COMPONENT[key].label} <b>{formatRaw(raw)}</b></dt>
              <dd>{COMPONENT[key].reading}</dd>
            </div>
          ))}
        </dl>
        {CAVEAT[axis] && <p className="axis-fold-caveat">{CAVEAT[axis]}</p>}
        <p className="axis-fold-foot">{MACHINERY[axis]}</p>
      </div>
    </details>
  );
}

/**
 * The focal sample's four difficulty scores, each one openable down to the raw
 * measurements it was built from.
 *
 * This page used to render the *legend* for the four-bar sparkline without ever
 * rendering the focal sample's own bars — the one screen with room for the full
 * picture showed the least of it. The sparkline gives the profile at a glance and
 * the digits give something citable; the folds exist because a composite
 * percentile you cannot decompose is unfalsifiable, and a reviewer who disagrees
 * with a 9 needs to see which half of it they disagree with.
 */
export default function AxisBreakdown({ axes }: { axes?: AxisScores | null }) {
  const scored = axes ? AXES.filter((a) => axes[a] != null) : [];

  if (!axes || scored.length === 0) {
    return (
      <p className="axis-absent">
        Not scored. The axes come from the analysis pass
        (<span className="mono">python -m app.analyze</span>), and difficulty
        additionally needs image embeddings. This sample is absent from these
        ranks, not at the bottom of them.
      </p>
    );
  }

  return (
    <div className="difficulty">
      <AxisSparkline axes={axes} size="regular" />
      <div className="axis-folds">
        {AXES.map((axis) => (
          <AxisFold key={axis} axis={axis} value={axes[axis]} detail={axes.detail?.[axis]} />
        ))}
      </div>
    </div>
  );
}
