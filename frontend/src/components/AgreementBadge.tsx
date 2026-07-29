/** SigLIP image-caption agreement, as the number and the band it falls in.
 *
 * The bands are the ones the sample page has always drawn: below 0.05 the
 * caption is suspect, 0.05–0.09 is the middle, above that it fits. They are
 * cosines from this corpus's own scorer, so they are a threshold on an opinion
 * — the badge shows the raw number beside the colour rather than only the
 * verdict, because a 0.049 and a 0.011 are not the same claim.
 *
 * Colour is never the only encoding: the value is printed. */
export default function AgreementBadge({ value }: { value?: number | null }) {
  if (value == null) return null;
  const cls = value < 0.05 ? "warn" : value < 0.09 ? "mid" : "ok";
  return (
    <span className={`agree-badge ${cls}`}
          title="SigLIP image-caption agreement (low = suspect caption)">
      {value.toFixed(3)}
    </span>
  );
}
