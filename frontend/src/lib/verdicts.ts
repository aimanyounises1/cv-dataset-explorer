/**
 * The review vocabulary, in one place.
 *
 * These names are not new. Quality's review guide has documented them since the
 * review workflow shipped — "record your call as a tag on the sample page" —
 * and `TagEditor`'s placeholder offers one as its example. What was missing was
 * a control: the convention existed, and recording a call meant typing the
 * string correctly by hand, on another page, one sample at a time.
 *
 * A verdict is an ordinary tag under a reserved prefix, which is what makes it
 * useful rather than decorative: `?tag=verdict:caption-error` filters the
 * gallery to a review session, and every export path carries tags already. No
 * new column, no new table, no migration.
 *
 * They are mutually exclusive — a sample has one call recorded against it, and
 * choosing a second replaces the first.
 */
export const VERDICT_PREFIX = "verdict:";

export const VERDICTS: { value: string; label: string; hint: string }[] = [
  { value: "ok", label: "ok",
    hint: "The captions describe the image. Nothing to fix." },
  { value: "caption-error", label: "caption error",
    hint: "A caption is wrong about the image — an annotation error." },
  { value: "scorer-error", label: "scorer error",
    hint: "The caption is fine; the low agreement score is the model's mistake." },
  { value: "ambiguous", label: "ambiguous",
    hint: "The image genuinely supports more than one reading." },
  { value: "duplicate", label: "duplicate",
    hint: "This image repeats another in the corpus." },
];

/** The tag a sample carries for a recorded call, if any. */
export function recordedVerdict(tags: string[]): string | null {
  return tags.find((t) => t.startsWith(VERDICT_PREFIX)) ?? null;
}
