import { AXES, Axis } from "../api/types";

/** What each axis means, in the user's terms rather than the pipeline's. The
 * low→high wording matters: every axis runs easy→hard, so "higher is harder"
 * holds for all four and the sliders can be read the same way. */
export const AXIS_META: Record<Axis, { label: string; low: string; high: string; hint: string }> = {
  legibility: {
    label: "Legibility", low: "clear", high: "blind",
    hint: "How hard the image is to see: sharpness and brightness. High = blurred or dark.",
  },
  rarity: {
    label: "Rarity", low: "common", high: "very rare",
    hint: "How unusual the sample is: rare caption vocabulary, and distance from its neighbours in embedding space.",
  },
  difficulty: {
    label: "Difficulty", low: "routine", high: "hard",
    hint: "Where the model's own understanding is weakest: low image-caption agreement, and captions that disagree with each other.",
  },
  clutter: {
    label: "Clutter", low: "simple", high: "busy",
    hint: "How much is going on: distinct things named across the five captions, and how much their lengths vary.",
  },
};

export type AxisRange = Partial<Record<Axis, [number | null, number | null]>>;

interface Props {
  value: AxisRange;
  onChange: (next: AxisRange) => void;
}

/** Range filters over the difficulty axes.
 *
 * These are the reason the tool can answer "show me the hardest 300 samples"
 * rather than only "show me samples containing a dog", so they sit in the
 * filter panel permanently rather than behind a menu. Content goes in the
 * search box; difficulty goes here. */
export default function AxisFilters({ value, onChange }: Props) {
  const active = AXES.filter((a) => value[a]);

  const set = (axis: Axis, lo: number | null, hi: number | null) => {
    const next = { ...value };
    if (lo == null && hi == null) delete next[axis];
    else next[axis] = [lo, hi];
    onChange(next);
  };

  return (
    <div className="axis-panel">
      {/* No disclosure, deliberately: FilterPanel's header comment promises the
          axes are "no longer behind a disclosure" — this wrapper was the last
          place still breaking that promise, rendering the app's headline
          feature as a dead label on every fresh load. */}
      <div className="axis-panel-title">
        Difficulty axes
        {active.length > 0 && <span className="axis-count">{active.length}</span>}
      </div>
      <div className="axis-rows">
        {AXES.map((axis) => {
          const [lo, hi] = value[axis] ?? [null, null];
          const min = lo ?? 0;
          const max = hi ?? 10;
          const meta = AXIS_META[axis];
          return (
            <div className="axis-row" key={axis}>
              <label className="axis-name" title={meta.hint}>
                {meta.label}
                <span className="axis-scale">{meta.low} → {meta.high}</span>
              </label>
              <div className="axis-sliders">
                <input type="range" min={0} max={10} value={min}
                       aria-label={`${meta.label} minimum`}
                       onChange={(e) => {
                         const v = Number(e.target.value);
                         set(axis, v, Math.max(v, max) === 10 && hi == null ? null : Math.max(v, max));
                       }} />
                <input type="range" min={0} max={10} value={max}
                       aria-label={`${meta.label} maximum`}
                       onChange={(e) => {
                         const v = Number(e.target.value);
                         set(axis, Math.min(v, min) === 0 && lo == null ? null : Math.min(v, min), v);
                       }} />
              </div>
              <span className={`axis-readout ${value[axis] ? "on" : ""}`}>
                {value[axis] ? `${min}–${max}` : "any"}
              </span>
              {value[axis] && (
                <button className="axis-clear" title={`Clear ${meta.label}`}
                        onClick={() => set(axis, null, null)}>×</button>
              )}
            </div>
          );
        })}
      </div>
      <p className="axis-note">
        Scores are percentile ranks over this dataset, so 7 means “harder than
        about 70% of these 8,000 images” — not an absolute measurement, and not
        comparable to another dataset.
      </p>
    </div>
  );
}
