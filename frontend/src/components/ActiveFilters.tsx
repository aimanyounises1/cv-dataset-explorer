import "../styles/active-filters.css";

export interface ActiveFilterChip {
  key: string;        // stable id, e.g. "split" or "difficulty"
  label: string;      // e.g. "Split" / "Difficulty"
  value: string;      // e.g. "validation" / "8-10"
}

export interface ActiveFiltersProps {
  chips: ActiveFilterChip[];
  onRemove: (key: string) => void;
  onClearAll: () => void;
}

/** Values the interface treats as numbers: counts, scores, and axis ranges such
 * as "8-10". The visual system reserves monospace for figures the tool asserts,
 * so a split name like "validation" must not be set in it — matching a chip's
 * value against this is cheaper than asking every caller to declare a type. */
const NUMERIC_VALUE = /^[\d.,\s]+(?:[-–—][\d.,\s]+)?%?$/;

/** The constraints currently narrowing the result set, one removable chip each.
 *
 * Filters live in four separate controls (search box, split/tag selects, the
 * axis sliders behind a collapsed panel), so a user could previously only infer
 * an active constraint from a result count that looked wrong. Showing them in
 * one row makes the query legible while it is being built, and makes any single
 * constraint removable without hunting for the control that set it. */
export default function ActiveFilters(props: ActiveFiltersProps): JSX.Element | null {
  const { chips, onRemove, onClearAll } = props;
  if (chips.length === 0) return null;

  return (
    <div className="active-filters" role="group" aria-label="Active filters">
      <span className="active-filters-label">Filtering by</span>
      {chips.map((chip) => (
        <span className="filter-chip" key={chip.key}>
          <span className="filter-chip-label">{chip.label}</span>
          <span className={NUMERIC_VALUE.test(chip.value) ? "filter-chip-value num" : "filter-chip-value"}>
            {chip.value}
          </span>
          <button
            type="button"
            className="filter-chip-remove"
            aria-label={`Remove ${chip.label} filter`}
            title={`Remove ${chip.label} filter`}
            onClick={() => onRemove(chip.key)}
          >
            ×
          </button>
        </span>
      ))}
      {/* One chip is faster to dismiss with its own ×; the shortcut only earns
          its place once there is more than one thing to undo. */}
      {chips.length > 1 && (
        <button type="button" className="filter-clear-all" onClick={onClearAll}>
          Clear all
        </button>
      )}
    </div>
  );
}
