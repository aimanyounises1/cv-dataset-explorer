import { useEffect, useRef, useState } from "react";
import { AXES, SearchMode } from "../api/types";
import { AXIS_META } from "./AxisFilters";

const MODES: SearchMode[] = ["hybrid", "semantic", "keyword"];

/** One label per sort key; "" is whatever ranking the active query produced. */
const SORT_OPTIONS = [
  { value: "", label: "Relevance" },
  ...AXES.flatMap((a) => [
    { value: `${a}_desc`, label: `${AXIS_META[a].label} — hardest first` },
    { value: `${a}_asc`, label: `${AXIS_META[a].label} — easiest first` },
  ]),
];

/** The order dial as a listbox the design system owns end to end — a native
 * <select> popup takes its colours and type from the OS, and this is the one
 * control in the command bar that opened a foreign-looking menu. The trigger
 * mirrors its value into `data-value` so the DOM states the choice as plainly
 * as the old select's `.value` did. */
function SortListbox({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const btnRef = useRef<HTMLButtonElement | null>(null);
  const listRef = useRef<HTMLUListElement | null>(null);

  const openList = () => {
    setActive(Math.max(0, SORT_OPTIONS.findIndex((o) => o.value === value)));
    setOpen(true);
  };
  // The list takes focus when it opens so arrow keys land on it, not the page.
  useEffect(() => { if (open) listRef.current?.focus(); }, [open]);
  useEffect(() => {
    if (open) {
      document.getElementById(`sort-opt-${active}`)
        ?.scrollIntoView({ block: "nearest" });
    }
  }, [open, active]);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [open]);

  const pick = (i: number) => {
    onChange(SORT_OPTIONS[i].value);
    setOpen(false);
    btnRef.current?.focus();
  };

  const onListKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setActive((i) => Math.min(SORT_OPTIONS.length - 1, i + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive((i) => Math.max(0, i - 1)); }
    else if (e.key === "Home") { e.preventDefault(); setActive(0); }
    else if (e.key === "End") { e.preventDefault(); setActive(SORT_OPTIONS.length - 1); }
    else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(active); }
    else if (e.key === "Escape") {
      // Stops here: Escape with the list open dismisses the list, and only a
      // second Escape reaches the panel's own document-level close.
      e.stopPropagation();
      setOpen(false);
      btnRef.current?.focus();
    } else if (e.key === "Tab") setOpen(false);
  };

  const current = SORT_OPTIONS.find((o) => o.value === value) ?? SORT_OPTIONS[0];
  return (
    <div className="sort-listbox" ref={wrapRef}>
      <button type="button" ref={btnRef} className="sort-trigger"
              data-value={value}
              aria-haspopup="listbox" aria-expanded={open} aria-label="Sort results"
              onClick={() => (open ? setOpen(false) : openList())}
              onKeyDown={(e) => {
                if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); openList(); }
              }}>
        <span>{current.label}</span>
      </button>
      {open && (
        <ul className="sort-options" role="listbox" aria-label="Sort results"
            tabIndex={-1} ref={listRef}
            aria-activedescendant={`sort-opt-${active}`}
            onKeyDown={onListKey}>
          {SORT_OPTIONS.map((o, i) => (
            <li key={o.value || "relevance"} id={`sort-opt-${i}`} role="option"
                aria-selected={o.value === value}
                className={i === active ? "active" : ""}
                onMouseEnter={() => setActive(i)}
                onClick={() => pick(i)}>
              {o.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

interface Props {
  mode: SearchMode;
  sort: string;
  density: string;
  /** Density keys in display order — owned by the gallery, whose grid is the
   * thing they size. */
  densities: string[];
  providerName: string;
  /** "" restores the default (hybrid mode / relevance order). */
  onMode: (mode: string) => void;
  onSort: (sort: string) => void;
  onDensity: (density: string) => void;
  /** When set, the mode pills are inert and this text says why. */
  modeLockedReason?: string | null;
}

/** One command bar: the query's dials live behind a single quiet button — the
 * spec's own rule (advanced controls hidden until requested). The dot marks a
 * non-default dial, so collapsing never silently hides a live choice. The
 * panel behaves as the popover it looks like: Escape and any press outside
 * both dismiss it. */
export default function SearchSettings({ mode, sort, density, densities,
                                         providerName, onMode, onSort,
                                         onDensity, modeLockedReason }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDetailsElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      // An Escape aimed at something else — the search box dismissing its own
      // recent-queries list — may still close this popover, but it must not
      // steal the focus that Escape belonged to.
      const inside = ref.current?.contains(document.activeElement);
      setOpen(false);
      if (inside) (ref.current?.querySelector("summary") as HTMLElement | null)?.focus();
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <details ref={ref} className="search-settings" open={open}
             onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary title="Mode, ordering and density">
        Search settings
        {(mode !== "hybrid" || sort !== "") && (
          <span className="settings-dot" aria-hidden="true" />
        )}
      </summary>
      <div className="search-settings-pop">
        <div className="setting-row">
          <span className="eyebrow">Mode</span>
          <div className="mode-toggle" role="group" aria-label="Search mode">
            {MODES.map((m) => (
              <button
                key={m}
                className={mode === m ? "active" : ""}
                aria-pressed={mode === m}
                disabled={Boolean(modeLockedReason)}
                onClick={() => onMode(m === "hybrid" ? "" : m)}
                title={
                  modeLockedReason
                  ?? (m === "semantic" ? `${providerName} text-to-image similarity`
                    : m === "keyword" ? "BM25 full-text over captions + VLM tags (Porter-stemmed)"
                    : "Reciprocal-rank fusion of semantic + keyword")
                }
              >
                {m}
              </button>
            ))}
          </div>
        </div>
        <div className="setting-row">
          <span className="eyebrow">Order</span>
          <SortListbox value={sort} onChange={onSort} />
        </div>
        <div className="setting-row">
          <span className="eyebrow">Size</span>
          <div className="density-group" role="group" aria-label="Thumbnail size">
            {densities.map((d) => (
              <button key={d} className={density === d ? "active" : ""}
                      aria-pressed={density === d}
                      onClick={() => onDensity(d)}>{d}</button>
            ))}
          </div>
        </div>
      </div>
    </details>
  );
}
