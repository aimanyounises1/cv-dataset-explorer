import { Fragment, KeyboardEvent, useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

/**
 * The rail's own control vocabulary: segmented pills, a listbox, a dual-range.
 *
 * Native <select> and stacked <input type=range> answered the question but in
 * the browser's voice, not the product's — the one part of an otherwise
 * composed rail that looked borrowed. These three keep the native elements'
 * keyboard grammar (arrows, Home/End, Escape) and ARIA roles, and take only
 * their pixels. Nothing here owns state: value in, callback out, like the
 * elements they replace.
 */

export interface PickOption {
  value: string;
  label: string;
  /** Right-aligned figure (a count) — monospace where it lands. */
  figure?: string;
  /** Section header drawn above this option when it differs from the previous
   * option's group — <optgroup>, without the native chrome. */
  group?: string;
  disabled?: boolean;
}

/** Radiogroup drawn as pills, for a choice small enough to show whole. A
 * dropdown hides its options; four splits fit in the rail, so showing all of
 * them costs nothing and saves a click. Roving tabindex, arrows to move. */
export function Segmented({ value, options, onChange, label }: {
  value: string;
  /** `figure` is an optional second line — how many rows this option selects.
   * A second LINE rather than a suffix because four segments share a 219px
   * column, and `.segmented-opt` ellipsises: "train 6,000" would render as
   * "train 6…" and a truncated count is worse than none. */
  options: { value: string; label: string; figure?: string }[];
  onChange: (v: string) => void;
  label: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const idx = Math.max(0, options.findIndex((o) => o.value === value));
  const move = (d: number) => {
    const next = (idx + d + options.length) % options.length;
    onChange(options[next].value);
    refs.current[next]?.focus();
  };
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o, i) => (
        <button
          key={o.value || "__all"}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          className={`segmented-opt${o.value === value ? " on" : ""}`}
          tabIndex={o.value === value ? 0 : -1}
          ref={(el) => { refs.current[i] = el; }}
          onClick={() => onChange(o.value)}
          onKeyDown={(e) => {
            if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); move(1); }
            if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); move(-1); }
          }}
        >
          <span className="segmented-label">{o.label}</span>
          {o.figure && <span className="segmented-figure">{o.figure}</span>}
        </button>
      ))}
    </div>
  );
}

/** Select-only combobox (APG pattern): focus stays on the trigger, arrows move
 * an active row, Enter picks, Escape closes. The popup positions inside the
 * rail column, so it scrolls with the panel it belongs to. */
export function Listbox({ trigger, options, onPick, label, selected, inline }: {
  /** Text on the closed control — the current value, or what picking adds. */
  trigger: string;
  options: PickOption[];
  onPick: (v: string) => void;
  label: string;
  /** Marked aria-selected; "" is a real value (the "All …" row). */
  selected?: string;
  /** Size to the widest option instead of filling the column.
   *
   * The rail stacks its controls in a fixed-width column, so the default is to
   * fill it. A toolbar is a row, and there the same default made the trigger
   * take every pixel its flex line had: measured at 319px to carry the word
   * "Difficulty", which pushed the legend beside it onto three lines. */
  inline?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const popRef = useRef<HTMLDivElement>(null);
  const id = useId();

  /* Where the popup sits, in viewport coordinates.
   *
   * The list used to be an absolutely positioned child of the trigger, which
   * put it inside `.rail-scroll`'s `overflow-y: auto`. z-index cannot escape a
   * clipping ancestor, so the attribute picker rendered 18 options and showed
   * ONE: the popup ran to y=921 against a clip edge at y=723. It is now
   * portalled to <body> and positioned from the trigger's rect, which is also
   * why `active`'s scrollIntoView no longer scrolls the rail out from under
   * the reader. */
  const [rect, setRect] = useState<
    { top: number; left: number; width: number; maxH: number } | null>(null);

  /* Below the trigger when there is room, above it when there is not.
   *
   * The rail's filter controls sit near the bottom of a 100vh column, so
   * "always downward" put the list off the bottom of the window — trading a
   * clipped popup for one that runs past y=1000. Whichever side is chosen, the
   * popup is capped to the space actually available on it. */
  const place = useCallback(() => {
    const r = rootRef.current?.getBoundingClientRect();
    if (!r) return;
    const GAP = 4;
    const below = window.innerHeight - r.bottom - GAP;
    const above = r.top - GAP;
    const useAbove = below < Math.min(220, above);
    const maxH = Math.max(120, Math.min(360, useAbove ? above : below));
    setRect({
      top: useAbove ? r.top - GAP - maxH : r.bottom + GAP,
      left: r.left, width: r.width, maxH,
    });
  }, []);

  const enabled = (i: number) => !options[i]?.disabled;
  const firstEnabled = options.findIndex((_, i) => enabled(i));

  const openAt = () => {
    const sel = options.findIndex((o) => o.value === selected);
    setActive(sel >= 0 && enabled(sel) ? sel : Math.max(0, firstEnabled));
    place();
    setOpen(true);
  };

  // Outside pointerdown closes; blur alone misses clicks on non-focusable text.
  // The popup is portalled, so it is NOT inside rootRef any more and has to be
  // asked separately — without this, picking an option counts as an outside
  // click and closes the list before the pick lands.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!rootRef.current?.contains(t) && !popRef.current?.contains(t)) setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [open]);

  // A fixed popup does not travel with its trigger, so it is re-placed while
  // open. Capture phase: the rail is the scroller, not the window.
  useLayoutEffect(() => {
    if (!open) return undefined;
    const onMove = () => place();
    window.addEventListener("scroll", onMove, true);
    window.addEventListener("resize", onMove);
    return () => {
      window.removeEventListener("scroll", onMove, true);
      window.removeEventListener("resize", onMove);
    };
  }, [open, place]);

  useEffect(() => {
    if (!open) return;
    document.getElementById(`${id}-opt-${active}`)
      ?.scrollIntoView({ block: "nearest" });
  }, [open, active, id]);

  const step = (from: number, d: number): number => {
    let i = from;
    do { i += d; } while (i >= 0 && i < options.length && !enabled(i));
    return i >= 0 && i < options.length ? i : from;
  };

  // A listbox owns only options and groups. The <li> wrapper this markup used
  // to have is neither, so every option fell out of the accessibility mapping
  // and the control exposed nothing to pick. Options arrive grouped in runs;
  // each run becomes a role="group" named by the header it already draws, and
  // the options are the listbox's own children. `from` keeps the flat index
  // that ids, aria-activedescendant and the arrow keys are stated in.
  const runs: { group?: string; from: number; items: PickOption[] }[] = [];
  options.forEach((o, i) => {
    const last = runs[runs.length - 1];
    if (last && last.group === o.group) last.items.push(o);
    else runs.push({ group: o.group, from: i, items: [o] });
  });

  const onKey = (e: KeyboardEvent) => {
    if (!open) {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(e.key)) {
        e.preventDefault();
        openAt();
      }
      return;
    }
    if (e.key === "Escape") { e.preventDefault(); setOpen(false); return; }
    if (e.key === "ArrowDown") { e.preventDefault(); setActive(step(active, 1)); return; }
    if (e.key === "ArrowUp") { e.preventDefault(); setActive(step(active, -1)); return; }
    if (e.key === "Home") { e.preventDefault(); setActive(Math.max(0, firstEnabled)); return; }
    if (e.key === "End") { e.preventDefault(); setActive(step(options.length, -1)); return; }
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (enabled(active)) { onPick(options[active].value); setOpen(false); }
      return;
    }
    if (e.key === "Tab") setOpen(false);
  };

  return (
    <div className={inline ? "listbox listbox-inline" : "listbox"} ref={rootRef}
         onBlur={(e) => {
           if (!rootRef.current?.contains(e.relatedTarget as Node)) setOpen(false);
         }}>
      <button
        type="button"
        className="listbox-btn"
        // The attribute the pattern is named after. Without it this is a plain
        // button wearing combobox attributes: `aria-activedescendant` and
        // `aria-controls` are only defined on a combobox, so a screen reader
        // had no reason to follow them and the active option went unannounced —
        // arrow keys moved a highlight nobody was told about. The docstring
        // above already claimed the select-only combobox pattern; this is the
        // line that makes the claim true.
        role="combobox"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={label}
        aria-controls={open ? `${id}-list` : undefined}
        aria-activedescendant={open ? `${id}-opt-${active}` : undefined}
        onClick={() => (open ? setOpen(false) : openAt())}
        onKeyDown={onKey}
      >
        <span className="listbox-value">{trigger}</span>
        <span className="listbox-caret" aria-hidden="true">▾</span>
      </button>
      {open && rect && createPortal(
        <div ref={popRef}
             className="listbox-pop portal" role="listbox" id={`${id}-list`}
             aria-label={label}
             style={{ top: rect.top, left: rect.left, minWidth: rect.width,
                      maxHeight: rect.maxH }}
             // Focus stays on the trigger (select-only combobox pattern). Without
             // this, mousedown on an option blurs the button, the blur handler
             // unmounts the popup, and the click lands on a detached node —
             // the pick silently never happens.
             onMouseDown={(e) => e.preventDefault()}>

          {runs.map((run) => {
            const opts = run.items.map((o, j) => {
              const i = run.from + j;
              return (
                <div
                  key={`${o.group ?? ""}:${o.value}`}
                  id={`${id}-opt-${i}`}
                  role="option"
                  // A native <option> could be picked by value; this one is only
                  // its label, which leaves anything driving the control (the QA
                  // sweep, a keyboard test) matching on display text. The value
                  // is what the option MEANS, so it rides along.
                  data-value={o.value}
                  aria-selected={o.value === selected}
                  aria-disabled={o.disabled || undefined}
                  className={"listbox-opt"
                    + (i === active ? " active" : "")
                    + (o.disabled ? " disabled" : "")
                    + (o.value === selected ? " selected" : "")}
                  onPointerMove={() => { if (!o.disabled) setActive(i); }}
                  onClick={() => {
                    if (!o.disabled) { onPick(o.value); setOpen(false); }
                  }}
                >
                  <span className="listbox-opt-label">{o.label}</span>
                  {o.disabled && <span className="listbox-check" aria-hidden="true">✓</span>}
                  {o.figure && <span className="listbox-figure">{o.figure}</span>}
                </div>
              );
            });
            // The header is the group's name; aria-label says it once.
            return run.group ? (
              <div key={`g${run.from}`} role="group" aria-label={run.group}>
                <div className="listbox-group" aria-hidden="true">{run.group}</div>
                {opts}
              </div>
            ) : <Fragment key={`g${run.from}`}>{opts}</Fragment>;
          })}
        </div>,
        document.body,
      )}
    </div>
  );
}

/** Two thumbs on one track. The pair of stacked single ranges said "two
 * settings"; one track with a filled span says what the control means — a
 * window on the axis. The inputs stay native <input type=range> (real focus,
 * arrows, screen-reader value); only their track is repainted. */
export function DualRange({ min, max, lo, hi, onInput, labelLo, labelHi }: {
  min: number;
  max: number;
  lo: number;
  hi: number;
  /** Which thumb moved and to where — the caller owns clamping and the
   * "untouched bound means unbounded" rule, which needs to know the thumb. */
  onInput: (which: "lo" | "hi", v: number) => void;
  labelLo: string;
  labelHi: string;
}) {
  const pct = (v: number) => ((v - min) / Math.max(1e-9, max - min)) * 100;
  return (
    <div className="dual-range">
      <div className="dual-track" aria-hidden="true">
        <div className="dual-fill"
             style={{ left: `${pct(lo)}%`, width: `${pct(hi) - pct(lo)}%` }} />
      </div>
      <input
        type="range" min={min} max={max} value={lo} aria-label={labelLo}
        // When both thumbs sit at the top, the lo thumb must win the click or
        // the range can never be reopened from the right.
        style={{ zIndex: lo > (min + max) / 2 ? 4 : 2 }}
        onChange={(e) => onInput("lo", Number(e.target.value))}
      />
      <input
        type="range" min={min} max={max} value={hi} aria-label={labelHi}
        style={{ zIndex: 3 }}
        onChange={(e) => onInput("hi", Number(e.target.value))}
      />
    </div>
  );
}
