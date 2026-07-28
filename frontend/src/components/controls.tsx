import { Fragment, KeyboardEvent, useEffect, useId, useRef, useState } from "react";

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
  options: { value: string; label: string }[];
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
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Select-only combobox (APG pattern): focus stays on the trigger, arrows move
 * an active row, Enter picks, Escape closes. The popup positions inside the
 * rail column, so it scrolls with the panel it belongs to. */
export function Listbox({ trigger, options, onPick, label, selected }: {
  /** Text on the closed control — the current value, or what picking adds. */
  trigger: string;
  options: PickOption[];
  onPick: (v: string) => void;
  label: string;
  /** Marked aria-selected; "" is a real value (the "All …" row). */
  selected?: string;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const id = useId();

  const enabled = (i: number) => !options[i]?.disabled;
  const firstEnabled = options.findIndex((_, i) => enabled(i));

  const openAt = () => {
    const sel = options.findIndex((o) => o.value === selected);
    setActive(sel >= 0 && enabled(sel) ? sel : Math.max(0, firstEnabled));
    setOpen(true);
  };

  // Outside pointerdown closes; blur alone misses clicks on non-focusable text.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [open]);

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
    <div className="listbox" ref={rootRef}
         onBlur={(e) => {
           if (!rootRef.current?.contains(e.relatedTarget as Node)) setOpen(false);
         }}>
      <button
        type="button"
        className="listbox-btn"
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
      {open && (
        <div className="listbox-pop" role="listbox" id={`${id}-list`} aria-label={label}
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
        </div>
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
