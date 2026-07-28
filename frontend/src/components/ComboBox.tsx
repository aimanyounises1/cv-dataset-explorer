import { useEffect, useId, useRef, useState } from "react";

/**
 * A text field that suggests, without becoming a picker.
 *
 * `<input list=...>` is the browser's own popup: it takes its type, colours and
 * placement from the OS, which is the one thing on a composed surface that
 * cannot be made to match — and on the tray's dark ink it reads as a hole. This
 * keeps the part a datalist gets right (type anything; a new name is as valid as
 * an existing one) and draws the suggestions itself.
 *
 * Free text is the point, so the value is never forced onto a suggestion:
 * arrows move a highlight, Enter takes the highlighted one, Escape dismisses the
 * list and leaves what was typed. Selecting nothing is a normal outcome.
 */
export default function ComboBox({
  value, onChange, options, placeholder, label, className = "", onSubmit,
}: {
  value: string;
  onChange: (v: string) => void;
  /** Suggestions, unfiltered — narrowing against what is typed happens here. */
  options: string[];
  placeholder?: string;
  label: string;
  className?: string;
  /** Enter with no highlighted suggestion, e.g. to submit the surrounding form. */
  onSubmit?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const id = useId();

  const needle = value.trim().toLowerCase();
  const shown = options
    .filter((o) => o.toLowerCase().includes(needle) && o.toLowerCase() !== needle)
    .slice(0, 6);

  // A shorter list can strand the highlight past its end.
  useEffect(() => { setActive(-1); }, [shown.length]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [open]);

  const take = (v: string) => { onChange(v); setOpen(false); setActive(-1); };

  return (
    <div className={`combo ${className}`.trim()} ref={rootRef}>
      <input
        role="combobox"
        aria-expanded={open && shown.length > 0}
        aria-controls={`${id}-list`}
        aria-autocomplete="list"
        aria-activedescendant={active >= 0 ? `${id}-opt-${active}` : undefined}
        aria-label={label}
        value={value}
        placeholder={placeholder}
        onChange={(e) => { onChange(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown" && shown.length) {
            e.preventDefault();
            setOpen(true);
            setActive((i) => (i + 1) % shown.length);
          } else if (e.key === "ArrowUp" && shown.length) {
            e.preventDefault();
            setActive((i) => (i <= 0 ? shown.length - 1 : i - 1));
          } else if (e.key === "Enter") {
            if (open && active >= 0) { e.preventDefault(); take(shown[active]); }
            else if (onSubmit) { e.preventDefault(); onSubmit(); }
          } else if (e.key === "Escape") {
            // Dismiss the list, keep the text: Escape here means "stop
            // suggesting", not "discard what I typed".
            e.stopPropagation();
            setOpen(false);
            setActive(-1);
          }
        }}
      />
      {open && shown.length > 0 && (
        <ul className="combo-list" id={`${id}-list`} role="listbox" aria-label={label}>
          {shown.map((o, i) => (
            <li key={o} id={`${id}-opt-${i}`} role="option"
                aria-selected={i === active}
                className={i === active ? "active" : ""}
                // Keeps focus in the field through the press, so blur cannot
                // close the list before the click lands.
                onMouseDown={(e) => e.preventDefault()}
                onMouseEnter={() => setActive(i)}
                onClick={() => take(o)}>
              {o}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
