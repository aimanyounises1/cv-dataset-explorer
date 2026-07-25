import { useRef, useState } from "react";
import "../styles/id-filter.css";
import { MAX_ID_LIST } from "../api/types";

interface Props {
  value: string;
  onChange: (v: string) => void;
  resolved?: number | null;
}

/** Paste a set of ids or filenames computed somewhere else.
 *
 * This is the way back in. A researcher's interesting set usually comes from
 * outside the tool — the worst-loss samples of a training run, a regression
 * list, another team's output — and without an ingress the tool can only ever
 * answer questions about sets it produced itself. Accepts ids or filenames
 * because both are things you already have, and composes with every other
 * filter rather than replacing them. */
export default function IdListFilter({ value, onChange, resolved }: Props) {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [draft, setDraft] = useState(value);
  const [readError, setReadError] = useState<string | null>(null);

  const entries = draft
    .replace(/,/g, "\n").split("\n").map((s) => s.trim()).filter(Boolean);
  const unique = new Set(entries).size;
  const overCap = unique > MAX_ID_LIST;
  const dirty = draft !== value;

  // 60,000 entries of "train_001722.jpg" is well under 2 MB, so anything much
  // larger is the wrong file. Checked before reading rather than after, because
  // File.text() on a multi-gigabyte pick would hang the tab before it failed.
  const MAX_FILE_BYTES = 4 * 1024 * 1024;

  const readFile = async (file: File) => {
    setReadError(null);
    if (file.size > MAX_FILE_BYTES) {
      setReadError(`${file.name} is ${(file.size / 1048576).toFixed(1)} MB — the limit `
                   + `is ${MAX_FILE_BYTES / 1048576} MB. An id list should be far smaller.`);
      return;
    }
    try {
      setDraft(await file.text());
    } catch {
      setReadError(`Could not read ${file.name}.`);
    }
  };

  return (
    <details className="idlist-panel" open={Boolean(value)}>
      <summary>
        Id list
        {value && <span className="idlist-count">{unique}</span>}
      </summary>

      <textarea
        className="idlist-input"
        aria-label="Sample ids or filenames"
        placeholder={`Enter sample ids or filenames (comma or newline separated) — max ${MAX_ID_LIST.toLocaleString()}`}
        value={draft}
        spellCheck={false}
        onChange={(e) => setDraft(e.target.value)}
      />

      <div className="idlist-actions">
        <button className="primary" disabled={overCap || !dirty}
                onClick={() => onChange(draft)}>
          Apply list
        </button>
        <button className="ghost" onClick={() => { fileRef.current?.click(); }}>
          Upload file
        </button>
        {value && (
          <button className="ghost" onClick={() => { setDraft(""); onChange(""); }}>
            Clear
          </button>
        )}
        <input ref={fileRef} type="file" accept=".txt,.csv,text/plain,text/csv"
               style={{ display: "none" }}
               onChange={(e) => {
                 const f = e.target.files?.[0];
                 if (f) void readFile(f);
                 e.target.value = "";   // let the same file be picked twice
               }} />
      </div>

      {readError && <div className="error">{readError}</div>}

      {overCap && (
        <p className="idlist-note over">
          {unique.toLocaleString()} entries is over the {MAX_ID_LIST.toLocaleString()} limit.
        </p>
      )}

      {/* Resolved-versus-pasted is the number that matters: a list carried over
          from a larger corpus is normal, and silence about the shortfall would
          leave the user thinking their filter did nothing. */}
      {!overCap && value && resolved != null && (
        <p className="idlist-note">
          {resolved.toLocaleString()} of {unique.toLocaleString()} entr
          {unique === 1 ? "y" : "ies"} found in this dataset
          {resolved < unique && " — the rest are not part of Flickr8k"}.
        </p>
      )}
    </details>
  );
}
