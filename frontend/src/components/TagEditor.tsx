import { FormEvent, useState } from "react";
import { api } from "../api/client";

interface Props {
  sampleId: number;
  tags: string[];
  onChanged: () => void;
}

export default function TagEditor({ sampleId, tags, onChanged }: Props) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const add = async (e: FormEvent) => {
    e.preventDefault();
    const name = value.trim();
    if (!name) return;
    setBusy(true);
    setError(null);
    try {
      await api.addTag(sampleId, name);
      setValue("");
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add tag");
    } finally {
      setBusy(false);
    }
  };

  const remove = async (name: string) => {
    setBusy(true);
    setError(null);
    try {
      await api.removeTag(sampleId, name);
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to remove tag");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      {error && <div className="error" style={{ padding: "4px 0", fontSize: 12 }}>{error}</div>}
      <div className="tag-row">
        {tags.length === 0 && <span style={{ color: "var(--text-dim)", fontSize: 13 }}>No tags yet</span>}
        {tags.map((t) => (
          <span className="tag" key={t}>
            {t}
            <button onClick={() => void remove(t)} title="Remove tag" disabled={busy}>×</button>
          </span>
        ))}
      </div>
      <form className="tag-input" onSubmit={(e) => void add(e)}>
        <input
          placeholder="Add tag (e.g. edge-case, verdict:caption-error)…"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={busy}
        />
        <button className="ghost" type="submit" disabled={busy || !value.trim()}>Add</button>
      </form>
    </div>
  );
}
