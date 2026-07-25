import { FormEvent, useCallback, useEffect, useState } from "react";
import "../styles/saved-views.css";

interface SavedView {
  name: string;
  query_string: string;
  created_at: string;
}

interface Props {
  /** The query string of the view on screen right now, with or without a leading "?". */
  current: string;
  onRestore: (queryString: string) => void;
}

/** Pull FastAPI's `{"detail": ...}` out of an error response, if it has one. */
async function errorText(res: Response, fallback: string): Promise<string> {
  try {
    const body: unknown = await res.json();
    if (body !== null && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string" && detail) return detail;
    }
  } catch {
    // Non-JSON error body (a proxy page, say) — the fallback is more useful.
  }
  return fallback;
}

/** The URL bar hands us "?a=1"; the API stores bare query strings. Normalising
 * here keeps a view saved from this component byte-identical to one saved by
 * any other client, so string comparison on query_string stays meaningful. */
function bareQuery(qs: string): string {
  return qs.startsWith("?") ? qs.slice(1) : qs;
}

function shortDate(iso: string): string {
  const t = Date.parse(iso);
  return Number.isNaN(t) ? "" : new Date(t).toLocaleDateString();
}

/** Named filter sets.
 *
 * A useful filter set here is several controls deep — split, an attribute
 * facet, two axis ranges, a sort — and rebuilding one from memory is the slow
 * part of coming back to an investigation. Saving the URL is enough to restore
 * it, and costs nothing when the feature is absent. */
export default function SavedViews({ current, onRestore }: Props) {
  const [views, setViews] = useState<SavedView[]>([]);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A backend without the views router (or an older one) must not leave a
  // broken panel on the page — the whole component stands down instead.
  const [available, setAvailable] = useState(true);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/views");
      if (res.status === 404) {
        setAvailable(false);
        return;
      }
      if (!res.ok) throw new Error(await errorText(res, "Could not load saved views"));
      const body = (await res.json()) as SavedView[];
      setViews(body);
      setAvailable(true);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load saved views");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const save = async (e: FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/views", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: trimmed, query_string: bareQuery(current) }),
      });
      if (!res.ok) {
        throw new Error(await errorText(
          res,
          res.status === 409
            ? `A view named “${trimmed}” already exists`
            : "Could not save this view",
        ));
      }
      setName("");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save this view");
    } finally {
      setBusy(false);
    }
  };

  const remove = async (viewName: string) => {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/views/${encodeURIComponent(viewName)}`, { method: "DELETE" });
      if (!res.ok) throw new Error(await errorText(res, "Could not delete this view"));
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete this view");
    } finally {
      setBusy(false);
    }
  };

  if (!available) return null;

  return (
    // Collapsed until it holds something: the filter column already stacks
    // chips, the id list and four axis sliders above the results, and an
    // expanded empty state costs ~110px of vertical space for no information.
    <details className="saved-views" open={views.length > 0}>
      <summary className="saved-views-title">
        Saved views
        {views.length > 0 && <span className="saved-views-count">{views.length}</span>}
      </summary>

      {error && <div className="error saved-views-error">{error}</div>}

      {views.length === 0 ? (
        <p className="saved-views-empty">
          No saved views yet — name the current filter set to keep it.
        </p>
      ) : (
        <ul className="saved-views-list">
          {views.map((v) => (
            <li className="saved-view" key={v.name}>
              <button
                type="button"
                className="saved-view-restore"
                title={v.query_string ? `?${v.query_string}` : "No filters"}
                onClick={() => onRestore(v.query_string)}
              >
                <span className="saved-view-name">{v.name}</span>
                <span className="saved-view-date">{shortDate(v.created_at)}</span>
              </button>
              <button
                type="button"
                className="saved-view-delete"
                aria-label={`Delete saved view ${v.name}`}
                title={`Delete ${v.name}`}
                disabled={busy}
                onClick={() => void remove(v.name)}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}

      <form className="saved-views-form" onSubmit={(e) => void save(e)}>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Name this view…"
          aria-label="Name for the current view"
          disabled={busy}
        />
        <button className="ghost" type="submit" disabled={busy || !name.trim()}>
          Save
        </button>
      </form>
    </details>
  );
}
