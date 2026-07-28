import { FormEvent, useCallback, useEffect, useState } from "react";

interface SavedView {
  name: string;
  query_string: string;
  created_at: string;
  /** Warning text when the view was saved under a different embedding model or
   * corpus than the one now loaded; null/absent when the environments match —
   * or when the view predates fingerprinting and there is nothing to compare. */
  stale_env?: string | null;
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

  const form = (
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
  );

  /* Empty is not a disclosure: the copy that explains what a saved view IS
   * was hidden behind a closed, marker-less summary — an apparently empty box
   * by this app's own empty-state standard. With nothing to collapse, render
   * the explanation and the form directly. */
  if (views.length === 0) {
    return (
      <div className="saved-views">
        {error && <div className="error saved-views-error">{error}</div>}
        <p className="saved-views-empty">
          No saved views yet — name the current filter set to keep it.
        </p>
        {form}
      </div>
    );
  }

  return (
    // Open by default but still collapsible — with a visible chevron, because
    // a flex summary silently suppresses the native disclosure marker.
    <details className="saved-views" open>
      <summary className="saved-views-title">
        Saved views
        <span className="saved-views-count">{views.length}</span>
      </summary>

      {error && <div className="error saved-views-error">{error}</div>}

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
                {/* Non-blocking by design: the view still restores — this only
                    says its results may not reproduce, and why is in the title. */}
                {v.stale_env && (
                  <span className="saved-view-stale" title={v.stale_env}>env differs</span>
                )}
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

      {form}
    </details>
  );
}
