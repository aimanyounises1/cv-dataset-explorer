import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { DescribeResponse, FacetLift } from "../api/types";
import { AXIS_META } from "./AxisFilters";
import "../styles/set-summary.css";

/**
 * "What is this selection made of?" — the inverse of every other control here.
 *
 * The rest of the gallery answers *given a filter, which samples*. This answers
 * *given these samples, what do they have in common*, by comparing each
 * attribute's share inside the selection against its share of the corpus.
 *
 * Three things it deliberately shows, because a lift multiplier on its own is a
 * machine for producing confident nonsense:
 *
 *  - **the raw count next to every multiplier**, since ×6 over five images and
 *    ×6 over five hundred are different findings;
 *  - **under-represented facets**, because "there is almost no water here" is as
 *    real a characterisation as "this is mostly night";
 *  - **the thresholds a facet had to clear**, so an empty panel reads as "nothing
 *    stood out" rather than as a broken one.
 *
 * Collapsed by default. It costs a request, and it answers a question the user
 * has to have decided to ask.
 */
export default function SetSummary({ params, active }: {
  /** The same query parameters the gallery is already filtering by. */
  params: Record<string, string | number | string[] | undefined>;
  /** Whether any filter is applied. Describing the whole corpus is meaningless. */
  active: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<DescribeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const key = JSON.stringify(params);

  /**
   * Where a facet row goes when clicked: this selection, *narrowed* by that facet.
   *
   * The distinction matters and was wrong. Every row here is counted inside the
   * current selection — "223 images, x3.07 indoor" means 223 of the 392 night
   * images, not 223 of the corpus. Navigating to `/?${drill}` threw the selection
   * away and landed on all 1,483 indoor images: a row advertising 223 opened a set
   * 6.6x larger, and the number the user clicked appeared nowhere on the page they
   * arrived at.
   *
   * `append`, not `set`, because `attr` is repeatable and intersected server-side,
   * so a second facet narrows rather than replaces. Page resets: page 5 of the
   * old set indexes nothing in the new one.
   *
   * The drills on assistant blocks are deliberately NOT merged this way. Those
   * come from the agent's own corpus-wide queries, so their counts already stand
   * on their own and replacing is the honest move there.
   */
  const drillHref = (drill: string) => {
    const next = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v === undefined || v === "" || k === "page") continue;
      if (Array.isArray(v)) for (const one of v) next.append(k, one);
      else next.append(k, String(v));
    }
    const [k, v] = [drill.slice(0, drill.indexOf("=")), drill.slice(drill.indexOf("=") + 1)];
    if (!next.getAll(k).includes(v)) next.append(k, v);
    return `/?${next.toString()}`;
  };
  useEffect(() => {
    if (!open || !active) return undefined;
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    api.describe(params, ctrl.signal)
      .then(setData)
      .catch((e) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => setLoading(false));
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, active, key]);

  if (!active) return null;

  return (
    <details className="set-summary" open={open}
             onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>
        What is in this selection?
        {data && data.filtered && (
          <span className="set-summary-n">{data.set_size.toLocaleString()} images</span>
        )}
      </summary>

      {loading && <div className="loading">Comparing against the dataset…</div>}
      {error && <div className="error">{error}</div>}

      {data && data.message && <div className="empty">{data.message}</div>}

      {data && !data.message && (
        <>
          <p className="meta-line">
            {data.set_size.toLocaleString()} of {data.corpus_size.toLocaleString()} images
            {data.mean_agreement != null && data.corpus_mean_agreement != null && (
              <> · mean caption agreement <strong>{data.mean_agreement.toFixed(3)}</strong>{" "}
                against {data.corpus_mean_agreement.toFixed(3)} for the dataset</>
            )}
          </p>

          {data.axes.length > 0 && (
            <div className="set-axes">
              {data.axes.map((a) => (
                <span key={a.axis} className="set-axis"
                      title={`${AXIS_META[a.axis]?.label ?? a.axis}: this selection `
                             + `averages ${a.set_mean} against ${a.corpus_mean} for the `
                             + `whole dataset`}>
                  <span className="set-axis-label">
                    {AXIS_META[a.axis]?.label ?? a.axis}
                  </span>
                  <span className="set-axis-val">{a.set_mean.toFixed(2)}</span>
                  <span className={`set-axis-delta ${a.delta > 0 ? "up" : a.delta < 0 ? "down" : ""}`}>
                    {a.delta > 0 ? "+" : ""}{a.delta.toFixed(2)}
                  </span>
                </span>
              ))}
            </div>
          )}

          {data.over.length === 0 && data.under.length === 0 ? (
            <div className="empty">
              Nothing stands out. No attribute is over- or under-represented here by
              enough to be worth reporting — a facet needs at least{" "}
              {data.min_facet_count} images and a z-score of {data.min_abs_z} against
              the dataset. A selection that is most of the corpus will always look
              like this, correctly.
            </div>
          ) : (
            <div className="facet-cols">
              <FacetList title="Over-represented" facets={data.over}
                         onOpen={(f) => navigate(drillHref(f.drill))} />
              <FacetList title="Under-represented" facets={data.under}
                         onOpen={(f) => navigate(drillHref(f.drill))} />
            </div>
          )}

          <p className="set-caveat">
            Facets are zero-shot SigLIP labels, so this describes the selection{" "}
            <em>as the model sees it</em>, not as ground truth. Multipliers are
            share-in-selection over share-in-dataset, tested against the
            hypergeometric distribution — which is why a selection covering most of
            the corpus correctly reports nothing.
          </p>
        </>
      )}
    </details>
  );
}

function FacetList({ title, facets, onOpen }: {
  title: string;
  facets: FacetLift[];
  onOpen: (f: FacetLift) => void;
}) {
  if (facets.length === 0) return null;
  return (
    <div className="facet-col">
      <div className="eyebrow">{title}</div>
      {facets.map((f) => (
        <button key={`${f.group}:${f.label}`} className="facet-row"
                onClick={() => onOpen(f)}
                title={`${f.count} images here are “${f.label}” — `
                       + `${(f.set_share * 100).toFixed(1)}% of this selection against `
                       + `${(f.corpus_share * 100).toFixed(1)}% of the dataset. `
                       + `Click to open every “${f.label}” image in the gallery.`}>
          <span className="facet-name">
            <span className="facet-group">{f.group.replace(/_/g, " ")}</span>
            {f.label}
          </span>
          {/* A bar drawn on a log scale, so x0.4 and x2.5 are visually equal and
              opposite — a linear bar makes every under-representation look tiny. */}
          <span className="facet-bar" aria-hidden="true">
            <i className={f.lift >= 1 ? "up" : "down"}
               style={{ width: `${Math.min(100, Math.abs(Math.log2(Math.max(f.lift, 0.05))) * 33)}%` }} />
          </span>
          <span className={`facet-lift ${f.lift >= 1 ? "up" : "down"}`}>
            ×{f.lift.toFixed(2)}
          </span>
          <span className="facet-count">{f.count}</span>
        </button>
      ))}
    </div>
  );
}
