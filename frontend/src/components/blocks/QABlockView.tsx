import { useEffect, useState } from "react";
import type { QABlock, QACheck, QAFlowResult } from "../../api/blocks";

const STATUS_WORD: Record<QAFlowResult["status"], string> = {
  pass: "pass", fail: "fail", skip: "skipped",
};

function duration(s: number | null | undefined): string | null {
  if (typeof s !== "number" || !Number.isFinite(s)) return null;
  return s >= 10 ? `${s.toFixed(0)}s` : `${s.toFixed(1)}s`;
}

function checkCount(checks: QACheck[]): string {
  const ok = checks.filter((c) => c.ok).length;
  return `${ok}/${checks.length} checks`;
}

export default function QABlockView({ block }: { block: QABlock }) {
  const [zoom, setZoom] = useState<{ url: string; name: string } | null>(null);

  // Escape closes the full-size view. Registered only while it is open, so the
  // transcript keeps its own key handling the rest of the time.
  useEffect(() => {
    if (zoom === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setZoom(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [zoom]);

  const flows = block.flows ?? [];
  const total = block.total ?? flows.length;
  const passed = block.passed ?? flows.filter((f) => f.status === "pass").length;
  const errors = block.console_errors ?? [];
  const running = block.status === "running";
  const done = total > 0 ? Math.min(flows.length, total) : flows.length;
  const failedFlows = flows.filter((f) => f.status === "fail").length;

  return (
    <div className="vblock-body vqa">
      <div className="vqa-summary">
        <span className={`vqa-badge ${block.status}`}>
          {running ? "running" : block.status}
        </span>
        {/* The headline is the ratio, not a colour: "42/44" says what a green
            tick cannot, which is how much of the suite actually ran.
            `passed`/`total` count individual CHECKS, not workflows — the two
            differ by roughly 5x, and labelling 44 checks as "44 flows" made an
            8-workflow sweep read as a 44-workflow one. Both are shown. */}
        <span className="vqa-score">
          <strong>{passed}</strong>/{total || "?"} checks passed
        </span>
        {flows.length > 0 && (
          <span className="vqa-run">
            {flows.length - failedFlows}/{flows.length} workflows
          </span>
        )}
        {failedFlows > 0 && (
          <span className="vqa-failed">{failedFlows} failed</span>
        )}
        <span className="vqa-run">run {block.run_id}</span>
        {(block.deck_url || block.markdown_url) && (
          <span className="vqa-downloads">
            {block.markdown_url && (
              <a className="export-pill" href={block.markdown_url} download>markdown</a>
            )}
            {block.deck_url && (
              <a className="export-pill" href={block.deck_url} download>deck</a>
            )}
          </span>
        )}
      </div>

      {running && (
        <div className="vqa-progress">
          <div
            className="vqa-progress-track"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={total || undefined}
            aria-valuenow={total > 0 ? done : undefined}
            aria-label={`QA run ${block.run_id} progress`}
          >
            <div
              className={`vqa-progress-fill${total > 0 ? "" : " indeterminate"}`}
              style={total > 0 ? { width: `${(done / total) * 100}%` } : undefined}
            />
          </div>
          <span className="vblock-hint">
            {total > 0
              ? `Flow ${Math.min(done + 1, total)} of ${total} — results appear as each finishes.`
              : "Starting the suite…"}
          </span>
        </div>
      )}

      {flows.length === 0 && !running && (
        <div className="vblock-empty">This run reported no flows.</div>
      )}

      <ul className="vqa-flows">
        {flows.map((f, i) => {
          const checks = f.checks ?? [];
          const secs = duration(f.duration_s);
          return (
            <li className={`vqa-flow ${f.status}`} key={`${i}:${f.name}`}>
              <div className="vqa-flow-head">
                <span className={`vqa-badge ${f.status}`}>{STATUS_WORD[f.status]}</span>
                <span className="vqa-flow-name">{f.name}</span>
                {secs !== null && <span className="vqa-flow-time">{secs}</span>}
                {checks.length > 0 && (
                  <span className="vqa-flow-checks">{checkCount(checks)}</span>
                )}
              </div>

              {f.detail && <p className="vqa-flow-detail">{f.detail}</p>}

              {f.screenshot && (
                <button
                  type="button"
                  className="vqa-shot"
                  title={`${f.name} — open the screenshot full size`}
                  onClick={() => setZoom({
                    url: f.screenshot as string,
                    name: f.name,
                  })}
                >
                  <img src={f.screenshot} alt={`Screenshot from the ${f.name} flow`} loading="lazy" />
                  <span className="vqa-shot-zoom" aria-hidden="true">⤢</span>
                </button>
              )}

              {checks.length > 0 && (
                <details className="vqa-checks">
                  <summary>{checkCount(checks)}</summary>
                  <ul>
                    {checks.map((c, ci) => (
                      <li key={`${ci}:${c.name}`} className={c.ok ? "ok" : "bad"}>
                        <span className="vqa-check-mark" aria-hidden="true">
                          {c.ok ? "✓" : "✗"}
                        </span>
                        {/* The word, not just the mark: a tick is invisible to a
                            screen reader and to anyone who cannot tell the two
                            colours apart. */}
                        <span className="vqa-check-state">{c.ok ? "passed" : "failed"}</span>
                        <span className="vqa-check-name">{c.name}</span>
                        {c.detail && <span className="vqa-check-detail">{c.detail}</span>}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </li>
          );
        })}
      </ul>

      {errors.length > 0 && (
        <details className="vqa-console" open={errors.length <= 3}>
          <summary>
            {errors.length} console {errors.length === 1 ? "error" : "errors"} during the run
          </summary>
          <ul>
            {errors.map((e, i) => <li key={`${i}:${e}`}>{e}</li>)}
          </ul>
        </details>
      )}

      {(block.started_at || block.finished_at) && (
        <p className="vblock-hint">
          {block.started_at && <>Started {block.started_at}. </>}
          {block.finished_at && <>Finished {block.finished_at}.</>}
        </p>
      )}

      {zoom !== null && (
        // A dialog rather than a new tab: the screenshot is evidence for the row
        // it sits next to, and losing the transcript to look at it makes the
        // comparison harder than it needs to be.
        <div
          className="vqa-lightbox"
          role="dialog"
          aria-modal="true"
          aria-label={`Screenshot from the ${zoom.name} flow`}
          onClick={() => setZoom(null)}
        >
          <div className="vqa-lightbox-inner" onClick={(e) => e.stopPropagation()}>
            <div className="vqa-lightbox-bar">
              <span>{zoom.name}</span>
              <span className="vqa-lightbox-actions">
                <a className="export-pill" href={zoom.url} target="_blank" rel="noreferrer">
                  open in a tab
                </a>
                <button type="button" className="vqa-lightbox-close" onClick={() => setZoom(null)}>
                  Close
                </button>
              </span>
            </div>
            <img src={zoom.url} alt={`Screenshot from the ${zoom.name} flow, full size`} />
            <p className="vblock-hint">Press Escape or click outside to close.</p>
          </div>
        </div>
      )}
    </div>
  );
}
