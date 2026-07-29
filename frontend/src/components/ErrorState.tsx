import type { ReactNode } from "react";
import { ApiError } from "../api/client";

/**
 * A failed request, presented as a reader can act on it.
 *
 * The pane used to print `String(e)` — which for a request that failed
 * validation is the whole `{"detail":[{"type":"int_parsing",…}]}` document, in
 * a red box, as the only thing on the page. That reads as a crash rather than
 * as an answer, and it is the one surface where looking unfinished costs the
 * most trust.
 *
 * The payload is not discarded, because on this product the reader is an
 * engineer and the status and body are the actual clue: they move behind a
 * disclosure, closed by default. So the first line says what happened, the
 * `hint` says what to do about it, `children` carries the way out, and the
 * detail is one click away for whoever wants it.
 */
export default function ErrorState({ title, error, hint, children }: {
  /** The sentence. Falls back to the error's own message when not given. */
  title?: string;
  error: unknown;
  /** What the reader can do about it, when the cause suggests something. */
  hint?: ReactNode;
  /** Actions — a way back, a retry. */
  children?: ReactNode;
}) {
  const api = error instanceof ApiError ? error : null;
  const sentence = title
    ?? (error instanceof Error ? error.message : String(error));
  // Only worth a disclosure when it holds more than the sentence already says.
  const raw = api && api.body.trim() && api.body.trim() !== sentence
    ? api.body : null;

  return (
    <div className="error-state" role="alert">
      <div className="error-state-head">
        {api && <span className="error-state-status mono">{api.status}</span>}
        <p className="error-state-title">{sentence}</p>
      </div>
      {hint && <p className="error-state-hint">{hint}</p>}
      {children && <div className="error-state-actions">{children}</div>}
      {raw && (
        <details className="error-state-raw">
          <summary>What the API returned</summary>
          <pre>{raw}</pre>
        </details>
      )}
    </div>
  );
}
