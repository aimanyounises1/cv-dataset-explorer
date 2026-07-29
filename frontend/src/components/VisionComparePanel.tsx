import {
  useEffect, useRef, useState,
} from "react";
import { api } from "../api/client";
import type {
  VisionPairCapabilityStatus,
  VisionPairCompareResponse,
} from "../api/types";
import "../styles/vision-compare.css";

interface Props {
  aId: number;
  bId: number;
  onGround?: (sampleId: number, term: string) => void;
}

type RunState = "idle" | "running" | "complete" | "failed";

const shortDigest = (value: string) => `${value.slice(0, 10)}…${value.slice(-6)}`;

function downloadProposal(result: VisionPairCompareResponse) {
  const blob = new Blob([`${JSON.stringify(result, null, 2)}\n`], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `cvde-pair-${result.image_a.sample_id}-${result.image_b.sample_id}.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export default function VisionComparePanel({ aId, bId, onGround }: Props) {
  // The server publishes ONE pair adapter, separately from model discovery, so
  // there is a capability to read rather than a catalogue to choose from.
  const [pair, setPair] = useState<VisionPairCapabilityStatus | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [runState, setRunState] = useState<RunState>("idle");
  const [result, setResult] = useState<VisionPairCompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const runAbort = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let current = true;
    setCatalogError(null);
    api.visionModels(controller.signal)
      .then((catalog) => {
        if (!current) return;
        setPair(catalog.pair_comparison);
        setLoaded(true);
      })
      .catch((caught: unknown) => {
        if (!current || controller.signal.aborted) return;
        setLoaded(true);
        setCatalogError(
          caught instanceof Error ? caught.message : "Could not inspect local vision models.",
        );
      });
    return () => {
      current = false;
      controller.abort();
    };
  }, []);

  useEffect(() => {
    runAbort.current?.abort();
    setRunState("idle");
    setResult(null);
    setError(null);
  }, [aId, bId]);

  useEffect(() => () => runAbort.current?.abort(), []);

  const ready = pair?.ready === true;

  const run = async () => {
    if (!ready || runState === "running") return;
    runAbort.current?.abort();
    const controller = new AbortController();
    runAbort.current = controller;
    setRunState("running");
    setResult(null);
    setError(null);
    try {
      // No model field: the request schema forbids extras, and which adapter
      // runs is the server's call, bound to the digest it validated.
      const next = await api.compareVision({
        a_sample_id: aId,
        b_sample_id: bId,
      }, controller.signal);
      setResult(next);
      setRunState("complete");
    } catch (caught: unknown) {
      if (controller.signal.aborted) return;
      setError(
        caught instanceof Error ? caught.message : "Visual comparison failed.",
      );
      setRunState("failed");
    } finally {
      if (runAbort.current === controller) runAbort.current = null;
    }
  };

  const proposal = result?.proposal;
  const groundingCount = proposal
    ? proposal.grounding_terms_a.length + proposal.grounding_terms_b.length
    : 0;
  const stages = [
    {
      label: "Asset health",
      detail: result
        ? "2 / 2 decoded"
        : runState === "running"
          ? "pending server result"
          : runState === "failed"
            ? "not confirmed by completed run"
            : "will be verified before inference",
      state: result ? "complete" : "queued",
      authority: "measurement",
    },
    {
      label: "Semantic difference",
      detail: runState === "running"
        ? "waits for decode checks"
        : runState === "complete"
          ? `${proposal?.differences.length ?? 0} differences proposed`
          : runState === "failed"
            ? "comparison failed"
            : "ready",
      state: runState === "complete"
        ? "complete"
        : runState === "failed"
            ? "failed"
            : "queued",
      authority: "model",
    },
    {
      label: "Ground objects",
      detail: groundingCount > 0 ? `${groundingCount} phrases proposed` : "choose after comparison",
      state: groundingCount > 0 ? "ready" : "queued",
      authority: "model",
    },
    {
      label: "Segment & isolate",
      detail: "after a box is chosen",
      state: "queued",
      authority: "model",
    },
    {
      label: "Human review",
      detail: "required before annotation",
      state: "queued",
      authority: "human",
    },
  ] as const;

  const unavailableReason = pair?.reason;

  return (
    <section className="vision-compare panel" aria-labelledby="vision-compare-title">
      <header className="vc-header">
        <div>
          <div className="vc-eyebrow">Inspection run · pair scope</div>
          <h2 id="vision-compare-title">Visual difference</h2>
          <p>
            Decode checks run first. A local vision model then proposes semantic
            differences; detector and mask actions remain separate, reviewable steps.
          </p>
        </div>
        <span className="vc-authority">Model proposal</span>
      </header>

      <ol className="vc-spine" aria-label="Pair inspection stages">
        {stages.map((stage) => (
          <li key={stage.label} className={`${stage.state} ${stage.authority}`}>
            <span className="vc-stage-mark" aria-hidden="true" />
            <span>
              <strong>{stage.label}</strong>
              <small>{stage.detail}</small>
            </span>
          </li>
        ))}
      </ol>

      {!loaded && (
        <div className="vc-notice" role="status">Checking the local pair inspector…</div>
      )}
      {catalogError && <div className="vc-error" role="alert">{catalogError}</div>}
      {loaded && !ready && !catalogError && (
        <div className="vc-notice">
          <strong>Pair inspection is unavailable.</strong>
          <span>
            {unavailableReason
              ?? "No exact local model artifact has passed the pair-comparison contract."}
          </span>
        </div>
      )}

      {ready && pair && (
        <>
          <div className="vc-runbar">
            <button
              type="button"
              className="primary"
              disabled={runState === "running"}
              onClick={run}
            >
              {runState === "running"
                ? "Analyzing visual differences…"
                : "Analyze visual differences"}
            </button>
            <span>
              A #{aId} → B #{bId} · local semantic inspector
            </span>
          </div>

          {/* Read-only provenance, not a setting: the adapter is pinned server
              side and the capability is bound to the digest it validated, so
              offering a picker here would promise a choice the request schema
              refuses. */}
          <details className="vc-model-settings">
            <summary>Local model &amp; contract</summary>
            <dl>
              <dt>Artifact</dt>
              <dd>{pair.model ?? "unavailable"}</dd>
              <dt>Digest</dt>
              <dd title={pair.model_digest ?? undefined}>
                {pair.model_digest ? shortDigest(pair.model_digest) : "unavailable"}
              </dd>
              <dt>Runtime</dt>
              <dd>
                {pair.runtime}
                {pair.runtime_version ? ` ${pair.runtime_version}` : ""}
              </dd>
              <dt>Adapter</dt>
              <dd>{pair.adapter_id} v{pair.adapter_version}</dd>
              <dt>Protocol</dt>
              <dd>{pair.protocol ?? "unavailable"}</dd>
            </dl>
          </details>
        </>
      )}

      <div className="vc-live" aria-live="polite">
        {runState === "running"
          ? "Validating both assets, then running semantic comparison."
          : runState === "complete"
            ? `Pair comparison complete. ${
              proposal?.differences.length ?? 0
            } differences proposed.`
            : runState === "failed"
              ? "Pair comparison failed. Review the error and retry."
              : ""}
      </div>
      {error && <div className="vc-error" role="alert">{error}</div>}

      {result && proposal && (
        <div className="vc-result">
          <div className="vc-result-head">
            <div>
              <span className="vc-result-label">Semantic difference proposal</span>
              <strong>{proposal.summary}</strong>
            </div>
            <button type="button" className="ghost" onClick={() => downloadProposal(result)}>
              Export JSON
            </button>
          </div>

          {proposal.differences.length > 0 && (
            <div className="vc-differences-scroll">
              <table className="vc-differences" aria-label="Visible differences">
                <thead>
                  <tr>
                    <th scope="col">Subject</th>
                    <th scope="col">Image A</th>
                    <th scope="col">Image B</th>
                  </tr>
                </thead>
                <tbody>
                  {proposal.differences.map((difference, index) => (
                    <tr key={`${difference.subject}-${difference.change_type}-${index}`}>
                      <th scope="row">
                        <strong>{difference.subject}</strong>
                        <small>{difference.change_type}</small>
                      </th>
                      <td>{difference.image_a}</td>
                      <td>{difference.image_b}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {(proposal.only_a.length > 0 || proposal.only_b.length > 0) && (
            <div className="vc-only">
              <section>
                <h3>Only in A</h3>
                {proposal.only_a.length > 0
                  ? <ul>{proposal.only_a.map((item) => <li key={item}>{item}</li>)}</ul>
                  : <p>None proposed.</p>}
              </section>
              <section>
                <h3>Only in B</h3>
                {proposal.only_b.length > 0
                  ? <ul>{proposal.only_b.map((item) => <li key={item}>{item}</li>)}</ul>
                  : <p>None proposed.</p>}
              </section>
            </div>
          )}

          {proposal.shared.length > 0 && (
            <section className="vc-shared">
              <h3>Shared visible evidence</h3>
              <div>{proposal.shared.map((item) => <span key={item}>{item}</span>)}</div>
            </section>
          )}

          {groundingCount > 0 && (
            <div className="vc-grounding">
              <div>
                <h3>Ground in image A</h3>
                <p>Pass one phrase to the open-vocabulary detector.</p>
                <div>
                  {proposal.grounding_terms_a.map((term) => (
                    <button
                      type="button"
                      key={term}
                      disabled={!onGround}
                      onClick={() => onGround?.(aId, term)}
                    >
                      {term}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <h3>Ground in image B</h3>
                <p>Geometry remains a proposal until human review.</p>
                <div>
                  {proposal.grounding_terms_b.map((term) => (
                    <button
                      type="button"
                      key={term}
                      disabled={!onGround}
                      onClick={() => onGround?.(bId, term)}
                    >
                      {term}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {proposal.uncertainties.length > 0 && (
            <section className="vc-uncertainties">
              <h3>Uncertainty</h3>
              <ul>{proposal.uncertainties.map((item) => <li key={item}>{item}</li>)}</ul>
            </section>
          )}

          <details className="vc-provenance">
            <summary>Provenance &amp; limits</summary>
            <dl>
              <dt>Image A</dt>
              <dd>
                {result.image_a.width} × {result.image_a.height} ·{" "}
                {result.image_a.mode} · {result.image_a.byte_length.toLocaleString()} bytes ·{" "}
                decoded ·{" "}
                <code>{shortDigest(result.image_a.image_sha256)}</code>
              </dd>
              <dt>Image B</dt>
              <dd>
                {result.image_b.width} × {result.image_b.height} ·{" "}
                {result.image_b.mode} · {result.image_b.byte_length.toLocaleString()} bytes ·{" "}
                decoded ·{" "}
                <code>{shortDigest(result.image_b.image_sha256)}</code>
              </dd>
              <dt>Model</dt>
              <dd><code>{result.model}@{shortDigest(result.model_digest)}</code></dd>
              <dt>Runtime</dt>
              <dd>{result.runtime} {result.runtime_version}</dd>
              <dt>Adapter</dt>
              <dd>{result.adapter_id} v{result.adapter_version}</dd>
              <dt>Contract</dt>
              <dd>
                prompt {result.prompt_version} · schema {result.schema_version}
                {" · "}{result.protocol}
              </dd>
              <dt>Proposal</dt>
              <dd><code>{result.proposal_id}</code></dd>
              <dt>Latency</dt>
              <dd>{(result.latency_ms / 1_000).toFixed(2)} s</dd>
            </dl>
            <p>{result.note}</p>
          </details>
        </div>
      )}
    </section>
  );
}
