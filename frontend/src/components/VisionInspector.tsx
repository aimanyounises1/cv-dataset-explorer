import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type {
  SampleDetail,
  VisionInspectResponse,
  VisionModelStatus,
  VisionProposal,
  VisionTask,
} from "../api/types";
import "../styles/vision-inspector.css";

interface Props {
  sample: SampleDetail;
  onDetectorQuery: (query: string) => void;
}

const TASKS: Array<{
  value: VisionTask;
  label: string;
  description: string;
}> = [
  {
    value: "scene",
    label: "Scene",
    description: "Objects, lighting, surfaces, and searchable concepts.",
  },
  {
    value: "road_scene",
    label: "Road triage",
    description: "Actors, controls, visibility, and road conditions.",
  },
  {
    value: "caption_audit",
    label: "Captions",
    description: "Check each caption against visible image evidence.",
  },
  {
    value: "ocr",
    label: "OCR",
    description: "Propose visible text with location and legibility.",
  },
  {
    value: "question",
    label: "Ask image",
    description: "Answer a focused question from the pixels only.",
  },
];

const taskLabel = (task: VisionTask) =>
  TASKS.find((item) => item.value === task)?.label ?? task;

const normalizeQuestion = (value: string) =>
  value.trim().replace(/\s+/g, " ");

const resultKey = (
  task: VisionTask,
  model: string,
  question?: string | null,
) => JSON.stringify([
  task,
  model,
  task === "question" ? normalizeQuestion(question ?? "") : null,
]);

const compactDigest = (digest?: string | null) =>
  digest ? `${digest.slice(0, 10)}…${digest.slice(-6)}` : "digest unavailable";

const isAbort = (error: unknown) =>
  error instanceof DOMException && error.name === "AbortError";

function problem(error: unknown): string {
  if (!(error instanceof Error)) return "Vision inspection failed.";
  const separator = error.message.indexOf(":");
  if (separator < 0) return error.message;
  try {
    const payload = JSON.parse(error.message.slice(separator + 1).trim()) as {
      detail?: string | Array<{ msg?: string }>;
    };
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) {
      return payload.detail[0]?.msg?.replace(/^Value error, /, "")
        ?? error.message;
    }
  } catch {
    // The transport error itself is more useful than replacing it with a
    // generic message when the backend did not return JSON.
  }
  return error.message;
}

function proposalTerms(proposal: VisionProposal): string[] {
  return proposal.kind === "question" ? [] : proposal.search_terms;
}

function detectorTerms(proposal: VisionProposal): string[] {
  if (proposal.kind === "scene") {
    return proposal.objects.map((item) => item.name);
  }
  if (proposal.kind === "road_scene") {
    return proposal.actors
      .map((item) => item.category)
      .filter((item) => item !== "other");
  }
  return [];
}

function uniqueTerms(values: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of values) {
    const value = raw.trim();
    const key = value.toLocaleLowerCase();
    if (!value || seen.has(key)) continue;
    seen.add(key);
    out.push(value);
  }
  return out;
}

function groundingQuery(term: string): string {
  return `${term.replace(/\.+$/g, "").trim()}.`.slice(0, 300);
}

function SearchTerms({ terms }: { terms: string[] }) {
  const unique = uniqueTerms(terms);
  if (unique.length === 0) return null;
  const visible = unique.slice(0, 8);
  const omitted = unique.length - visible.length;
  return (
    <div className="vi-search-terms" aria-label="Searchable proposal terms">
      <span>Search the corpus</span>
      {visible.map((term) => (
        <Link
          key={term}
          to={`/?q=${encodeURIComponent(term)}&mode=semantic`}
          title={`Run measured semantic retrieval for “${term}”`}
        >
          {term}
        </Link>
      ))}
      {omitted > 0 && (
        <span className="vi-more-terms">+{omitted} in exported JSON</span>
      )}
    </div>
  );
}

function TextList({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div className="vi-list">
      <h4>{title}</h4>
      <ul>{items.map((item, index) => <li key={`${item}:${index}`}>{item}</li>)}</ul>
    </div>
  );
}

function ProposalBody({
  result,
  sample,
}: {
  result: VisionInspectResponse;
  sample: SampleDetail;
}) {
  const proposal = result.proposal;
  if (proposal.kind === "scene") {
    return (
      <>
        <p className="vi-summary">{proposal.summary}</p>
        <dl className="vi-facts">
          <div><dt>Setting</dt><dd>{proposal.setting}</dd></div>
          <div><dt>Lighting</dt><dd>{proposal.lighting.replace(/_/g, " ")}</dd></div>
        </dl>
        {proposal.objects.length > 0 && (
          <div className="vi-objects">
            <h4>Proposed object classes</h4>
            <ul>
              {proposal.objects.map((item, index) => (
                <li key={`${item.name}:${index}`}>
                  <strong>{item.name}</strong>
                  <span>{item.location}</span>
                  {item.attributes.length > 0 && (
                    <small>{item.attributes.join(" · ")}</small>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
        <div className="vi-columns">
          <TextList title="Surface conditions" items={proposal.surface_conditions} />
          <TextList title="Visible text" items={proposal.visible_text} />
        </div>
        <TextList title="Uncertain" items={proposal.uncertainties} />
      </>
    );
  }

  if (proposal.kind === "road_scene") {
    return (
      <>
        <div className={`vi-road-verdict${proposal.road_scene ? " yes" : ""}`}>
          {proposal.road_scene ? "Road scene candidate" : "Not identified as a road scene"}
        </div>
        <p className="vi-summary">{proposal.summary}</p>
        {proposal.actors.length > 0 && (
          <div className="vi-objects">
            <h4>Proposed actors</h4>
            <ul>
              {proposal.actors.map((actor, index) => (
                <li key={`${actor.category}:${index}`}>
                  <strong>{actor.category}</strong>
                  <span>{actor.location.replace(/_/g, " ")}</span>
                  <small>{actor.description}</small>
                </li>
              ))}
            </ul>
          </div>
        )}
        <div className="vi-columns">
          <TextList title="Traffic controls" items={proposal.traffic_controls} />
          <TextList title="Surface conditions" items={proposal.surface_conditions} />
          <TextList title="Visibility limits" items={proposal.visibility_limitations} />
          <TextList title="Uncertain" items={proposal.uncertainties} />
        </div>
      </>
    );
  }

  if (proposal.kind === "caption_audit") {
    return (
      <>
        {proposal.assessments.length === 0 ? (
          <p className="vi-empty">The model returned no caption assessments.</p>
        ) : (
          <ol className="vi-caption-audit">
            {proposal.assessments.map((assessment) => (
              <li key={`${assessment.caption_index}:${assessment.status}`}>
                <div>
                  <span className={`vi-assessment ${assessment.status}`}>
                    {assessment.status.replace(/_/g, " ")}
                  </span>
                  <span className="vi-caption-index">
                    caption {assessment.caption_index + 1}
                  </span>
                </div>
                <blockquote>
                  {sample.captions[assessment.caption_index]?.text
                    ?? "Source caption unavailable."}
                </blockquote>
                <p>{assessment.visible_evidence}</p>
              </li>
            ))}
          </ol>
        )}
        <div className="vi-columns">
          <TextList title="Discrepancies" items={proposal.discrepancies} />
          <TextList title="Uncertain" items={proposal.uncertainties} />
        </div>
      </>
    );
  }

  if (proposal.kind === "ocr") {
    return (
      <>
        {proposal.regions.length === 0 ? (
          <p className="vi-empty">No legible text proposed.</p>
        ) : (
          <ol className="vi-ocr">
            {proposal.regions.map((region, index) => (
              <li key={`${region.text}:${index}`}>
                <code>{region.text}</code>
                <span>{region.location.replace(/_/g, " ")}</span>
                <span className={`vi-legibility ${region.legibility}`}>
                  {region.legibility}
                </span>
              </li>
            ))}
          </ol>
        )}
        <TextList title="Uncertain" items={proposal.uncertainties} />
      </>
    );
  }

  return (
    <>
      <p className="vi-summary">{proposal.answer}</p>
      <div className="vi-columns">
        <TextList title="Visible evidence" items={proposal.visible_evidence} />
        <TextList title="Uncertain" items={proposal.uncertainties} />
      </div>
    </>
  );
}

function downloadResult(result: VisionInspectResponse) {
  const body = JSON.stringify(result, null, 2) + "\n";
  const url = URL.createObjectURL(
    new Blob([body], { type: "application/json;charset=utf-8" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = [
    "cvde",
    result.sample_id,
    result.task,
    result.model.replace(/[^a-z0-9._-]+/gi, "-"),
    result.model_digest.slice(0, 12),
  ].join("-") + ".json";
  anchor.click();
  URL.revokeObjectURL(url);
}

function ResultCard({
  result,
  sample,
  onDetectorQuery,
}: {
  result: VisionInspectResponse;
  sample: SampleDetail;
  onDetectorQuery: (query: string) => void;
}) {
  const detect = uniqueTerms(detectorTerms(result.proposal)).slice(0, 8);
  return (
    <article className="vi-result">
      <header>
        <div>
          <strong>{result.model}</strong>
          <span>{result.model_family ?? "local VLM"} · {result.latency_ms.toLocaleString()} ms</span>
        </div>
        <span className="vi-proposal-chip">model proposal</span>
      </header>

      {result.question && (
        <div className="vi-list">
          <h4>Focused question</h4>
          <p>{result.question}</p>
        </div>
      )}
      <ProposalBody result={result} sample={sample} />
      <SearchTerms terms={proposalTerms(result.proposal)} />

      {detect.length > 0 && (
        <div className="vi-grounding-handoff">
          <span>Localize one proposal</span>
          <div className="vi-ground-actions">
            {detect.map((term) => (
              <button
                key={term}
                type="button"
                className="ghost"
                onClick={() => onDetectorQuery(groundingQuery(term))}
                title={`Ask the open-vocabulary detector to measure “${term}”; no box or mask is accepted automatically`}
              >
                Ground “{term}”
              </button>
            ))}
          </div>
          <small>
            Each phrase starts a measured detector → mask → human-review handoff.
          </small>
        </div>
      )}
      <div className="vi-result-actions">
        <button type="button" className="ghost" onClick={() => downloadResult(result)}>
          Download JSON
        </button>
      </div>

      <details className="vi-provenance">
        <summary>Exact run provenance</summary>
        <dl>
          <dt>Source decode</dt>
          <dd>
            {result.source.width}×{result.source.height} · {result.source.mode}
            {" · "}{result.source.byte_length.toLocaleString()} bytes
          </dd>
          <dt>Source SHA-256</dt><dd>{result.source.image_sha256}</dd>
          <dt>Model digest</dt><dd>{result.model_digest}</dd>
          <dt>Artifact</dt>
          <dd>
            {[result.parameter_size, result.quantization_level]
              .filter(Boolean).join(" · ") || "details unavailable"}
          </dd>
          <dt>Input SHA-256</dt><dd>{result.input_sha256}</dd>
          {result.question && (
            <><dt>Question</dt><dd>{result.question}</dd></>
          )}
          <dt>Contracts</dt>
          <dd>prompt v{result.prompt_version} · schema v{result.schema_version}</dd>
        </dl>
      </details>
    </article>
  );
}

export default function VisionInspector({ sample, onDetectorQuery }: Props) {
  const [catalog, setCatalog] = useState<VisionModelStatus[] | null>(null);
  const [defaultModel, setDefaultModel] = useState<string | null>(null);
  const [model, setModel] = useState("");
  const [task, setTask] = useState<VisionTask>("scene");
  const [question, setQuestion] = useState("");
  const [results, setResults] = useState<Record<string, VisionInspectResponse>>({});
  const [running, setRunning] = useState<string | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const runAbort = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setCatalog(null);
    setCatalogError(null);
    api.visionModels(controller.signal)
      .then((response) => {
        setCatalog(response.models);
        setDefaultModel(response.default_model ?? null);
        const ready = response.models.filter((item) => item.ready);
        setModel((current) => (
          ready.some((item) => item.name === current)
            ? current
            : response.default_model ?? ready[0]?.name ?? ""
        ));
      })
      .catch((error: unknown) => {
        if (!isAbort(error)) setCatalogError(problem(error));
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    runAbort.current?.abort();
    setResults({});
    setRunError(null);
    setRunning(null);
  }, [sample.id]);

  const readyModels = useMemo(
    () => (catalog ?? []).filter((item) => item.ready),
    [catalog],
  );
  const normalizedQuestion = normalizeQuestion(question);
  const taskResults = readyModels
    .map((item) => results[resultKey(task, item.name, normalizedQuestion)])
    .filter((item): item is VisionInspectResponse => Boolean(item));
  const activeTask = TASKS.find((item) => item.value === task) ?? TASKS[0];
  const canRun = Boolean(
    model
    && running === null
    && (task !== "question" || normalizedQuestion),
  );

  const run = async () => {
    if (!canRun) return;
    runAbort.current?.abort();
    const controller = new AbortController();
    runAbort.current = controller;
    setRunning(model);
    setRunError(null);
    try {
      const result = await api.inspectVision(
        {
          sample_id: sample.id,
          model,
          task,
          ...(task === "question" ? { question: normalizedQuestion } : {}),
        },
        controller.signal,
      );
      setResults((current) => ({
        ...current,
        [resultKey(result.task, result.model, result.question)]: result,
      }));
    } catch (error: unknown) {
      if (!isAbort(error)) setRunError(problem(error));
    } finally {
      if (!controller.signal.aborted) setRunning(null);
    }
  };

  return (
    <section className="vision-inspector" aria-labelledby="vision-inspector-title">
      <header className="vi-head">
        <div>
          <span className="vi-eyebrow">Local vision lab</span>
          <h2 id="vision-inspector-title">Inspect, then act</h2>
        </div>
        <span className="vi-readonly">read-only inference</span>
      </header>
      <p className="vi-intro">
        Compare a structured VLM proposal, then hand useful concepts to measured
        search or the detector → mask workflow. Nothing here becomes a label
        automatically.
      </p>

      <div className="vi-tasks" role="tablist" aria-label="Vision inspection task">
        {TASKS.map((item) => (
          <button
            key={item.value}
            type="button"
            role="tab"
            aria-selected={task === item.value}
            className={task === item.value ? "active" : ""}
            onClick={() => {
              setTask(item.value);
              setRunError(null);
            }}
          >
            {item.label}
          </button>
        ))}
      </div>
      <p className="vi-task-description">{activeTask.description}</p>

      {task === "question" && (
        <label className="vi-question">
          Focused image question
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            maxLength={500}
            rows={2}
            placeholder="e.g. Which visible factors could make this frame difficult to annotate?"
          />
        </label>
      )}

      {catalog === null && !catalogError && (
        <div className="vi-loading" role="status">Checking local vision models…</div>
      )}
      {catalogError && <div className="vi-error" role="alert">{catalogError}</div>}
      {catalog !== null && readyModels.length === 0 && (
        <div className="vi-unavailable">
          <strong>No configured vision model is ready.</strong>
          {(catalog ?? []).map((item) => (
            <p key={item.name}><code>{item.name}</code> — {item.reason}</p>
          ))}
        </div>
      )}

      {readyModels.length > 0 && (
        <>
          <fieldset className="vi-models">
            <legend>Local model</legend>
            {readyModels.map((item) => (
              <label key={item.name} className={model === item.name ? "selected" : ""}>
                <input
                  type="radio"
                  name={`vision-model-${sample.id}`}
                  value={item.name}
                  checked={model === item.name}
                  onChange={() => setModel(item.name)}
                />
                <span>
                  <strong>{item.name}</strong>
                  <small>
                    {[item.parameter_size, item.quantization_level]
                      .filter(Boolean).join(" · ")}
                    {item.name === defaultModel ? " · default" : ""}
                  </small>
                  <code title={item.digest ?? undefined}>{compactDigest(item.digest)}</code>
                </span>
              </label>
            ))}
          </fieldset>

          <div className="vi-run-row">
            <button type="button" className="primary" onClick={() => void run()}
                    disabled={!canRun}>
              {running
                ? `Running ${running}…`
                : `Run ${taskLabel(task)} with ${model}`}
            </button>
            <span>Run the second model to compare proposals on the same pixels.</span>
          </div>
        </>
      )}

      {runError && <div className="vi-error" role="alert">{runError}</div>}
      <div className="vi-live" role="status" aria-live="polite">
        {running ? `${running} is inspecting sample ${sample.id}.` : ""}
      </div>

      {taskResults.length > 0 && (
        <div className={`vi-results${taskResults.length > 1 ? " compare" : ""}`}>
          {taskResults.map((result) => (
            <ResultCard
              key={`${result.model_digest}:${result.input_sha256}`}
              result={result}
              sample={sample}
              onDetectorQuery={onDetectorQuery}
            />
          ))}
        </div>
      )}
    </section>
  );
}
