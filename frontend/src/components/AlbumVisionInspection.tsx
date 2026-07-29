import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, api } from "../api/client";
import type {
  AlbumDetail,
  SampleCard,
  VisionInspectResponse,
  VisionModelStatus,
} from "../api/types";
import "../styles/album-vision-inspection.css";

const RUN_CAP = 8;

type AlbumVisionTask = "scene" | "caption_audit";
type OutcomeStatus = "queued" | "running" | "proposed" | "failed" | "not_run";

interface InspectionOutcome {
  position: number;
  sample: SampleCard;
  status: OutcomeStatus;
  result?: VisionInspectResponse;
  error?: {
    status?: number;
    message: string;
  };
}

interface InspectionRun {
  runId: string;
  task: AlbumVisionTask;
  startedAt: string;
  completedAt?: string;
  stopped: boolean;
  snapshot: {
    albumId: number;
    albumName: string;
    updatedAt: string;
    totalCount: number;
    selectedIds: number[];
    sha256: string;
  };
  model: VisionModelStatus;
  outcomes: InspectionOutcome[];
}

const compactDigest = (digest?: string | null) =>
  digest ? `${digest.slice(0, 10)}…${digest.slice(-6)}` : "digest unavailable";

const isAbort = (error: unknown) =>
  error instanceof DOMException && error.name === "AbortError";

function errorRecord(error: unknown): NonNullable<InspectionOutcome["error"]> {
  if (error instanceof ApiError) {
    return { status: error.status, message: error.message };
  }
  return {
    message: error instanceof Error ? error.message : "Image inspection failed.",
  };
}

async function snapshotDigest(album: AlbumDetail, selectedIds: number[]) {
  const payload = [
    `album:${album.id}`,
    `updated:${album.updated_at}`,
    `total:${album.item_count}`,
    `ordered:${selectedIds.join(",")}`,
  ].join("\n");
  const digest = await window.crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(payload),
  );
  return Array.from(
    new Uint8Array(digest),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");
}

function visualSummary(result: VisionInspectResponse) {
  const proposal = result.proposal;
  if (proposal.kind === "scene") return proposal.summary;
  if (proposal.kind === "caption_audit") {
    const counts = new Map<string, number>();
    proposal.assessments.forEach((item) => {
      counts.set(item.status, (counts.get(item.status) ?? 0) + 1);
    });
    return ["supported", "partly_supported", "unsupported", "uncertain"]
      .filter((status) => counts.has(status))
      .map((status) => `${status.replace(/_/g, " ")} ${counts.get(status)}`)
      .join(" · ");
  }
  return "Structured image proposal ready for review.";
}

function detectorTerms(result: VisionInspectResponse) {
  if (result.proposal.kind !== "scene") return [];
  const seen = new Set<string>();
  return result.proposal.objects
    .map((item) => item.name)
    .map((term) => term.trim())
    .filter((term) => {
      const key = term.toLocaleLowerCase();
      if (!term || seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .slice(0, 4);
}

function groundingQuery(term: string) {
  return `${term.replace(/\.+$/g, "").trim()}.`.slice(0, 300);
}

function downloadRun(run: InspectionRun) {
  const payload = {
    format: "cvde.album-vision-inspection",
    version: 1,
    run_id: run.runId,
    epistemic_status: "model_proposals",
    task: run.task,
    started_at: run.startedAt,
    completed_at: run.completedAt ?? null,
    stopped_after_current_image: run.stopped,
    album_snapshot: {
      album_id: run.snapshot.albumId,
      album_name: run.snapshot.albumName,
      album_updated_at: run.snapshot.updatedAt,
      total_count: run.snapshot.totalCount,
      selected_ids: run.snapshot.selectedIds,
      sha256: run.snapshot.sha256,
    },
    model: run.model,
    items: run.outcomes.map((outcome) => ({
      position: outcome.position,
      sample: {
        id: outcome.sample.id,
        filename: outcome.sample.filename,
        split: outcome.sample.split,
      },
      status: outcome.status,
      result: outcome.result ?? null,
      error: outcome.error ?? null,
    })),
    note: (
      "Results are read-only model proposals over decoded local images. "
      + "No caption, tag, box, mask, or ground-truth annotation was changed."
    ),
  };
  const url = URL.createObjectURL(new Blob(
    [JSON.stringify(payload, null, 2) + "\n"],
    { type: "application/json;charset=utf-8" },
  ));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${run.runId}-album-vision.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function AlbumVisionInspection({ albumId }: { albumId: number }) {
  const [catalog, setCatalog] = useState<VisionModelStatus[] | null>(null);
  const [modelName, setModelName] = useState("");
  const [task, setTask] = useState<AlbumVisionTask>("scene");
  const [run, setRun] = useState<InspectionRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const stopRequested = useRef(false);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    setCatalog(null);
    setCatalogError(null);
    setRun(null);
    setRunError(null);
    setBusy(false);
    stopRequested.current = false;
    api.visionModels(controller.signal)
      .then((response) => {
        const ready = response.models.filter((item) => item.ready && item.digest);
        setCatalog(response.models);
        setModelName(
          response.default_model
          ?? ready[0]?.name
          ?? "",
        );
      })
      .catch((error: unknown) => {
        if (!isAbort(error)) {
          setCatalogError(error instanceof Error
            ? error.message
            : "Could not inspect local vision models.");
        }
      });
    return () => {
      mounted.current = false;
      controller.abort();
      stopRequested.current = true;
    };
  }, [albumId]);

  const readyModels = useMemo(
    () => (catalog ?? []).filter((item) => item.ready && item.digest),
    [catalog],
  );
  const selectedModel = readyModels.find((item) => item.name === modelName) ?? null;
  const completed = run?.outcomes.filter(
    (item) => item.status === "proposed" || item.status === "failed",
  ).length ?? 0;
  const proposed = run?.outcomes.filter((item) => item.status === "proposed").length ?? 0;
  const failed = run?.outcomes.filter((item) => item.status === "failed").length ?? 0;

  const publish = (next: InspectionRun) => {
    if (!mounted.current) return;
    setRun({
      ...next,
      outcomes: next.outcomes.map((outcome) => ({ ...outcome })),
    });
  };

  const start = async () => {
    if (!selectedModel || busy) return;
    setBusy(true);
    setRunError(null);
    setRun(null);
    stopRequested.current = false;
    try {
      // Refresh both mutable inputs immediately before freezing the run. A
      // model pull or album reorder between panel-open and Run must not inherit
      // the stale alias/digest/order that was merely displayed earlier.
      const [album, refreshedCatalog] = await Promise.all([
        api.albumDetail(albumId),
        api.visionModels(),
      ]);
      const refreshed = refreshedCatalog.models.find(
        (item) => item.name === selectedModel.name && item.ready && item.digest,
      );
      if (!refreshed) {
        setCatalog(refreshedCatalog.models);
        throw new Error(
          `The selected artifact ${selectedModel.name} is no longer ready. `
          + "Review the model status and run again.",
        );
      }
      if (refreshed.digest !== selectedModel.digest) {
        setCatalog(refreshedCatalog.models);
        throw new Error(
          `The digest for ${selectedModel.name} changed before execution. `
          + "The run was stopped before reading any image; review the new artifact.",
        );
      }
      const selected = album.items.slice(0, RUN_CAP);
      if (selected.length === 0) {
        throw new Error("This album is empty. Add images before visual inspection.");
      }
      const selectedIds = selected.map((sample) => sample.id);
      const sha256 = await snapshotDigest(album, selectedIds);
      const next: InspectionRun = {
        runId: `avir-${window.crypto.randomUUID()}`,
        task,
        startedAt: new Date().toISOString(),
        stopped: false,
        snapshot: {
          albumId: album.id,
          albumName: album.name,
          updatedAt: album.updated_at,
          totalCount: album.item_count,
          selectedIds,
          sha256,
        },
        model: refreshed,
        outcomes: selected.map((sample, position) => ({
          position,
          sample,
          status: "queued",
        })),
      };
      publish(next);

      for (let index = 0; index < next.outcomes.length; index += 1) {
        if (stopRequested.current) break;
        next.outcomes[index].status = "running";
        publish(next);
        try {
          const result = await api.inspectVision({
            sample_id: next.outcomes[index].sample.id,
            model: refreshed.name,
            task,
          });
          if (
            result.model !== refreshed.name
            || result.model_digest !== refreshed.digest
          ) {
            next.outcomes[index].status = "failed";
            next.outcomes[index].error = {
              message: (
                "The response artifact differs from the frozen run artifact. "
                + "Remaining images were not started."
              ),
            };
            stopRequested.current = true;
          } else {
            next.outcomes[index].status = "proposed";
            next.outcomes[index].result = result;
          }
        } catch (error: unknown) {
          const record = errorRecord(error);
          next.outcomes[index].status = "failed";
          next.outcomes[index].error = record;
          // A busy accelerator or one-image timeout is a run-level boundary.
          // Decode/output failures remain item-local so a corrupt member cannot
          // erase useful work on the images after it.
          if (record.status === 409 || record.status === 504) {
            stopRequested.current = true;
          }
        }
        publish(next);
      }

      next.stopped = stopRequested.current;
      next.outcomes.forEach((outcome) => {
        if (outcome.status === "queued") outcome.status = "not_run";
      });
      next.completedAt = new Date().toISOString();
      publish(next);
    } catch (error: unknown) {
      if (mounted.current) {
        setRunError(error instanceof Error
          ? error.message
          : "Album inspection could not start.");
      }
    } finally {
      if (mounted.current) setBusy(false);
    }
  };

  return (
    <section className="avi" aria-labelledby={`avi-title-${albumId}`}>
      <header className="avi-head">
        <div>
          <span className="avi-eyebrow">Sequential local run · maximum {RUN_CAP}</span>
          <h3 id={`avi-title-${albumId}`}>Inspect album pixels</h3>
        </div>
        <span className="avi-boundary">proposals only</span>
      </header>
      <p className="avi-intro">
        Freeze album order, decode each source, and inspect one image at a time.
        Failures remain beside successes. Review a proposal on its sample before
        grounding instances or refining a mask.
      </p>

      <div className="avi-controls">
        <label>
          Task
          <select
            value={task}
            onChange={(event) => setTask(event.target.value as AlbumVisionTask)}
            disabled={busy}
          >
            <option value="scene">Scene description</option>
            <option value="caption_audit">Audit stored captions</option>
          </select>
        </label>
        <label>
          Exact local artifact
          <select
            value={modelName}
            onChange={(event) => setModelName(event.target.value)}
            disabled={busy || readyModels.length === 0}
          >
            {readyModels.map((item) => (
              <option key={item.name} value={item.name}>{item.name}</option>
            ))}
          </select>
        </label>
        <div className="avi-model-proof">
          <span>Digest shown before execution</span>
          <code>{compactDigest(selectedModel?.digest)}</code>
        </div>
      </div>

      {catalog === null && !catalogError && (
        <p className="avi-note" role="status">Checking local vision artifacts…</p>
      )}
      {catalogError && <p className="avi-error" role="alert">{catalogError}</p>}
      {catalog !== null && readyModels.length === 0 && (
        <div className="avi-error" role="alert">
          <strong>No configured vision artifact is ready.</strong>
          {catalog.map((item) => (
            <span key={item.name}><code>{item.name}</code> — {item.reason}</span>
          ))}
        </div>
      )}
      {runError && <p className="avi-error" role="alert">{runError}</p>}

      <div className="avi-actions">
        <button
          type="button"
          className="ai-btn"
          onClick={() => void start()}
          disabled={busy || !selectedModel}
        >
          {busy ? `Inspecting ${completed + 1} of ${run?.outcomes.length ?? "…"}` : "Run inspection"}
        </button>
        {busy && (
          <button
            type="button"
            className="ghost"
            onClick={() => { stopRequested.current = true; }}
          >
            Stop after current image
          </button>
        )}
        {run?.completedAt && (
          <button type="button" className="ghost" onClick={() => downloadRun(run)}>
            Export ordered run manifest
          </button>
        )}
      </div>

      <div className="avi-live" role="status" aria-live="polite">
        {run && (
          busy
            ? `${completed} of ${run.outcomes.length} complete; ${proposed} proposals, ${failed} failures.`
            : `${completed} of ${run.outcomes.length} inspected; ${proposed} proposals, ${failed} failures.`
        )}
      </div>

      {run && (
        <>
          <ol className="avi-stages" aria-label="Album inspection stages">
            <li className="done">
              <span>1</span>
              <div><strong>Membership frozen</strong><small>{run.snapshot.selectedIds.length} of {run.snapshot.totalCount} images</small></div>
            </li>
            <li className={busy ? "active" : "done"}>
              <span>2</span>
              <div><strong>Decode + inspect</strong><small>{completed} / {run.outcomes.length}</small></div>
            </li>
            <li className={!busy && proposed > 0 ? "active" : ""}>
              <span>3</span>
              <div><strong>Human review</strong><small>open source before action</small></div>
            </li>
          </ol>

          <div className="avi-run-meta">
            <span><strong>Run</strong> <code>{run.runId}</code></span>
            <span><strong>Snapshot</strong> <code>{compactDigest(run.snapshot.sha256)}</code></span>
            <span><strong>Artifact</strong> <code>{run.model.name}@{compactDigest(run.model.digest)}</code></span>
          </div>

          <ol className="avi-results">
            {run.outcomes.map((outcome) => {
              const terms = outcome.result ? detectorTerms(outcome.result) : [];
              return (
                <li key={outcome.sample.id} className={`status-${outcome.status}`}>
                  <div className="avi-item-index">{outcome.position + 1}</div>
                  <Link
                    className="avi-thumb"
                    to={`/samples/${outcome.sample.id}`}
                    aria-label={`Review ${outcome.sample.filename}`}
                  >
                    <img
                      src={outcome.sample.thumb_url}
                      alt=""
                      loading="lazy"
                    />
                  </Link>
                  <div className="avi-item-main">
                    <header>
                      <Link to={`/samples/${outcome.sample.id}`}>
                        #{outcome.sample.id} · {outcome.sample.filename}
                      </Link>
                      <span>{outcome.status.replace("_", " ")}</span>
                    </header>
                    {outcome.result && (
                      <>
                        <p>{visualSummary(outcome.result)}</p>
                        <small>
                          decoded {outcome.result.source.width}×{outcome.result.source.height}
                          {" · "}{outcome.result.latency_ms.toLocaleString()} ms
                          {" · source "}{compactDigest(outcome.result.source.image_sha256)}
                        </small>
                      </>
                    )}
                    {outcome.error && (
                      <p className="avi-item-error">{outcome.error.message}</p>
                    )}
                    {outcome.status === "not_run" && (
                      <p className="avi-note">Not started after the run stopped.</p>
                    )}
                  </div>
                  <div className="avi-item-actions">
                    <Link className="ghost" to={`/samples/${outcome.sample.id}`}>
                      Review source
                    </Link>
                    {terms.map((term) => (
                      <Link
                        key={term}
                        className="ghost"
                        to={`/samples/${outcome.sample.id}?detector=${encodeURIComponent(groundingQuery(term))}`}
                        title={`Open this sample and measure “${term}” with the open-vocabulary detector`}
                      >
                        Ground “{term}”
                      </Link>
                    ))}
                  </div>
                </li>
              );
            })}
          </ol>
          <p className="avi-footnote">
            Scene summaries are visual-description proposals, not saved captions.
            Nothing becomes a label, box, mask, or caption until a reviewer acts
            on the sample page.
          </p>
        </>
      )}
    </section>
  );
}
