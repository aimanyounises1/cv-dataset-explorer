import {
  KeyboardEvent, PointerEvent, useCallback, useEffect, useId, useRef, useState,
} from "react";
import { Link } from "react-router-dom";
import type { DetectBox, SegmentAnnotation, SegmentBox } from "../api/types";
import { useSegmentEditor } from "../hooks/useSegmentEditor";
import type { SegmentTool } from "../hooks/useSegmentEditor";
import ComboBox from "./ComboBox";
import { Segmented } from "./controls";
import ImageCard from "./ImageCard";
import "../styles/region.css";

interface Props {
  sampleId: number;
  imageUrl: string;
  alt: string;
  detectorSuggestion?: {
    query: string;
    token: number;
  } | null;
}

interface Point {
  x: number;
  y: number;
}

const TOOL_OPTIONS = [
  { value: "positive", label: "Keep point" },
  { value: "negative", label: "Remove point" },
  { value: "box", label: "Box" },
];

const clamp = (value: number) => Math.min(1, Math.max(0, value));

const rectFrom = (a: Point, b: Point): SegmentBox => ({
  x: Math.min(a.x, b.x),
  y: Math.min(a.y, b.y),
  w: Math.abs(a.x - b.x),
  h: Math.abs(a.y - b.y),
});

const usableBox = (box: SegmentBox) => box.w >= 0.02 && box.h >= 0.02;

function proposalDisplayLabel(
  proposal: DetectBox,
  index: number,
  proposals: DetectBox[],
) {
  const peers = proposals.filter((item) => item.label === proposal.label);
  if (peers.length < 2) return proposal.label;
  const occurrence = proposals
    .slice(0, index + 1)
    .filter((item) => item.label === proposal.label)
    .length;
  return `${proposal.label}-${occurrence}`;
}

const ZIP_SIGNATURE = [80, 75, 3, 4] as const;

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

async function downloadAnnotationExport(annotation: SegmentAnnotation) {
  if (!annotation.artifact_package_url) {
    throw new Error("This accepted mask has no reproducible artifact package.");
  }

  const packageUrl = new URL(annotation.artifact_package_url, window.location.href);
  if (packageUrl.origin !== window.location.origin) {
    throw new Error("The artifact package must resolve to this local application.");
  }

  const response = await fetch(packageUrl, {
    cache: "no-store",
    credentials: "same-origin",
  });
  if (!response.ok) {
    throw new Error(`The artifact package could not be built (HTTP ${response.status}).`);
  }

  const mediaType = response.headers.get("content-type")
    ?.split(";", 1)[0]
    .trim()
    .toLowerCase();
  if (mediaType !== "application/zip") {
    throw new Error(`Expected a ZIP package, but the server returned ${mediaType || "no media type"}.`);
  }
  const packageBytes = await response.arrayBuffer();
  const signature = new Uint8Array(
    packageBytes,
    0,
    Math.min(ZIP_SIGNATURE.length, packageBytes.byteLength),
  );
  if (
    signature.length !== ZIP_SIGNATURE.length
    || ZIP_SIGNATURE.some((byte, index) => signature[index] !== byte)
  ) {
    throw new Error("The downloaded artifact is not a valid ZIP package.");
  }

  const baseName = `cvde-sample-${annotation.sample_id}-annotation-${annotation.id}`;
  downloadBlob(
    new Blob([packageBytes], { type: mediaType }),
    `${baseName}.zip`,
  );
}

/** Promptable mask editing and the legacy rectangle-retrieval lane share one
 * normalized image stage. When the segmenter is absent, box drawing, detector
 * proposals and region search continue to work without pretending a mask was
 * produced. */
export default function RegionSearch({
  sampleId,
  imageUrl,
  alt,
  detectorSuggestion = null,
}: Props) {
  const editor = useSegmentEditor(sampleId);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const toolsRef = useRef<HTMLDetailsElement | null>(null);
  const detectorInputRef = useRef<HTMLInputElement | null>(null);
  const instructionsId = useId();
  const [dragStart, setDragStart] = useState<Point | null>(null);
  const [pendingBox, setPendingBox] = useState<SegmentBox | null>(null);
  const [keyboardCursor, setKeyboardCursor] = useState<Point>({ x: 0.5, y: 0.5 });
  const [keyboardBoxStart, setKeyboardBoxStart] = useState<Point | null>(null);
  const [stageFocused, setStageFocused] = useState(false);
  const [annotationMode, setAnnotationMode] = useState(false);
  const [imageState, setImageState] = useState<"loading" | "decoded" | "failed">(
    "loading",
  );
  const [steer, setSteer] = useState("");
  const [exportingId, setExportingId] = useState<number | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);

  useEffect(() => {
    setDragStart(null);
    setPendingBox(null);
    setKeyboardBoxStart(null);
    setKeyboardCursor({ x: 0.5, y: 0.5 });
    setAnnotationMode(false);
    setImageState("loading");
    setSteer("");
    setExportingId(null);
    setExportError(null);
  }, [sampleId]);

  useEffect(() => {
    if (!detectorSuggestion) return undefined;
    editor.setDetectQuery(detectorSuggestion.query);
    if (imageState !== "decoded") return undefined;
    setAnnotationMode(true);
    const frame = requestAnimationFrame(() => {
      detectorInputRef.current?.scrollIntoView({ block: "center" });
      detectorInputRef.current?.focus();
    });
    return () => cancelAnimationFrame(frame);
  }, [detectorSuggestion, editor.setDetectQuery, imageState]);

  const normalizedPoint = useCallback((clientX: number, clientY: number): Point | null => {
    const stage = stageRef.current;
    if (!stage) return null;
    const bounds = stage.getBoundingClientRect();
    if (bounds.width === 0 || bounds.height === 0) return null;
    return {
      x: clamp((clientX - bounds.left) / bounds.width),
      y: clamp((clientY - bounds.top) / bounds.height),
    };
  }, []);

  const commitPointerBox = useCallback((box: SegmentBox | null) => {
    setDragStart(null);
    setPendingBox(null);
    if (box && usableBox(box)) editor.commitBox(box);
  }, [editor]);

  const handlePointerDown = useCallback((event: PointerEvent<HTMLDivElement>) => {
    if (!annotationMode || event.button !== 0 || editor.busy !== null) return;
    const point = normalizedPoint(event.clientX, event.clientY);
    if (!point) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
    if (editor.tool === "box") {
      setDragStart(point);
      setPendingBox({ x: point.x, y: point.y, w: 0, h: 0 });
      return;
    }
    editor.addPoint(point.x, point.y, editor.tool === "positive" ? 1 : 0);
  }, [annotationMode, editor, normalizedPoint]);

  const handlePointerMove = useCallback((event: PointerEvent<HTMLDivElement>) => {
    if (!dragStart) return;
    const point = normalizedPoint(event.clientX, event.clientY);
    if (point) setPendingBox(rectFrom(dragStart, point));
  }, [dragStart, normalizedPoint]);

  const handlePointerUp = useCallback((event: PointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const end = dragStart
      ? normalizedPoint(event.clientX, event.clientY)
      : null;
    commitPointerBox(dragStart && end ? rectFrom(dragStart, end) : pendingBox);
  }, [commitPointerBox, dragStart, normalizedPoint, pendingBox]);

  const handlePointerCancel = useCallback(() => {
    setDragStart(null);
    setPendingBox(null);
  }, []);

  const moveKeyboardCursor = useCallback((dx: number, dy: number) => {
    setKeyboardCursor((point) => ({
      x: clamp(point.x + dx),
      y: clamp(point.y + dy),
    }));
  }, []);

  const commitKeyboardPrompt = useCallback(() => {
    if (editor.tool !== "box") {
      editor.addPoint(
        keyboardCursor.x,
        keyboardCursor.y,
        editor.tool === "positive" ? 1 : 0,
      );
      return;
    }
    if (!keyboardBoxStart) {
      setKeyboardBoxStart(keyboardCursor);
      return;
    }
    const next = rectFrom(keyboardBoxStart, keyboardCursor);
    if (usableBox(next)) {
      editor.commitBox(next);
      setKeyboardBoxStart(null);
    }
  }, [editor, keyboardBoxStart, keyboardCursor]);

  const handleStageKeyDown = useCallback((event: KeyboardEvent<HTMLDivElement>) => {
    if (
      !annotationMode
      || event.target !== event.currentTarget
      || editor.busy !== null
    ) return;
    const step = event.shiftKey ? 0.1 : 0.02;
    if (event.key === "ArrowLeft") moveKeyboardCursor(-step, 0);
    else if (event.key === "ArrowRight") moveKeyboardCursor(step, 0);
    else if (event.key === "ArrowUp") moveKeyboardCursor(0, -step);
    else if (event.key === "ArrowDown") moveKeyboardCursor(0, step);
    else if (event.key === "Enter" || event.key === " ") commitKeyboardPrompt();
    else if (event.key === "Backspace" || event.key === "Delete") editor.undoPrompt();
    else if (event.key === "Escape") {
      if (keyboardBoxStart) setKeyboardBoxStart(null);
      else editor.clearDraft();
    } else if (event.key === "1") editor.setTool("positive");
    else if (event.key === "2") editor.setTool("negative");
    else if (event.key === "3") editor.setTool("box");
    else return;
    event.preventDefault();
  }, [
    annotationMode,
    commitKeyboardPrompt,
    editor,
    keyboardBoxStart,
    moveKeyboardCursor,
  ]);

  const handleTool = useCallback((value: string) => {
    editor.setTool(value as SegmentTool);
    setKeyboardBoxStart(null);
  }, [editor]);

  const chooseProposal = useCallback((proposal: DetectBox) => {
    editor.useProposal(proposal);
    setKeyboardBoxStart(null);
  }, [editor]);

  const confirmDelete = useCallback((id: number, label: string) => {
    if (window.confirm(`Delete the saved “${label}” annotation?`)) {
      void editor.remove(id);
    }
  }, [editor]);

  const doPositiveRegionSearch = useCallback(() => {
    void editor.searchRegion("positive", steer);
  }, [editor, steer]);

  const doNegativeRegionSearch = useCallback(() => {
    void editor.searchRegion("negative", "");
  }, [editor]);

  const exportAnnotation = useCallback(async (annotation: SegmentAnnotation) => {
    setExportingId(annotation.id);
    setExportError(null);
    try {
      await downloadAnnotationExport(annotation);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : "The annotation export failed.");
    } finally {
      setExportingId(null);
    }
  }, []);

  const boxToDraw = pendingBox
    ?? (keyboardBoxStart ? rectFrom(keyboardBoxStart, keyboardCursor) : editor.box);
  const maskSrc = editor.mask?.mask_data_url ?? editor.mask?.mask_url ?? null;
  const maskIou = editor.mask?.predicted_iou ?? null;
  const maskArea = editor.mask && "area_fraction" in editor.mask
    ? editor.mask.area_fraction
    : null;
  const maskModel = editor.mask
    ? ("model" in editor.mask ? editor.mask.model : editor.mask.model_id)
    : null;
  const toolOptions = editor.segmentStatus?.ready
    ? TOOL_OPTIONS
    : TOOL_OPTIONS.filter((option) => option.value === "box");
  const hasAcceptedPreview = Boolean(
    editor.mask
    && "preview_token" in editor.mask
    && editor.mask.preview_token
    && editor.mask.mask_data_url,
  );
  const canSave = Boolean(
    maskSrc && hasAcceptedPreview && editor.labelName.trim()
    && (editor.points.length > 0 || editor.box)
    && editor.annotationsReady && editor.selectedId == null,
  );
  const anyBusy = editor.busy !== null || exportingId !== null;
  const stageInteractive = annotationMode && imageState === "decoded";

  return (
    <section className="region-search" aria-labelledby={`${instructionsId}-title`}>
      <header className="rs-head">
        <div>
          <h2 id={`${instructionsId}-title`}>
            {annotationMode ? "Segment and annotate" : "Source image"}
          </h2>
        </div>
        <span className={`rs-status${imageState === "decoded" ? " ready" : ""}`}>
          {imageState === "loading"
            ? "loading source…"
            : imageState === "failed"
              ? "source unavailable"
              : annotationMode
                ? editor.segmentStatus?.ready
                  ? "annotation mode"
                  : "rectangle mode"
                : "browser decoded"}
        </span>
      </header>

      {annotationMode && editor.segmentStatus?.ready === false && (
        <div className="notice rs-model-note">
          {editor.segmentStatus.reason
            ?? "Promptable segmentation is unavailable. Draw a box to search by region."}
        </div>
      )}

      {/* The tools wait; the photograph does not.
          Measured, this panel put 316px of chrome above the subject at 1440 and
          535px at 390 — a full phone viewport of controls before the image they
          act on. Nothing here is needed until someone has decided to annotate,
          and most visits to a sample page are to look at it. Folded, not cut:
          one click reaches the whole toolset, and the disclosure keeps the
          reading order it draws in. */}
      <details
        className="caveat rs-tools"
        ref={toolsRef}
        open={annotationMode}
        onToggle={(event) => {
          const open = event.currentTarget.open;
          setAnnotationMode(open && imageState === "decoded");
          if (open && imageState !== "decoded") {
            event.currentTarget.open = false;
          }
          if (!open) {
            setDragStart(null);
            setPendingBox(null);
            setKeyboardBoxStart(null);
            setStageFocused(false);
          }
        }}
      >
        <summary>
          {annotationMode
            ? "Exit annotation mode"
            : "Enter annotation mode — points, box, and detector"}
        </summary>
        <p className="rs-note">
          Grounding DINO proposes boxes from an open-vocabulary phrase. Choose
          one instance, then guide SAM2 with keep/remove points or a box. Neither
          proposal becomes a label until you save it.
        </p>

        <dl className="rs-model-contract">
          <dt>Ground boxes</dt>
          <dd>
            {editor.detectStatus === null
              ? "checking local detector…"
              : editor.detectStatus.ready
                ? `${editor.detectStatus.model}@${
                  editor.detectStatus.revision?.slice(0, 10) ?? "unknown"
                }…`
                : editor.detectStatus.reason ?? "detector unavailable"}
          </dd>
          <dt>Refine mask</dt>
          <dd>
            {editor.segmentStatus === null
              ? "checking local segmenter…"
              : editor.segmentStatus.ready
                ? `${editor.segmentStatus.model}@${
                  editor.segmentStatus.revision?.slice(0, 10) ?? "unknown"
                }…`
                : "rectangle search only"}
          </dd>
        </dl>

      <div className="rs-toolbar" role="toolbar" aria-label="Segmentation tools">
        <Segmented
          value={editor.tool}
          options={toolOptions}
          onChange={handleTool}
          label="Prompt tool"
        />
        <button type="button" className="ghost" onClick={editor.undoPrompt}
                disabled={anyBusy || (editor.points.length === 0 && !editor.box)}>
          Undo prompt
        </button>
        <button type="button" className="ghost" onClick={editor.clearDraft}
                disabled={anyBusy
                  || (editor.points.length === 0 && !editor.box && !editor.mask)}>
          Clear draft
        </button>
        <span className="rs-tool-help">
          {editor.tool === "positive"
            ? "Click what belongs in the mask"
            : editor.tool === "negative"
              ? "Click spill to remove it"
              : "Drag a tight box around the object"}
        </span>
      </div>

      {editor.detectStatus?.ready && (
        <div className="rs-detector">
          <label htmlFor={`${instructionsId}-detect`}>Detector query</label>
          <input
            ref={detectorInputRef}
            id={`${instructionsId}-detect`}
            value={editor.detectQuery}
            onChange={(event) => editor.setDetectQuery(event.target.value)}
            maxLength={300}
            placeholder="e.g. person in a red coat. unusual hand-held object."
          />
          <button type="button" className="ghost" onClick={editor.suggest}
                  disabled={anyBusy || editor.detectQuery.trim().length < 3}>
            {editor.busy === "detect" ? "Detecting…" : "Suggest boxes"}
          </button>
        </div>
      )}
      </details>

      <p className="sr-only" id={instructionsId}>
        The image editor is keyboard accessible. Arrow keys move the prompt
        cursor; Shift plus an arrow moves faster. Enter places a point. In box
        mode, Enter sets the first corner and a second Enter accepts the box.
        Keys 1, 2, and 3 switch tools. Delete undoes and Escape clears.
      </p>

      <div
        ref={stageRef}
        className={`rs-stage ${
          stageInteractive ? `tool-${editor.tool}` : "inspection-only"
        }`}
        role={stageInteractive ? "group" : undefined}
        tabIndex={stageInteractive ? 0 : undefined}
        aria-label={stageInteractive
          ? `Segmentation editor for sample ${sampleId}`
          : undefined}
        aria-describedby={stageInteractive ? instructionsId : undefined}
        aria-busy={anyBusy}
        aria-disabled={stageInteractive ? anyBusy : undefined}
        onFocus={() => setStageFocused(true)}
        onBlur={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
            setStageFocused(false);
          }
        }}
        onKeyDown={stageInteractive ? handleStageKeyDown : undefined}
        onPointerDown={stageInteractive ? handlePointerDown : undefined}
        onPointerMove={stageInteractive ? handlePointerMove : undefined}
        onPointerUp={stageInteractive ? handlePointerUp : undefined}
        onPointerCancel={stageInteractive ? handlePointerCancel : undefined}
      >
        <img
          className="detail-image"
          src={imageUrl}
          alt={alt}
          draggable={false}
          onLoad={() => setImageState("decoded")}
          onError={() => {
            setImageState("failed");
            setAnnotationMode(false);
          }}
        />

        {imageState === "failed" && (
          <div className="rs-image-error" role="alert">
            <strong>Source image could not be decoded.</strong>
            <span>
              Sample #{sampleId} stays inspection-only; detection and
              segmentation were not run.
            </span>
          </div>
        )}

        {maskSrc && (
          <div
            className="rs-mask"
            aria-hidden="true"
            style={{
              maskImage: `url("${maskSrc}")`,
              WebkitMaskImage: `url("${maskSrc}")`,
            }}
          />
        )}

        {annotationMode && boxToDraw && boxToDraw.w > 0 && boxToDraw.h > 0 && (
          <div className="rs-rect" aria-hidden="true" style={{
            left: `${boxToDraw.x * 100}%`,
            top: `${boxToDraw.y * 100}%`,
            width: `${boxToDraw.w * 100}%`,
            height: `${boxToDraw.h * 100}%`,
          }} />
        )}

        {annotationMode && editor.points.map((point, index) => (
          <span
            key={`${point.label}:${point.x}:${point.y}:${index}`}
            className={`rs-point ${point.label === 1 ? "positive" : "negative"}`}
            aria-hidden="true"
            style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
          />
        ))}

        {annotationMode && stageFocused && (
          <span
            className={`rs-key-cursor ${keyboardBoxStart ? "anchored" : ""}`}
            aria-hidden="true"
            style={{
              left: `${keyboardCursor.x * 100}%`,
              top: `${keyboardCursor.y * 100}%`,
            }}
          />
        )}

        {annotationMode && editor.proposals.map((proposal, index) => {
          const displayLabel = proposalDisplayLabel(
            proposal,
            index,
            editor.proposals,
          );
          return (
          <button
            key={`${proposal.label}:${proposal.x}:${proposal.y}:${index}`}
            type="button"
            className={`rs-box${proposal.verified === false ? " unconfirmed" : ""}`}
            style={{
              left: `${proposal.x * 100}%`,
              top: `${proposal.y * 100}%`,
              width: `${proposal.w * 100}%`,
              height: `${proposal.h * 100}%`,
              // Proposals arrive confidence-sorted. A large low-confidence
              // region (often "sidewalk" or "background") must not paint over
              // and intercept the tighter, higher-confidence object button.
              zIndex: 5 + Math.round(proposal.score * 1_000),
            }}
            aria-label={proposal.verified === false
              ? `Use ${displayLabel} proposal — not confirmed; this region reads as ${proposal.best_alternative}`
              : `Use ${displayLabel} proposal, ${Math.round(proposal.score * 100)} percent confidence`}
            /* The detector grounds a phrase and has no way to answer "absent",
               so its own percentage is not evidence the thing is there. When a
               second model reads the crop and prefers something else, the box
               says so instead of printing the phrase back as a finding. */
            title={proposal.verified === false
              ? `${displayLabel} — NOT CONFIRMED. The detector grounds any phrase; `
                + `SigLIP reads this region as “${proposal.best_alternative}” `
                + `(${proposal.alternative_score}) over “${displayLabel}” `
                + `(${proposal.verified_score}).`
              : proposal.verified === true
                ? `${displayLabel} — confirmed by SigLIP (${proposal.verified_score}), `
                  + `ahead of “${proposal.best_alternative}” (${proposal.alternative_score}).`
                : `${displayLabel} — detector confidence ${Math.round(proposal.score * 100)}%, unchecked`}
            onPointerDown={(event) => event.stopPropagation()}
            onClick={() => chooseProposal(proposal)}
            disabled={anyBusy}
          >
            <span>
              {proposal.verified === false
                ? `not ${displayLabel} — reads as ${proposal.best_alternative}`
                : `${displayLabel} ${Math.round(proposal.score * 100)}%`}
            </span>
          </button>
          );
        })}

        {editor.busy === "segment" && (
          <div className="rs-busy" role="status">Refining mask…</div>
        )}
      </div>

      <div className="rs-live" role="status" aria-live="polite">
        {editor.announcement}
      </div>
      {editor.error && <div className="rs-error" role="alert">{editor.error}</div>}
      {exportError && <div className="rs-error" role="alert">{exportError}</div>}

      {(editor.mask || editor.box) && (
        <div className="rs-workbench">
          <div className="rs-fields">
            <ComboBox
              value={editor.labelName}
              onChange={editor.setLabelName}
              options={editor.labelOptions}
              label="Annotation class"
              placeholder="Class, e.g. dog"
            />
            <span className="rs-hierarchy-mark" aria-hidden="true">inside</span>
            <ComboBox
              value={editor.parentName}
              onChange={editor.setParentName}
              options={editor.parentOptions}
              label="Parent class"
              placeholder="Parent, e.g. animal"
            />
          </div>

          {editor.mask && (
            <dl className="rs-metrics">
              {maskIou != null && (
                <div>
                  <dt>Predicted IoU</dt>
                  <dd>{Math.round(maskIou * 100)}%</dd>
                </div>
              )}
              {maskArea != null && (
                <div>
                  <dt>Image area</dt>
                  <dd>{(maskArea * 100).toFixed(1)}%</dd>
                </div>
              )}
              {maskModel && (
                <div>
                  <dt>Model</dt>
                  <dd>{maskModel}</dd>
                </div>
              )}
              {editor.proposalSource && (
                <div>
                  <dt>Detector source</dt>
                  <dd title={[
                    editor.proposalSource.model_id,
                    editor.proposalSource.model_revision,
                    editor.proposalSource.queries,
                  ].join("\n")}>
                    {editor.proposalSource.original_label}
                    {" · "}
                    {Math.round(editor.proposalSource.score * 100)}%
                    {" · "}
                    {editor.proposalSource.model_revision.slice(0, 10)}…
                  </dd>
                </div>
              )}
            </dl>
          )}

          <div className="rs-primary-actions">
            <button type="button" className="primary" onClick={editor.save}
                    disabled={!canSave || anyBusy}>
              {editor.busy === "save" ? "Saving…" : "Accept & save"}
            </button>
            {editor.selectedId != null && (
              <button type="button" className="ghost"
                      onClick={() => void editor.searchAnnotation(editor.selectedId!)}
                      disabled={anyBusy}>
                {editor.busy === "search" ? "Searching…" : "Search accepted segment"}
              </button>
            )}
            {!editor.annotationsReady && (
              <span className="rs-inline-note">
                Annotation storage is unavailable; the draft remains local.
              </span>
            )}
          </div>

          {editor.box && (
            <div className="rs-region-fallback">
              <label htmlFor={`${instructionsId}-steer`}>Optional search steering</label>
              <input id={`${instructionsId}-steer`} value={steer}
                     onChange={(event) => setSteer(event.target.value)}
                     maxLength={200} placeholder="e.g. but at night" />
              <button type="button" className="ghost" onClick={doPositiveRegionSearch}
                      disabled={anyBusy}>
                {editor.busy === "region-positive"
                  ? "Searching…" : "Find images like this crop"}
              </button>
              <button type="button" className="ghost" onClick={doNegativeRegionSearch}
                      disabled={anyBusy}>
                {editor.busy === "region-negative"
                  ? "Searching…" : "Exclude this crop concept"}
              </button>
            </div>
          )}
        </div>
      )}

      {(annotationMode || editor.annotations.length > 0) && (
      <section className="rs-saved" aria-labelledby={`${instructionsId}-saved`}>
        <div className="rs-saved-head">
          <h3 id={`${instructionsId}-saved`}>Accepted annotations</h3>
          <span>{editor.annotations.length}</span>
        </div>
        {editor.annotations.length === 0 ? (
          <p className="rs-empty">
            No accepted masks for this image yet. Prompt a mask, name its class,
            then review and accept it.
          </p>
        ) : (
          <ul>
            {editor.annotations.map((annotation) => (
              <li key={annotation.id}
                  className={editor.selectedId === annotation.id ? "selected" : ""}>
                <button
                  type="button"
                  className="rs-ann-main"
                  aria-pressed={editor.selectedId === annotation.id}
                  onClick={() => editor.selectAnnotation(annotation)}
                  disabled={anyBusy}
                >
                  <span className="rs-ann-name">
                    {annotation.label_name ?? annotation.label ?? "unnamed"}
                  </span>
                  <span className="rs-ann-parent">
                    {annotation.parent_name ? `inside ${annotation.parent_name}` : "root class"}
                  </span>
                  <span className="rs-ann-score">
                    {annotation.predicted_iou != null
                      ? `${Math.round(annotation.predicted_iou * 100)}% predicted IoU`
                      : "accepted mask"}
                  </span>
                </button>
                <div className="rs-ann-actions">
                  <button type="button" className="ghost"
                          onClick={() => void editor.searchAnnotation(annotation.id)}
                          disabled={anyBusy}>
                    Search
                  </button>
                  {annotation.cutout_url && (
                    <a
                      className="ghost"
                      href={annotation.cutout_url}
                      download={`cvde-sample-${annotation.sample_id}-annotation-${annotation.id}-cutout.png`}
                      aria-disabled={anyBusy}
                      onClick={(event) => {
                        if (anyBusy) event.preventDefault();
                      }}
                      title="Download the mask-isolated object as a transparent RGBA PNG"
                    >
                      Export cutout
                    </a>
                  )}
                  {annotation.artifact_package_url && (
                    <button
                      type="button"
                      className="ghost"
                      onClick={() => void exportAnnotation(annotation)}
                      disabled={anyBusy}
                      title="Download one ZIP containing the accepted mask, transparent object cutout, and SHA-256-linked provenance manifest"
                    >
                      {exportingId === annotation.id
                        ? "Exporting…"
                        : "Evidence package"}
                    </button>
                  )}
                  <button type="button" className="ghost danger"
                          onClick={() => confirmDelete(
                            annotation.id,
                            annotation.label_name ?? annotation.label ?? "unnamed",
                          )}
                          disabled={anyBusy}>
                    Delete
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
      )}

      {editor.results && (
        <section className="rs-results" aria-labelledby={`${instructionsId}-results`}>
          <div className="rs-results-head">
            <h3 id={`${instructionsId}-results`}>
              {editor.results.role === "annotation"
                ? "Accepted-mask matches"
                : editor.results.role === "positive"
                  ? "Crop-similar images"
                  : "Images unlike this crop"}
            </h3>
            <div className="rs-results-actions">
              <span>{editor.results.items.length}</span>
              {editor.results.items.length > 0 && (
                <Link
                  className="open-all"
                  to={`/?ids=${editor.results.items.map((sample) => sample.id).join(",")}`}
                >
                  Open in gallery
                </Link>
              )}
            </div>
          </div>
          {editor.results.message && <p className="rs-note">{editor.results.message}</p>}
          <div className="grid rs-grid">
            {editor.results.items.map((sample) => (
              <ImageCard key={sample.id} sample={sample}
                         scoreBasis={editor.results?.scoreBasis} />
            ))}
          </div>
        </section>
      )}
    </section>
  );
}
