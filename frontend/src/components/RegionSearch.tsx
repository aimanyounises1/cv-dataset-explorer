import {
  KeyboardEvent, PointerEvent, useCallback, useEffect, useId, useRef, useState,
} from "react";
import type { DetectBox, SegmentBox } from "../api/types";
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

/** Promptable mask editing and the legacy rectangle-retrieval lane share one
 * normalized image stage. When the segmenter is absent, box drawing, detector
 * proposals and region search continue to work without pretending a mask was
 * produced. */
export default function RegionSearch({ sampleId, imageUrl, alt }: Props) {
  const editor = useSegmentEditor(sampleId);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const instructionsId = useId();
  const [dragStart, setDragStart] = useState<Point | null>(null);
  const [pendingBox, setPendingBox] = useState<SegmentBox | null>(null);
  const [keyboardCursor, setKeyboardCursor] = useState<Point>({ x: 0.5, y: 0.5 });
  const [keyboardBoxStart, setKeyboardBoxStart] = useState<Point | null>(null);
  const [stageFocused, setStageFocused] = useState(false);
  const [steer, setSteer] = useState("");

  useEffect(() => {
    setDragStart(null);
    setPendingBox(null);
    setKeyboardBoxStart(null);
    setKeyboardCursor({ x: 0.5, y: 0.5 });
    setSteer("");
  }, [sampleId]);

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
    if (event.button !== 0 || editor.busy !== null) return;
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
  }, [editor, normalizedPoint]);

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
    if (event.target !== event.currentTarget || editor.busy !== null) return;
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
  }, [commitKeyboardPrompt, editor, keyboardBoxStart, moveKeyboardCursor]);

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
  const canSave = Boolean(
    maskSrc && editor.labelName.trim()
    && (editor.points.length > 0 || editor.box)
    && editor.annotationsReady && editor.selectedId == null,
  );
  const anyBusy = editor.busy !== null;

  return (
    <section className="region-search" aria-labelledby={`${instructionsId}-title`}>
      <header className="rs-head">
        <div>
          <h2 id={`${instructionsId}-title`}>Segment and annotate</h2>
          <p className="rs-note">
            Guide the local model with keep/remove points or a box. Saved masks
            remain separate from the source image.
          </p>
        </div>
        <span className={`rs-status${editor.segmentStatus?.ready ? " ready" : ""}`}>
          {editor.segmentStatus === null
            ? "checking model…"
            : editor.segmentStatus.ready
              ? editor.segmentStatus.model ?? "segmenter ready"
              : "box search fallback"}
        </span>
      </header>

      {editor.segmentStatus?.ready === false && (
        <div className="notice rs-model-note">
          {editor.segmentStatus.reason
            ?? "Promptable segmentation is unavailable. Draw a box to search by region."}
        </div>
      )}

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
            id={`${instructionsId}-detect`}
            value={editor.detectQuery}
            onChange={(event) => editor.setDetectQuery(event.target.value)}
            maxLength={300}
            placeholder="person. dog. cat. horse. bicycle. car."
          />
          <button type="button" className="ghost" onClick={editor.suggest}
                  disabled={anyBusy || editor.detectQuery.trim().length < 3}>
            {editor.busy === "detect" ? "Detecting…" : "Suggest boxes"}
          </button>
        </div>
      )}

      <p className="sr-only" id={instructionsId}>
        The image editor is keyboard accessible. Arrow keys move the prompt
        cursor; Shift plus an arrow moves faster. Enter places a point. In box
        mode, Enter sets the first corner and a second Enter accepts the box.
        Keys 1, 2, and 3 switch tools. Delete undoes and Escape clears.
      </p>

      <div
        ref={stageRef}
        className={`rs-stage tool-${editor.tool}`}
        role="group"
        tabIndex={0}
        aria-label={`Segmentation editor for sample ${sampleId}`}
        aria-describedby={instructionsId}
        aria-busy={anyBusy}
        aria-disabled={anyBusy}
        onFocus={() => setStageFocused(true)}
        onBlur={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
            setStageFocused(false);
          }
        }}
        onKeyDown={handleStageKeyDown}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerCancel}
      >
        <img className="detail-image" src={imageUrl} alt={alt} draggable={false} />

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

        {boxToDraw && boxToDraw.w > 0 && boxToDraw.h > 0 && (
          <div className="rs-rect" aria-hidden="true" style={{
            left: `${boxToDraw.x * 100}%`,
            top: `${boxToDraw.y * 100}%`,
            width: `${boxToDraw.w * 100}%`,
            height: `${boxToDraw.h * 100}%`,
          }} />
        )}

        {editor.points.map((point, index) => (
          <span
            key={`${point.label}:${point.x}:${point.y}:${index}`}
            className={`rs-point ${point.label === 1 ? "positive" : "negative"}`}
            aria-hidden="true"
            style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
          />
        ))}

        {stageFocused && (
          <span
            className={`rs-key-cursor ${keyboardBoxStart ? "anchored" : ""}`}
            aria-hidden="true"
            style={{
              left: `${keyboardCursor.x * 100}%`,
              top: `${keyboardCursor.y * 100}%`,
            }}
          />
        )}

        {editor.proposals.map((proposal, index) => (
          <button
            key={`${proposal.label}:${proposal.x}:${proposal.y}:${index}`}
            type="button"
            className="rs-box"
            style={{
              left: `${proposal.x * 100}%`,
              top: `${proposal.y * 100}%`,
              width: `${proposal.w * 100}%`,
              height: `${proposal.h * 100}%`,
            }}
            aria-label={`Use ${proposal.label} proposal, ${Math.round(proposal.score * 100)} percent confidence`}
            title={`${proposal.label} — detector confidence ${Math.round(proposal.score * 100)}%`}
            onPointerDown={(event) => event.stopPropagation()}
            onClick={() => chooseProposal(proposal)}
            disabled={anyBusy}
          >
            <span>{proposal.label} {Math.round(proposal.score * 100)}%</span>
          </button>
        ))}

        {editor.busy === "segment" && (
          <div className="rs-busy" role="status">Refining mask…</div>
        )}
      </div>

      <div className="rs-live" role="status" aria-live="polite">
        {editor.announcement}
      </div>
      {editor.error && <div className="rs-error" role="alert">{editor.error}</div>}

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
            </dl>
          )}

          <div className="rs-primary-actions">
            <button type="button" className="primary" onClick={editor.save}
                    disabled={!canSave || anyBusy}>
              {editor.busy === "save" ? "Saving…" : "Save annotation"}
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
                  ? "Searching…" : "Find similar to box"}
              </button>
              <button type="button" className="ghost" onClick={doNegativeRegionSearch}
                      disabled={anyBusy}>
                {editor.busy === "region-negative"
                  ? "Searching…" : "Search away from box"}
              </button>
            </div>
          )}
        </div>
      )}

      <section className="rs-saved" aria-labelledby={`${instructionsId}-saved`}>
        <div className="rs-saved-head">
          <h3 id={`${instructionsId}-saved`}>Saved annotations</h3>
          <span>{editor.annotations.length}</span>
        </div>
        {editor.annotations.length === 0 ? (
          <p className="rs-empty">
            No saved masks for this image yet. Prompt a mask, name its class, and save it.
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
                      ? `${Math.round(annotation.predicted_iou * 100)}% IoU`
                      : "saved mask"}
                  </span>
                </button>
                <button type="button" className="ghost"
                        onClick={() => void editor.searchAnnotation(annotation.id)}
                        disabled={anyBusy}>
                  Search
                </button>
                <button type="button" className="ghost danger"
                        onClick={() => confirmDelete(
                          annotation.id,
                          annotation.label_name ?? annotation.label ?? "unnamed",
                        )}
                        disabled={anyBusy}>
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {editor.results && (
        <section className="rs-results" aria-labelledby={`${instructionsId}-results`}>
          <div className="rs-results-head">
            <h3 id={`${instructionsId}-results`}>
              {editor.results.role === "annotation"
                ? "Segment matches"
                : editor.results.role === "positive"
                  ? "Similar regions"
                  : "Away from this region"}
            </h3>
            <span>{editor.results.items.length}</span>
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
