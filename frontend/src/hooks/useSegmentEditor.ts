import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type {
  DetectionProposalSource,
  DetectBox,
  DetectStatus,
  ObjectLabel,
  SampleCard,
  SegmentAnnotation,
  SegmentBox,
  SegmentPoint,
  SegmentResult,
  SegmentStatus,
} from "../api/types";

export type SegmentTool = "positive" | "negative" | "box";
export type EditorBusy =
  | "segment"
  | "detect"
  | "save"
  | "delete"
  | "search"
  | "region-positive"
  | "region-negative"
  | null;

export interface RegionResults {
  items: SampleCard[];
  message: string | null;
  role: "positive" | "negative" | "annotation";
  scoreBasis?: string | null;
}

const aborted = (error: unknown) =>
  error instanceof DOMException && error.name === "AbortError";

/** Keep the server's useful validator sentence while removing the transport
 * wrapper emitted by api/client. */
const problem = (error: unknown, fallback: string): string => {
  if (!(error instanceof Error)) return fallback;
  const text = error.message;
  const colon = text.indexOf(":");
  if (colon < 0) return text || fallback;
  try {
    const payload = JSON.parse(text.slice(colon + 1).trim()) as {
      detail?: string | { msg?: string }[];
    };
    const detail = Array.isArray(payload.detail)
      ? payload.detail[0]?.msg
      : payload.detail;
    return (detail || fallback).replace(/^Value error, /, "");
  } catch {
    return text || fallback;
  }
};

export function useSegmentEditor(sampleId: number) {
  const [tool, setTool] = useState<SegmentTool>("box");
  const [points, setPoints] = useState<SegmentPoint[]>([]);
  const [box, setBox] = useState<SegmentBox | null>(null);
  const [mask, setMask] = useState<SegmentResult | SegmentAnnotation | null>(null);
  const [segmentStatus, setSegmentStatus] = useState<SegmentStatus | null>(null);
  const [detectStatus, setDetectStatus] = useState<DetectStatus | null>(null);
  const [annotationsReady, setAnnotationsReady] = useState(true);
  const [annotations, setAnnotations] = useState<SegmentAnnotation[]>([]);
  const [taxonomy, setTaxonomy] = useState<ObjectLabel[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [labelName, setLabelName] = useState("");
  const [parentName, setParentName] = useState("");
  const [detectQuery, setDetectQuery] = useState("");
  const [proposals, setProposals] = useState<DetectBox[]>([]);
  const [detectorRun, setDetectorRun] = useState<{
    model: string;
    revision: string;
    queries: string;
  } | null>(null);
  const [proposalSource, setProposalSource] =
    useState<DetectionProposalSource | null>(null);
  const [proposalToken, setProposalToken] = useState<string | null>(null);
  const [results, setResults] = useState<RegionResults | null>(null);
  const [busy, setBusy] = useState<EditorBusy>(null);
  const [error, setError] = useState<string | null>(null);
  const [announcement, setAnnouncement] = useState(
    "Source ready for inspection. Enter annotation mode before adding prompts.",
  );

  const loadAbort = useRef<AbortController | null>(null);
  const segmentAbort = useRef<AbortController | null>(null);
  const detectAbort = useRef<AbortController | null>(null);
  const detectQueryRef = useRef(detectQuery);
  const mutationAbort = useRef<AbortController | null>(null);
  const searchAbort = useRef<AbortController | null>(null);
  const busyKind = useRef<EditorBusy>(null);
  const busyEpoch = useRef(0);

  /** One editor operation at a time. Aborting fetch does not cancel a sync
   * model forward already running on the server, so accepting more prompt
   * input would only queue stale SAM work behind the inference lock. The epoch
   * also prevents an old finally block from clearing a newer operation. */
  const beginBusy = useCallback((next: Exclude<EditorBusy, null>) => {
    if (busyKind.current !== null) return null;
    const token = ++busyEpoch.current;
    busyKind.current = next;
    setBusy(next);
    return token;
  }, []);

  const endBusy = useCallback((token: number) => {
    if (busyEpoch.current !== token) return;
    busyKind.current = null;
    setBusy(null);
  }, []);

  const resetBusy = useCallback(() => {
    busyEpoch.current += 1;
    busyKind.current = null;
    setBusy(null);
  }, []);

  const abortAll = useCallback(() => {
    loadAbort.current?.abort();
    segmentAbort.current?.abort();
    detectAbort.current?.abort();
    mutationAbort.current?.abort();
    searchAbort.current?.abort();
  }, []);

  useEffect(() => {
    abortAll();
    resetBusy();
    const controller = new AbortController();
    loadAbort.current = controller;

    setPoints([]);
    setBox(null);
    setMask(null);
    setProposals([]);
    setDetectorRun(null);
    setProposalSource(null);
    setProposalToken(null);
    setResults(null);
    setAnnotations([]);
    setTaxonomy([]);
    setSelectedId(null);
    setLabelName("");
    setParentName("");
    setError(null);
    setSegmentStatus(null);
    setDetectStatus(null);
    setAnnotationsReady(true);
    setAnnouncement("Loading segmentation tools and saved annotations.");

    void api.segmentStatus(controller.signal)
      .then((status) => {
        setSegmentStatus(status);
        setAnnouncement(status.ready
          ? "Segmenter ready. Enter annotation mode to add a point or box."
          : `Segmenter unavailable. ${status.reason ?? "Rectangle search remains available."}`);
        setTool(status.ready ? "positive" : "box");
      })
      .catch((e: unknown) => {
        if (aborted(e)) return;
        setSegmentStatus({
          ready: false,
          reason: "Segmentation service is unavailable; rectangle search remains available.",
        });
        setTool("box");
      });

    void api.detectStatus(controller.signal)
      .then(setDetectStatus)
      .catch((e: unknown) => {
        if (!aborted(e)) setDetectStatus({ ready: false, reason: "Detector unavailable" });
      });

    void api.listSegmentAnnotations(sampleId, controller.signal)
      .then((saved) => {
        setAnnotations(Array.isArray(saved) ? saved : []);
        setAnnotationsReady(true);
      })
      .catch((e: unknown) => {
        if (aborted(e)) return;
        setAnnotationsReady(false);
      });

    void api.objectLabels(controller.signal)
      .then((labels) => setTaxonomy(Array.isArray(labels) ? labels : []))
      .catch((e: unknown) => {
        if (!aborted(e)) setTaxonomy([]);
      });

    return () => controller.abort();
  }, [abortAll, resetBusy, sampleId]);

  const requestSegment = useCallback(async (
    nextPoints: SegmentPoint[], nextBox: SegmentBox | null,
  ) => {
    if (nextPoints.length === 0 && nextBox === null) {
      setMask(null);
      return;
    }
    if (!segmentStatus?.ready) {
      setMask(null);
      setAnnouncement("No segmenter is available. The box can still drive region search.");
      return;
    }
    const operation = beginBusy("segment");
    if (operation === null) return;
    segmentAbort.current?.abort();
    const controller = new AbortController();
    segmentAbort.current = controller;
    setError(null);
    setAnnouncement("Refining the mask from the current prompts.");
    try {
      const next = await api.segment({
        sample_id: sampleId,
        points: nextPoints,
        ...(nextBox ? { box: nextBox } : {}),
      }, controller.signal);
      setMask(next);
      setSelectedId(null);
      setAnnouncement(
        `Mask ready: ${Math.round(next.area_fraction * 100)}% of the image, `
        + `${Math.round(next.predicted_iou * 100)}% predicted IoU.`,
      );
    } catch (e: unknown) {
      if (!aborted(e)) {
        setMask(null);
        setError(problem(e, "Could not create a mask"));
        setAnnouncement("Mask generation failed.");
      }
    } finally {
      endBusy(operation);
    }
  }, [beginBusy, endBusy, sampleId, segmentStatus]);

  const addPoint = useCallback((x: number, y: number, label: 0 | 1) => {
    if (busyKind.current !== null) return;
    const next = [...points, { x, y, label }];
    setPoints(next);
    setSelectedId(null);
    void requestSegment(next, box);
  }, [box, points, requestSegment]);

  const commitBox = useCallback((next: SegmentBox) => {
    if (busyKind.current !== null) return;
    setBox(next);
    setSelectedId(null);
    setProposalSource(null);
    setProposalToken(null);
    void requestSegment(points, next);
  }, [points, requestSegment]);

  const useProposal = useCallback((proposal: DetectBox) => {
    if (busyKind.current !== null) return;
    const next = {
      x: proposal.x, y: proposal.y, w: proposal.w, h: proposal.h,
    };
    setProposals([]);
    setBox(next);
    setSelectedId(null);
    const leaf = proposal.label_name?.trim()
      || proposal.label.replace(/^(a|an)\s+/i, "").replace(/\.$/, "");
    const known = taxonomy.find(
      (item) => item.name.toLocaleLowerCase() === leaf.toLocaleLowerCase(),
    );
    const path = proposal.label_path?.length ? proposal.label_path : known?.path ?? [];
    const explicitParent = proposal.parent_name?.trim()
      || (path.length > 1 ? path[path.length - 2].trim() : "");
    // A detector proposal is a new draft identity, not just new geometry.
    // Retaining the previous draft's taxonomy here can save a valid mask under
    // the wrong class (for example, choosing a person box after a sidewalk
    // draft). The proposal label and its resolved parent must travel together.
    setLabelName(leaf);
    setParentName(explicitParent);
    setProposalSource(detectorRun ? {
      kind: "detector",
      model_id: detectorRun.model,
      model_revision: detectorRun.revision,
      queries: detectorRun.queries,
      original_label: proposal.label,
      proposed_label: leaf,
      score: proposal.score,
      box: next,
    } : null);
    setProposalToken(proposal.proposal_token);
    void requestSegment(points, next);
  }, [detectorRun, points, requestSegment, taxonomy]);

  const updateDetectQuery = useCallback((value: string) => {
    detectQueryRef.current = value;
    setDetectQuery(value);
    // Proposals are evidence for the exact query that produced them. Once the
    // query changes, leaving *unselected* boxes selectable would attach stale
    // evidence to a new user intent. A proposal already selected into the
    // current draft keeps its immutable token until geometry is changed,
    // undone, or cleared; changing the input for a future run must not silently
    // relabel that detector-origin draft as manual.
    setProposals([]);
    setDetectorRun(null);
  }, []);

  const suggest = useCallback(async () => {
    const operation = beginBusy("detect");
    if (operation === null) return;
    detectAbort.current?.abort();
    const controller = new AbortController();
    detectAbort.current = controller;
    setError(null);
    setProposals([]);
    setDetectorRun(null);
    setAnnouncement("Finding candidate boxes.");
    const requestedQuery = detectQuery.trim();
    try {
      const response = await api.detectRegions(
        sampleId, requestedQuery, controller.signal,
      );
      if (detectQueryRef.current.trim() !== requestedQuery) {
        setAnnouncement("Detector query changed; discarded the stale proposals.");
        return;
      }
      setProposals(response.boxes);
      setDetectorRun({
        model: response.model,
        revision: response.revision,
        queries: response.queries,
      });
      setAnnouncement(response.boxes.length
        ? `${response.boxes.length} detector proposals ready. Choose one to segment.`
        : "The detector proposed no regions for this query.");
    } catch (e: unknown) {
      if (!aborted(e)) {
        setError(problem(e, "Could not suggest regions"));
        setAnnouncement("Detector request failed.");
      }
    } finally {
      endBusy(operation);
    }
  }, [beginBusy, detectQuery, endBusy, sampleId]);

  const clearDraft = useCallback(() => {
    if (busyKind.current !== null) return;
    segmentAbort.current?.abort();
    setPoints([]);
    setBox(null);
    setMask(null);
    setSelectedId(null);
    setProposals([]);
    setDetectorRun(null);
    setProposalSource(null);
    setProposalToken(null);
    setResults(null);
    setError(null);
    setAnnouncement("Prompts cleared.");
  }, []);

  const undoPrompt = useCallback(() => {
    if (busyKind.current !== null) return;
    setSelectedId(null);
    if (points.length > 0) {
      const next = points.slice(0, -1);
      setPoints(next);
      void requestSegment(next, box);
      return;
    }
    if (box) {
      setBox(null);
      setProposalSource(null);
      setProposalToken(null);
      void requestSegment([], null);
    }
  }, [box, points, requestSegment]);

  const selectAnnotation = useCallback((annotation: SegmentAnnotation) => {
    if (busyKind.current !== null) return;
    segmentAbort.current?.abort();
    setSelectedId(annotation.id);
    setMask(annotation);
    setPoints(annotation.points ?? []);
    setBox(annotation.box ?? annotation.bbox ?? null);
    const name = annotation.label_name ?? annotation.label ?? "";
    setLabelName(name);
    setParentName(annotation.parent_name ?? "");
    setProposalSource(annotation.proposal_source ?? null);
    setProposalToken(null);
    setResults(null);
    setError(null);
    setAnnouncement(`Selected accepted annotation “${name || "unnamed"}”.`);
  }, []);

  const save = useCallback(async () => {
    const label = labelName.trim();
    const preview = mask && "preview_token" in mask ? mask : null;
    if (
      !preview
      || !preview.preview_token
      || !preview.mask_data_url
      || !label
      || (points.length === 0 && box === null)
    ) return;
    const operation = beginBusy("save");
    if (operation === null) return;
    mutationAbort.current?.abort();
    const controller = new AbortController();
    mutationAbort.current = controller;
    setError(null);
    setAnnouncement(`Accepting annotation “${label}”.`);
    try {
      const saved = await api.saveSegmentAnnotation(sampleId, {
        points,
        ...(box ? { box } : {}),
        label_name: label,
        parent_name: parentName.trim() || null,
        preview_token: preview.preview_token,
        mask_data_url: preview.mask_data_url,
        ...(proposalToken ? { proposal_token: proposalToken } : {}),
      }, controller.signal);
      setAnnotations((current) => [
        ...current.filter((item) => item.id !== saved.id),
        saved,
      ]);
      setAnnotationsReady(true);
      setSelectedId(saved.id);
      setMask(saved);
      setProposalSource(saved.proposal_source ?? null);
      setProposalToken(null);
      const savedName = saved.label_name ?? saved.label ?? label;
      setAnnouncement(`Accepted “${savedName}”.`);
      void api.objectLabels(controller.signal)
        .then((labels) => setTaxonomy(Array.isArray(labels) ? labels : []))
        .catch(() => {});
    } catch (e: unknown) {
      if (!aborted(e)) {
        setError(problem(e, "Could not save the annotation"));
        setAnnouncement("Annotation acceptance failed.");
      }
    } finally {
      endBusy(operation);
    }
  }, [
    beginBusy, box, endBusy, labelName, mask, parentName, points,
    proposalToken, sampleId,
  ]);

  const remove = useCallback(async (annotationId: number) => {
    const operation = beginBusy("delete");
    if (operation === null) return;
    mutationAbort.current?.abort();
    const controller = new AbortController();
    mutationAbort.current = controller;
    setError(null);
    try {
      await api.deleteSegmentAnnotation(annotationId, controller.signal);
      setAnnotations((current) => current.filter((item) => item.id !== annotationId));
      if (selectedId === annotationId) {
        setSelectedId(null);
        setMask(null);
        setPoints([]);
        setBox(null);
        setProposalSource(null);
        setProposalToken(null);
      }
      setAnnouncement("Annotation deleted.");
    } catch (e: unknown) {
      if (!aborted(e)) {
        setError(problem(e, "Could not delete the annotation"));
        setAnnouncement("Annotation delete failed.");
      }
    } finally {
      endBusy(operation);
    }
  }, [beginBusy, endBusy, selectedId]);

  const searchAnnotation = useCallback(async (annotationId: number) => {
    const operation = beginBusy("search");
    if (operation === null) return;
    searchAbort.current?.abort();
    const controller = new AbortController();
    searchAbort.current = controller;
    setError(null);
    setAnnouncement("Searching with the accepted segment.");
    try {
      const response = await api.searchByAnnotation(
        annotationId, 24, controller.signal,
      );
      setResults({
        items: response.items,
        message: response.message ?? null,
        role: "annotation",
        scoreBasis: response.score_basis,
      });
      setAnnouncement(`${response.items.length} segment matches ready.`);
    } catch (e: unknown) {
      if (!aborted(e)) {
        setError(problem(e, "Could not search with this annotation"));
        setAnnouncement("Segment search failed.");
      }
    } finally {
      endBusy(operation);
    }
  }, [beginBusy, endBusy]);

  const searchRegion = useCallback(async (
    role: "positive" | "negative", text: string,
  ) => {
    if (!box) return;
    const operation = beginBusy(
      role === "positive" ? "region-positive" : "region-negative",
    );
    if (operation === null) return;
    searchAbort.current?.abort();
    const controller = new AbortController();
    searchAbort.current = controller;
    setError(null);
    setAnnouncement(role === "positive"
      ? "Searching for images like the marked region."
      : "Searching away from the marked region.");
    try {
      const response = await api.searchByRegion({
        sample_id: sampleId,
        ...box,
        role,
        ...(role === "positive" && text.trim() ? { text: text.trim() } : {}),
        top_k: 18,
      }, controller.signal);
      setResults({
        items: response.items,
        message: response.message ?? null,
        role,
        scoreBasis: response.score_basis,
      });
      setAnnouncement(`${response.items.length} region matches ready.`);
    } catch (e: unknown) {
      if (!aborted(e)) {
        setError(problem(e, "Could not search the marked region"));
        setAnnouncement("Region search failed.");
      }
    } finally {
      endBusy(operation);
    }
  }, [beginBusy, box, endBusy, sampleId]);

  const setResolvedLabelName = useCallback((value: string) => {
    setLabelName(value);
    const match = taxonomy.find(
      (item) => item.name.toLocaleLowerCase() === value.trim().toLocaleLowerCase(),
    );
    if (match) {
      setParentName(match.path.length > 1 ? match.path[match.path.length - 2] : "");
    }
  }, [taxonomy]);

  const labelOptions = Array.from(new Set([
    ...taxonomy.map((item) => item.name),
    ...annotations.map((item) => item.label_name).filter(
      (value): value is string => Boolean(value),
    ),
  ])).sort();
  const parentIds = new Set(taxonomy.map((item) => item.parent_id).filter(
    (value): value is number => value != null,
  ));
  const parentOptions = Array.from(new Set([
    ...taxonomy.filter((item) => parentIds.has(item.id)).map((item) => item.name),
    ...annotations.map((item) => item.parent_name).filter(
      (value): value is string => Boolean(value),
    ),
  ])).sort();

  return {
    tool, setTool,
    points, box, mask,
    segmentStatus, detectStatus, annotationsReady,
    annotations, selectedId,
    labelName, setLabelName: setResolvedLabelName, parentName, setParentName,
    labelOptions, parentOptions,
    detectQuery, setDetectQuery: updateDetectQuery, proposals, proposalSource,
    results, busy, error, announcement,
    addPoint, commitBox, useProposal, suggest,
    clearDraft, undoPrompt, selectAnnotation,
    save, remove, searchAnnotation, searchRegion,
  };
}
