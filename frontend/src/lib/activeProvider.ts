import { useEffect, useState } from "react";
import { api } from "../api/client";

/** One truth for "which embedding model is ranking right now".
 *
 * Every label that names the ACTIVE retrieval model reads it from here —
 * hard-coding a model name into a tooltip is how the UI ends up crediting
 * SigLIP for a ranking Qwen produced. Ingest-time artifacts (the map, the
 * difficulty axes, caption agreement, zero-shot facets) are the deliberate
 * exception: they really are SigLIP-derived regardless of who serves queries,
 * and their labels say so statically.
 *
 * Memoized per page lifetime; falls back to a neutral phrase when the backend
 * predates the provider fields or the fetch fails — a wrong name is worse
 * than a generic one. */
export interface ActiveProvider {
  short: string;          // "SigLIP 2" | "Qwen3-VL" | neutral fallback
  model: string;          // full model id, "" when unknown
  provider: string | null;
  total: number | null;   // corpus size, for copy that counts the frames
}

const NEUTRAL: ActiveProvider = { short: "the embedding model", model: "",
                                  provider: null, total: null };
let memo: Promise<ActiveProvider> | null = null;

export function activeProvider(): Promise<ActiveProvider> {
  memo ??= api.overview().then((o): ActiveProvider => {
    const model = o.embed_model ?? "";
    const provider = o.embed_provider ?? null;
    const short = provider === "qwen3_vl" ? "Qwen3-VL"
      : provider === "siglip2" ? "SigLIP 2"
      : model || "the embedding model";
    return { short, model, provider, total: o.total_samples ?? null };
  }).catch(() => NEUTRAL);
  return memo;
}

/** Corpus size for copy that counts frames; null until known or on failure. */
export function useCorpusTotal(): number | null {
  const [total, setTotal] = useState<number | null>(null);
  useEffect(() => { activeProvider().then((p) => setTotal(p.total)); }, []);
  return total;
}

/** The short display name, for labels built during render. */
export function useActiveProviderName(): string {
  const [short, setShort] = useState(NEUTRAL.short);
  useEffect(() => { activeProvider().then((p) => setShort(p.short)); }, []);
  return short;
}
