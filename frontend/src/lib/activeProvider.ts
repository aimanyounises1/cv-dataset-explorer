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
  /** Images per split, as the profile reports them. Carried here because the
   * rail's split control wants to state its own sizes and this promise is
   * already in flight — a second fetch for a number already on the wire is
   * how a control ends up disagreeing with the page it filters. */
  splits: Record<string, number> | null;
}

const NEUTRAL: ActiveProvider = { short: "the embedding model", model: "",
                                  provider: null, total: null, splits: null };
let memo: Promise<ActiveProvider> | null = null;

export function activeProvider(): Promise<ActiveProvider> {
  memo ??= api.overview().then((o): ActiveProvider => {
    const model = o.embed_model ?? "";
    const provider = o.embed_provider ?? null;
    const short = provider === "qwen3_vl" ? "Qwen3-VL"
      : provider === "siglip2" ? "SigLIP 2"
      : model || "the embedding model";
    return { short, model, provider, total: o.total_samples ?? null,
             splits: o.splits ?? null };
  }).catch(() => NEUTRAL);
  return memo;
}

/** Images per split, plus the corpus total under the key "" — which is what
 * the split control calls its "All" option, so the caller needs no special
 * case. Null until known or on failure: a control that cannot count says
 * nothing rather than showing a zero it did not measure. */
export function useSplitCounts(): Record<string, number> | null {
  const [counts, setCounts] = useState<Record<string, number> | null>(null);
  useEffect(() => {
    activeProvider().then((p) => {
      if (!p.splits) return;
      setCounts({ ...p.splits, ...(p.total != null ? { "": p.total } : {}) });
    });
  }, []);
  return counts;
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
