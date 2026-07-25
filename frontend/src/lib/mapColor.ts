import type { MapPoint } from "../api/types";
import { SURFACE, categorical, sequential } from "./viz";

/**
 * How the embedding map is coloured.
 *
 * Cluster id was the only option and it is the least informative: a k-means
 * label is nominal, so "cluster 7" carries no ordering and cannot be read
 * against anything. The dimensions below are the ones a researcher would
 * actually ask the map about — where the model agrees least with its captions,
 * where the hard samples sit, whether a split occupies its own region.
 */
export type ColorMode =
  | "cluster" | "split" | "agreement"
  | "legibility" | "rarity" | "difficulty" | "clutter";

export const COLOR_MODES: { value: ColorMode; label: string; kind: "nominal" | "quantity" }[] = [
  { value: "cluster", label: "Cluster (k-means)", kind: "nominal" },
  { value: "split", label: "Split", kind: "nominal" },
  { value: "agreement", label: "Caption agreement", kind: "quantity" },
  { value: "difficulty", label: "Difficulty", kind: "quantity" },
  { value: "legibility", label: "Legibility", kind: "quantity" },
  { value: "rarity", label: "Rarity", kind: "quantity" },
  { value: "clutter", label: "Clutter", kind: "quantity" },
];

const SPLIT_INDEX: Record<string, number> = { train: 0, validation: 1, test: 2 };

/** Fixed so a colour means the same thing on every visit. */
export const NO_DATA = SURFACE.textFaint;

export interface Ramp {
  kind: "nominal" | "quantity";
  /** Quantity: the observed value range, for the legend's end labels. */
  domain?: [number, number];
  /** Nominal: the distinct categories present, in draw order. */
  categories?: { label: string; color: string }[];
}

/**
 * Build a colour function plus the legend that explains it.
 *
 * The two are returned together on purpose: a colour scale without a legend is
 * decoration, and every previous version of this map had exactly that.
 */
export function buildColorScale(points: MapPoint[], mode: ColorMode): {
  colorOf: (p: MapPoint) => string;
  ramp: Ramp;
} {
  if (mode === "cluster") {
    const ids = [...new Set(points.map((p) => p.cluster))].sort((a, b) => a - b);
    return {
      colorOf: (p) => categorical(p.cluster),
      ramp: {
        kind: "nominal",
        categories: ids.map((c) => ({ label: `#${c}`, color: categorical(c) })),
      },
    };
  }

  if (mode === "split") {
    const present = [...new Set(points.map((p) => p.split))].sort();
    return {
      colorOf: (p) => categorical(SPLIT_INDEX[p.split] ?? 9),
      ramp: {
        kind: "nominal",
        categories: present.map((s) => ({
          label: s, color: categorical(SPLIT_INDEX[s] ?? 9),
        })),
      },
    };
  }

  // Quantities. The domain is measured from the data rather than assumed: the
  // axes are 0-10 by construction but agreement is a cosine whose real range is
  // narrow (~0.05-0.30), and stretching the ramp across [0,1] would render the
  // whole map one flat colour.
  const values = points
    .map((p) => p[mode])
    .filter((v): v is number => typeof v === "number");
  if (values.length === 0) {
    return { colorOf: () => NO_DATA, ramp: { kind: "quantity", domain: [0, 0] } };
  }
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  return {
    colorOf: (p) => {
      const v = p[mode];
      return typeof v === "number" ? sequential((v - lo) / span) : NO_DATA;
    },
    ramp: { kind: "quantity", domain: [lo, hi] },
  };
}

/** Formats a legend end label without implying more precision than the value has. */
export function formatDomainValue(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(3);
}
