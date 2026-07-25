/**
 * Visualization primitives: one place for every colour, scale and chart token.
 *
 * Written by hand rather than pulled from d3-scale or chroma-js. The whole
 * surface needed is two interpolators and a categorical palette — about forty
 * lines — and the app has a hard local-only, dependency-light constraint. A
 * scale library would be the larger half of that trade.
 *
 * Colour choices are constrained, not decorative:
 *  - Quantities use **viridis**, which is perceptually uniform (equal steps in
 *    value look like equal steps in colour) and legible under all three common
 *    forms of colour blindness. A rainbow ramp is neither, and makes artefacts
 *    appear at its yellow/cyan boundaries that are not in the data.
 *  - Categories use a hand-picked qualitative set, ordered so adjacent entries
 *    stay distinguishable for deuteranopia.
 *  - Nothing encodes meaning in hue alone: every colour-carrying element also
 *    carries a number, a label, or a position.
 */

// ---------------------------------------------------------------- surfaces --

/** Mirrors the CSS custom properties in index.css. Duplicated here because
 * canvas and recharts cannot read CSS variables; index.css remains the source
 * of truth for anything the DOM styles. */
export const SURFACE = {
  bg: "#0b0e13",
  raised: "#141922",
  hover: "#1c232e",
  inset: "#080a0e",
  border: "#242c39",
  borderStrong: "#35404f",
  text: "#e9ecf2",
  textDim: "#939eae",
  textFaint: "#64707f",
  accent: "#5b9dff",
  green: "#3ecf8e",
  amber: "#f0b429",
  red: "#ff7a7a",
} as const;

// -------------------------------------------------------------- recharts ----

export const GRID_STROKE = SURFACE.border;
export const AXIS_STROKE = SURFACE.textDim;
export const TOOLTIP_STYLE = {
  background: SURFACE.raised,
  border: `1px solid ${SURFACE.borderStrong}`,
  borderRadius: 6,
  fontSize: 12,
};

/** Series colours for grouped charts. Distinct in hue *and* lightness, so they
 * survive greyscale printing and deuteranopia. */
export const SERIES = {
  blue: SURFACE.accent,
  green: SURFACE.green,
  purple: "#c792ea",
  amber: SURFACE.amber,
} as const;

// ----------------------------------------------------------------- scales ---

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function interpolate(stops: readonly [number, number, number][], t: number): string {
  const x = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  const seg = x * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(seg));
  const f = seg - i;
  const [r1, g1, b1] = stops[i];
  const [r2, g2, b2] = stops[i + 1];
  return `rgb(${Math.round(lerp(r1, r2, f))},${Math.round(lerp(g1, g2, f))},${
    Math.round(lerp(b1, b2, f))})`;
}

/** Viridis control points (Smith & van der Walt, public domain). */
const VIRIDIS: readonly [number, number, number][] = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
  [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
  [180, 222, 44], [253, 231, 37],
];

/** Perceptually-uniform ramp for a quantity in [0,1]. Low = deep violet,
 * high = yellow. Use for anything where "more" is meaningful. */
export function sequential(t: number): string {
  return interpolate(VIRIDIS, t);
}

/** Cool→warm diverging ramp for a signed quantity in [0,1] centred at 0.5.
 * Deliberately avoids red/green as its endpoints. */
const DIVERGING: readonly [number, number, number][] = [
  [59, 118, 175], [126, 168, 204], [200, 208, 216],
  [232, 168, 116], [214, 96, 77],
];
export function diverging(t: number): string {
  return interpolate(DIVERGING, t);
}

/** Categorical palette for nominal values such as cluster id or split. */
export const CATEGORICAL = [
  "#5b9dff", "#3ecf8e", "#f0b429", "#c792ea", "#4dd0e1",
  "#ff9d9d", "#aed581", "#ffb74d", "#90a4ae", "#7986cb",
  "#f48fb1", "#80cbc4",
] as const;

export function categorical(i: number): string {
  return CATEGORICAL[((i % CATEGORICAL.length) + CATEGORICAL.length) % CATEGORICAL.length];
}

// ------------------------------------------------- difficulty axis colours --

/** Difficulty axes are 0-10 ranks where high always means harder, so one ramp
 * serves all four. Four bands rather than a continuous scale: the underlying
 * value is a coarse percentile bucket, and pretending to more resolution than
 * that would be a false precision. */
export type Heat = "cool" | "mid" | "warm" | "hot";

export function heatBand(v: number): Heat {
  if (v >= 9) return "hot";
  if (v >= 7) return "warm";
  if (v >= 4) return "mid";
  return "cool";
}

export const HEAT_COLOR: Record<Heat, string> = {
  cool: SURFACE.textFaint,
  mid: SURFACE.textDim,
  warm: SURFACE.amber,
  hot: "#ff9d9d",
};

/** Continuous colour for a 0-10 axis value, for canvas use. */
export function axisColor(v: number | null | undefined): string {
  if (v == null) return SURFACE.textFaint;
  return sequential(v / 10);
}

// ------------------------------------------------------------- formatting ---

export function formatCount(n: number): string {
  return n.toLocaleString();
}

/** Ticks for a legend: evenly spaced samples of a ramp. */
export function rampStops(fn: (t: number) => string, n = 8): string[] {
  return Array.from({ length: n }, (_, i) => fn(i / (n - 1)));
}
