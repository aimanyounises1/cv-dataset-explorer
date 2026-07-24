import { useMemo } from "react";

/** The gallery publishes its visible result order here so the sample page can
 * offer next/previous without re-running the query. Triage is a sequential
 * scan; returning to the grid for every item is what makes it slow. */
const RESULT_IDS_KEY = "cvde-result-ids";

export function saveResultOrder(ids: number[]): void {
  try {
    sessionStorage.setItem(RESULT_IDS_KEY, JSON.stringify(ids));
  } catch {
    // Non-essential: without it the sample page simply offers no next/prev.
  }
}

export interface Neighbours {
  prev: number | null;
  next: number | null;
  position: number | null; // 1-based position in the result list
  total: number;
}

/** Where the given sample sits in the last result list, if it is in one. */
export function useNeighbours(id: string | undefined): Neighbours {
  return useMemo(() => {
    const empty: Neighbours = { prev: null, next: null, position: null, total: 0 };
    if (!id) return empty;
    let ids: number[];
    try {
      ids = JSON.parse(sessionStorage.getItem(RESULT_IDS_KEY) ?? "[]") as number[];
    } catch {
      return empty;
    }
    if (!Array.isArray(ids)) return empty;
    const i = ids.indexOf(Number(id));
    if (i === -1) return { ...empty, total: ids.length };
    return {
      prev: i > 0 ? ids[i - 1] : null,
      next: i < ids.length - 1 ? ids[i + 1] : null,
      position: i + 1,
      total: ids.length,
    };
  }, [id]);
}
