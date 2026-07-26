import { useId, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { TableBlock, TableColumn, TableRow } from "../../api/blocks";

/** Rows are `list[dict]`: a cell can be a number, a string, a bool, or null.
 * Numbers in a numeric column are set with separators and a fraction cap so a
 * column of measurements lines up and does not claim precision it lacks. */
function cellText(v: unknown, numeric: boolean): string {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") {
    return Number.isFinite(v)
      ? v.toLocaleString(undefined, { maximumFractionDigits: numeric ? 4 : 20 })
      : "—";
  }
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/** Sorting a numeric column by its rendered text would put 10 before 9, so
 * numeric columns compare as numbers with blanks pushed to the end in both
 * directions — a missing measurement is not a small one. */
function compare(a: TableRow, b: TableRow, col: TableColumn, dir: number): number {
  const av = a[col.key];
  const bv = b[col.key];
  const aBlank = av === null || av === undefined || av === "";
  const bBlank = bv === null || bv === undefined || bv === "";
  if (aBlank || bBlank) return aBlank && bBlank ? 0 : (aBlank ? 1 : -1);
  if (col.numeric) {
    const an = Number(av);
    const bn = Number(bv);
    if (Number.isFinite(an) && Number.isFinite(bn)) return (an - bn) * dir;
  }
  return cellText(av, false).localeCompare(cellText(bv, false), undefined,
                                           { numeric: true, sensitivity: "base" }) * dir;
}

function drillOf(row: TableRow, key: string | null): string | null {
  if (key === null) return null;
  const v = row[key];
  return typeof v === "string" && v !== "" ? v : null;
}

export default function TableBlockView({ block }: { block: TableBlock }) {
  const navigate = useNavigate();
  const filterId = useId();
  const [sort, setSort] = useState<{ key: string; dir: number } | null>(null);
  const [query, setQuery] = useState("");

  const columns = block.columns;
  const drillKey = block.drill_key ?? null;

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (q === "") return block.rows;
    // Matching the *rendered* text, not the raw value, so what the reader sees
    // is what the box searches.
    return block.rows.filter((r) => columns.some(
      (c) => cellText(r[c.key], c.numeric === true).toLowerCase().includes(q)));
  }, [block.rows, columns, query]);

  const sorted = useMemo(() => {
    if (sort === null) return filtered;
    const col = columns.find((c) => c.key === sort.key);
    if (col === undefined) return filtered;
    return [...filtered].sort((a, b) => compare(a, b, col, sort.dir));
  }, [filtered, sort, columns]);

  const clickHeader = (key: string) =>
    setSort((prev) => (prev !== null && prev.key === key
      ? (prev.dir === 1 ? { key, dir: -1 } : null)   // asc → desc → unsorted
      : { key, dir: 1 }));

  const open = (drill: string) => navigate(`/?${drill}`);

  return (
    <div className="vblock-body">
      <div className="vblock-table-controls">
        <label className="vblock-table-filter" htmlFor={filterId}>
          <span className="vblock-table-filter-label">Filter</span>
          <input
            id={filterId} type="search" value={query} placeholder="any column…"
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <span className="vblock-table-count">
          {sorted.length === block.rows.length
            ? `${block.rows.length} rows`
            : `${sorted.length} of ${block.rows.length} rows`}
        </span>
      </div>

      <div className="vblock-table-scroll">
        <table className="vblock-table">
          <thead>
            <tr>
              {columns.map((c) => {
                const active = sort !== null && sort.key === c.key;
                const dir = active ? (sort.dir === 1 ? "ascending" : "descending") : "none";
                return (
                  <th
                    key={c.key}
                    aria-sort={dir}
                    className={c.numeric === true ? "num" : undefined}
                  >
                    <button
                      type="button"
                      className={`vblock-th-sort${active ? " active" : ""}`}
                      onClick={() => clickHeader(c.key)}
                      title={`Sort by ${c.label}`}
                    >
                      {c.label}
                      <span className="vblock-sort-caret" aria-hidden="true">
                        {active ? (sort.dir === 1 ? "▲" : "▼") : "↕"}
                      </span>
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => {
              const drill = drillOf(r, drillKey);
              const key = drill ?? `${i}:${cellText(r[columns[0]?.key ?? ""], false)}`;
              return (
                <tr
                  key={key}
                  className={drill !== null ? "drillable" : undefined}
                  // A row that navigates is a link, and has to be reachable and
                  // activatable without a mouse: `<tr>` cannot hold an `<a>`
                  // without breaking the table's row semantics, so it takes the
                  // role and the key handling instead.
                  role={drill !== null ? "link" : undefined}
                  tabIndex={drill !== null ? 0 : undefined}
                  title={drill !== null
                    ? `Open this slice in the gallery (?${drill})` : undefined}
                  onClick={drill !== null ? () => open(drill) : undefined}
                  onKeyDown={drill !== null ? (e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      open(drill);
                    }
                  } : undefined}
                >
                  {columns.map((c) => (
                    <td key={c.key} className={c.numeric === true ? "num" : undefined}>
                      {cellText(r[c.key], c.numeric === true)}
                    </td>
                  ))}
                </tr>
              );
            })}
            {sorted.length === 0 && (
              <tr>
                <td className="vblock-table-empty" colSpan={Math.max(1, columns.length)}>
                  No rows match “{query}”.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {drillKey !== null && (
        <p className="vblock-hint">Click a row to open that slice in the gallery.</p>
      )}
    </div>
  );
}
