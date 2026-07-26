import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { AttributeGroup, TagInfo } from "../api/types";

export interface Filters {
  split: string;
  tag: string;
  vlm_tag: string;
  /** Attribute facets, intersected. Several, because "night" and "indoor" is a
   * real question and one dropdown value cannot express it. */
  attrs: string[]; // each "group:label"
}

interface Props {
  filters: Filters;
  onChange: (f: Filters) => void;
}

export default function FilterBar({ filters, onChange }: Props) {
  const [tags, setTags] = useState<TagInfo[]>([]);
  const [vlmTags, setVlmTags] = useState<TagInfo[]>([]);
  const [coverage, setCoverage] = useState<AttributeGroup[]>([]);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    // Facets that fail to load must be distinguishable from facets that
    // simply have no data (the dropdowns hide themselves when empty).
    const fail = () => setFailed(true);
    api.tags().then(setTags).catch(fail);
    api.vlmTags().then(setVlmTags).catch(fail);
    api.coverage().then(setCoverage).catch(fail);
  }, []);

  return (
    <>
      <select
        value={filters.split}
        onChange={(e) => onChange({ ...filters, split: e.target.value })}
        aria-label="Filter by split"
      >
        <option value="">All splits</option>
        <option value="train">train</option>
        <option value="validation">validation</option>
        <option value="test">test</option>
      </select>
      {coverage.length > 0 && (
        // An *add* control, not a current-value control: it resets to the
        // placeholder after each pick and the selection rail's chips are where
        // the applied facets live, each removable on its own. A single-value
        // <select> cannot show "night AND indoor", and pretending otherwise is
        // what let the address bar and the result count disagree.
        <select
          value=""
          onChange={(e) => {
            if (e.target.value) {
              onChange({ ...filters, attrs: [...filters.attrs, e.target.value] });
            }
          }}
          aria-label="Add an attribute filter"
        >
          <option value="">
            {filters.attrs.length === 0
              ? "All attributes"
              : `${filters.attrs.length} attribute${filters.attrs.length > 1 ? "s" : ""} — add another`}
          </option>
          {coverage.map((g) => (
            <optgroup key={g.grp} label={g.grp.replace(/_/g, " ")}>
              {g.labels.map((l) => {
                const value = `${g.grp}:${l.label}`;
                return (
                  <option key={l.label} value={value}
                          disabled={filters.attrs.includes(value)}>
                    {l.label} ({l.count}){filters.attrs.includes(value) ? " ✓" : ""}
                  </option>
                );
              })}
            </optgroup>
          ))}
        </select>
      )}
      {tags.length > 0 && (
        <select
          value={filters.tag}
          onChange={(e) => onChange({ ...filters, tag: e.target.value })}
          aria-label="Filter by my tags"
        >
          <option value="">All my tags</option>
          {tags.map((t) => (
            <option key={t.name} value={t.name}>{t.name} ({t.count})</option>
          ))}
        </select>
      )}
      {vlmTags.length > 0 && (
        <select
          value={filters.vlm_tag}
          onChange={(e) => onChange({ ...filters, vlm_tag: e.target.value })}
          aria-label="Filter by VLM tags"
        >
          <option value="">All VLM tags</option>
          {vlmTags.map((t) => (
            <option key={t.name} value={t.name}>{t.name} ({t.count})</option>
          ))}
        </select>
      )}
      {failed && (
        <span className="pill warn-pill"
              title="Some filter facets failed to load from the API.">
          filters degraded
        </span>
      )}
    </>
  );
}
