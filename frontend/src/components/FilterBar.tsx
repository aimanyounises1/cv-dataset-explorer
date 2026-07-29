import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { AttributeGroup, TagInfo } from "../api/types";
import { Listbox, Segmented } from "./controls";
import { useSplitCounts } from "../lib/activeProvider";

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
  // Reuses the overview promise the app already has in flight — no request of
  // its own, so the split sizes cost the rail nothing.
  const splitCounts = useSplitCounts();

  useEffect(() => {
    // Facets that fail to load must be distinguishable from facets that
    // simply have no data (the pickers hide themselves when empty).
    const fail = () => setFailed(true);
    api.tags().then(setTags).catch(fail);
    api.vlmTags().then(setVlmTags).catch(fail);
    api.coverage().then(setCoverage).catch(fail);
  }, []);

  return (
    <>
      {/* Four values fit the rail whole, so the split filter shows all of them
          — the current split is read at a glance instead of behind a click. */}
      <Segmented
        label="Filter by split"
        value={filters.split}
        options={["", "train", "validation", "test"].map((value) => ({
          value,
          label: value || "All",
          // Absent until the overview lands, and absent for good if it fails:
          // the control still filters, it just does not claim a size it has
          // not been told.
          figure: splitCounts?.[value]?.toLocaleString(),
        }))}
        onChange={(split) => onChange({ ...filters, split })}
      />
      {coverage.length > 0 && (
        // An *add* control, not a current-value control: it stays on the
        // placeholder after each pick and the selection rail's chips are where
        // the applied facets live, each removable on its own. A single-value
        // picker cannot show "night AND indoor", and pretending otherwise is
        // what let the address bar and the result count disagree.
        <Listbox
          label="Add an attribute filter"
          trigger={filters.attrs.length === 0
            ? "All attributes"
            : `${filters.attrs.length} attribute${filters.attrs.length > 1 ? "s" : ""} — add another`}
          options={coverage.flatMap((g) =>
            g.labels.map((l) => {
              const value = `${g.grp}:${l.label}`;
              return {
                value,
                label: l.label,
                figure: l.count.toLocaleString(),
                group: g.grp.replace(/_/g, " "),
                disabled: filters.attrs.includes(value),
              };
            }))}
          onPick={(v) => onChange({ ...filters, attrs: [...filters.attrs, v] })}
        />
      )}
      {tags.length > 0 && (
        <Listbox
          label="Filter by my tags"
          trigger={filters.tag || "All my tags"}
          selected={filters.tag}
          options={[
            { value: "", label: "All my tags" },
            ...tags.map((t) => ({
              value: t.name, label: t.name, figure: t.count.toLocaleString(),
            })),
          ]}
          onPick={(tag) => onChange({ ...filters, tag })}
        />
      )}
      {vlmTags.length > 0 && (
        <Listbox
          label="Filter by VLM tags"
          trigger={filters.vlm_tag || "All VLM tags"}
          selected={filters.vlm_tag}
          options={[
            { value: "", label: "All VLM tags" },
            ...vlmTags.map((t) => ({
              value: t.name, label: t.name, figure: t.count.toLocaleString(),
            })),
          ]}
          onPick={(vlm_tag) => onChange({ ...filters, vlm_tag })}
        />
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
