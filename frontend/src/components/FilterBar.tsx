import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { TagInfo } from "../api/types";

export interface Filters {
  split: string;
  tag: string;
  vlm_tag: string;
}

interface Props {
  filters: Filters;
  onChange: (f: Filters) => void;
}

export default function FilterBar({ filters, onChange }: Props) {
  const [tags, setTags] = useState<TagInfo[]>([]);
  const [vlmTags, setVlmTags] = useState<TagInfo[]>([]);

  useEffect(() => {
    api.tags().then(setTags).catch(() => setTags([]));
    api.vlmTags().then(setVlmTags).catch(() => setVlmTags([]));
  }, []);

  return (
    <>
      <select
        value={filters.split}
        onChange={(e) => onChange({ ...filters, split: e.target.value })}
        title="Split"
      >
        <option value="">All splits</option>
        <option value="train">train</option>
        <option value="validation">validation</option>
        <option value="test">test</option>
      </select>
      {tags.length > 0 && (
        <select
          value={filters.tag}
          onChange={(e) => onChange({ ...filters, tag: e.target.value })}
          title="My tags"
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
          title="VLM tags"
        >
          <option value="">All VLM tags</option>
          {vlmTags.map((t) => (
            <option key={t.name} value={t.name}>{t.name} ({t.count})</option>
          ))}
        </select>
      )}
    </>
  );
}
