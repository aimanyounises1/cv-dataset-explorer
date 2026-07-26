import type { ReportBlock } from "../../api/blocks";
// A report contains blocks, so its renderer is the registry's own consumer.
// The cycle is safe because `BlockView` is a hoisted function declaration and is
// only dereferenced during render, long after both module bodies have run.
import { BlockView } from "./index";

export default function ReportBlockView({ block }: { block: ReportBlock }) {
  const md = block.download_md ?? null;
  const json = block.download_json ?? null;

  return (
    <div className="vblock-body vreport">
      {(md !== null || json !== null) && (
        <div className="vreport-downloads">
          {/* Real links, not fetch-and-blob buttons: a report is a file, and
              should be middle-clickable, copyable and saveable like one. */}
          {md !== null && (
            <a className="export-pill" href={md} download>markdown</a>
          )}
          {json !== null && (
            <a className="export-pill" href={json} download>json</a>
          )}
        </div>
      )}

      {block.sections.map((section, i) => (
        // The first section open, the rest closed: a report is read top-down,
        // and ten expanded sections of charts would bury the answer it opens
        // with.
        <details className="vreport-section" key={`${i}:${section.heading}`} open={i === 0}>
          <summary>
            <span className="vreport-heading">{section.heading}</span>
            {(section.blocks?.length ?? 0) > 0 && (
              <span className="vreport-count">
                {section.blocks?.length} {section.blocks?.length === 1 ? "figure" : "figures"}
              </span>
            )}
          </summary>
          {section.text && <p className="vreport-text">{section.text}</p>}
          {section.blocks?.map((child, ci) => (
            <BlockView block={child} key={`${ci}:${child.kind}:${child.title}`} />
          ))}
        </details>
      ))}

      {block.sections.length === 0 && (
        <div className="vblock-empty">This report has no sections.</div>
      )}
    </div>
  );
}
