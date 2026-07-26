import { Fragment, ReactNode } from "react";

/**
 * The small subset of Markdown a chat model actually emits, rendered as React
 * nodes.
 *
 * Without this, a reply reads `**train (75%)**` with the asterisks visible,
 * which looks like the interface is broken rather than like emphasis. A full
 * Markdown renderer would be ~40 kB of bundle and a much larger surface to trust
 * with model output; what models put in a sentence is bold, italics and inline
 * code, so that is what this handles.
 *
 * Deliberately not `dangerouslySetInnerHTML`. Model output is untrusted text —
 * it can and does contain whatever appeared in a caption — so it is only ever
 * placed in text nodes. Nothing here can produce markup.
 *
 * Block-level Markdown (headings, lists, tables) is out of scope on purpose: a
 * table belongs in a table block on the canvas, where it can be sorted and
 * filtered, not flattened into a paragraph.
 */

// One pass, alternation ordered so `**` wins over `*`. Underscore italics
// require a word boundary, because snake_case identifiers (`time_of_day`,
// `max_agreement`) appear constantly in this domain and must not turn italic.
const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*|\b_[^_\n]+_\b)/g;

function renderInline(text: string, keyBase: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let i = 0;
  for (const match of text.matchAll(INLINE)) {
    const at = match.index ?? 0;
    if (at > last) out.push(text.slice(last, at));
    const token = match[0];
    const key = `${keyBase}-${i++}`;
    if (token.startsWith("**")) {
      out.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("`")) {
      out.push(<code key={key}>{token.slice(1, -1)}</code>);
    } else {
      out.push(<em key={key}>{token.slice(1, -1)}</em>);
    }
    last = at + token.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export default function InlineMarkdown({ text }: { text: string }) {
  // Blank lines separate paragraphs; single newlines are line breaks. Both occur
  // in model output and collapsing either loses the structure of a list-like
  // answer ("1. …\n2. …").
  const paragraphs = text.split(/\n{2,}/);
  return (
    <>
      {paragraphs.map((para, p) => (
        <p className="chat-para" key={p}>
          {para.split("\n").map((lineText, l) => (
            <Fragment key={l}>
              {l > 0 && <br />}
              {renderInline(lineText, `${p}-${l}`)}
            </Fragment>
          ))}
        </p>
      ))}
    </>
  );
}
