/** Renders text with query terms highlighted (case-insensitive prefix match,
 * so Porter-stemmed keyword hits like "running" light up for query "run"). */
export default function Highlight({ text, terms }: { text: string; terms?: string[] | null }) {
  if (!terms || terms.length === 0) return <>{text}</>;
  const escaped = terms
    .filter((t) => t.length > 1)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (escaped.length === 0) return <>{text}</>;
  const re = new RegExp(`\\b(${escaped.join("|")})\\w*`, "gi");
  const parts: (string | JSX.Element)[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    parts.push(<mark key={key++}>{m[0]}</mark>);
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return <>{parts}</>;
}
