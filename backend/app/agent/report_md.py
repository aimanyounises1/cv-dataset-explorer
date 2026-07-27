"""Render a report block as Markdown.

A chart cannot survive the trip to a text file, so each block is rendered as the
data behind it: a bar chart becomes the table of its values, a histogram becomes
its bins, a flow diagram becomes its edges. That is the honest translation — the
alternative is an image the reader cannot check, or a sentence describing a
picture they cannot see.

Every rendered block keeps its `source` line. The provenance is the part most
worth carrying out of the tool, because a number in a downloaded file has lost
all the context the UI gave it.
"""
from typing import Any


def report_to_markdown(block: dict) -> str:
    if block.get("kind") != "report":
        return _block_md(block)
    out = [f"# {block.get('title', 'Report')}", ""]
    if block.get("source"):
        out += [f"*{block['source']}*", ""]
    if block.get("note"):
        out += [f"> {block['note']}", ""]

    sections = block.get("sections") or []
    if len(sections) > 1:
        out.append("## Contents")
        for s in sections:
            out.append(f"- {s.get('heading', '')}")
        out.append("")

    for section in sections:
        out.append(f"## {section.get('heading', '')}")
        out.append("")
        if section.get("text"):
            out += [section["text"], ""]
        for child in section.get("blocks") or []:
            out += [_block_md(child), ""]
    return "\n".join(out).rstrip() + "\n"


def _block_md(block: dict) -> str:
    kind = block.get("kind", "?")
    head = [f"### {block.get('title', kind)}", ""]
    body = _RENDERERS.get(kind, _unsupported)(block)
    tail = []
    if block.get("source"):
        tail += ["", f"*Source: {block['source']}*"]
    if block.get("note"):
        tail += ["", f"> {block['note']}"]
    return "\n".join(head + [body] + tail)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_(no rows)_"
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        # Pipes inside a cell would break the table; escaping beats truncating,
        # and captions in this dataset do occasionally contain one.
        out.append("| " + " | ".join(_fmt(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def _points_table(block: dict) -> str:
    series = block.get("series") or []
    if not series:
        return "_(no data)_"
    labels: list[str] = []
    for s in series:
        for p in s.get("points") or []:
            if p["label"] not in labels:
                labels.append(p["label"])
    headers = [block.get("x_label") or "category"] + [s.get("name", "value") for s in series]
    rows = []
    for label in labels:
        row: list[Any] = [label]
        for s in series:
            match = next((p for p in s.get("points") or [] if p["label"] == label), None)
            row.append(match["value"] if match else None)
        rows.append(row)
    return _md_table(headers, rows)


def _line_md(block: dict) -> str:
    series = block.get("series") or []
    xs: list[float] = []
    for s in series:
        for p in s.get("points") or []:
            if p["x"] not in xs:
                xs.append(p["x"])
    xs.sort()
    headers = [block.get("x_label") or "x"] + [s.get("name", "y") for s in series]
    rows = []
    for x in xs:
        row: list[Any] = [x]
        for s in series:
            match = next((p for p in s.get("points") or [] if p["x"] == x), None)
            row.append(match["y"] if match else None)
        rows.append(row)
    return _md_table(headers, rows)


def _pie_md(block: dict) -> str:
    points = block.get("points") or []
    total = sum(float(p.get("value", 0)) for p in points) or 1.0
    return _md_table(["category", "value", "share"],
                     [[p["label"], p["value"], f"{float(p['value']) / total:.1%}"]
                      for p in points])


def _histogram_md(block: dict) -> str:
    bins = block.get("bins") or []
    out = _md_table([block.get("x_label") or "bin", "count"],
                    [[f"{_fmt(b.get('lo'))}–{_fmt(b.get('hi'))}", b.get("count")]
                     for b in bins])
    if block.get("marker") is not None:
        label = block.get("marker_label") or "marker"
        out += f"\n\nMarker: **{label}** at {_fmt(block['marker'])}."
    return out


def _table_md(block: dict) -> str:
    cols = block.get("columns") or []
    keys = [c["key"] for c in cols]
    return _md_table([c.get("label", c["key"]) for c in cols],
                     [[r.get(k) for k in keys] for r in block.get("rows") or []])


def _stat_md(block: dict) -> str:
    items = block.get("items") or []
    return _md_table(["metric", "value", "note"],
                     [[i.get("label"), i.get("value"), i.get("hint")] for i in items])


def _flow_md(block: dict) -> str:
    labels = {n["id"]: (n.get("label") or n["id"]).replace("\n", " ")
              for n in block.get("nodes") or []}
    edges = block.get("edges") or []
    if not edges:
        return "\n".join(f"- {v}" for v in labels.values()) or "_(empty diagram)_"
    lines = []
    for e in edges:
        arrow = f" —{e['label']}→ " if e.get("label") else " → "
        lines.append(f"- {labels.get(e['src'], e['src'])}{arrow}"
                     f"{labels.get(e['dst'], e['dst'])}")
    return "\n".join(lines)


def _images_md(block: dict) -> str:
    ids = block.get("sample_ids") or []
    total = block.get("total")
    line = "Sample ids: " + ", ".join(str(i) for i in ids)
    if total and total > len(ids):
        line += f" _(showing {len(ids)} of {total})_"
    return line


def _report_md(block: dict) -> str:
    # A nested report is flattened one level rather than recursing without bound.
    return report_to_markdown(block)


def _qa_md(block: dict) -> str:
    flows = block.get("flows") or []
    out = [f"**{block.get('passed', 0)}/{block.get('total', 0)} workflows passed**", ""]
    out.append(_md_table(["workflow", "status", "checks", "duration"],
                         [[f.get("name"), f.get("status"),
                           f"{sum(1 for c in f.get('checks') or [] if c.get('ok'))}"
                           f"/{len(f.get('checks') or [])}",
                           f"{f['duration_s']:.1f}s" if f.get("duration_s") else "—"]
                          for f in flows]))
    failing = [f for f in flows if f.get("status") == "fail"]
    if failing:
        out += ["", "#### Failures"]
        for f in failing:
            out.append(f"- **{f.get('name')}**: " + "; ".join(
                c.get("name", "") for c in f.get("checks") or [] if not c.get("ok")))
    if block.get("console_errors"):
        out += ["", "#### Console errors"]
        out += [f"- `{e}`" for e in block["console_errors"]]
    return "\n".join(out)


def _unsupported(block: dict) -> str:
    return f"_(no Markdown rendering for block kind `{block.get('kind')}`)_"


_RENDERERS = {
    "bar": _points_table,
    "line": _line_md,
    "pie": _pie_md,
    "histogram": _histogram_md,
    "table": _table_md,
    "stat": _stat_md,
    "flow": _flow_md,
    "images": _images_md,
    "report": _report_md,
    "qa": _qa_md,
}
