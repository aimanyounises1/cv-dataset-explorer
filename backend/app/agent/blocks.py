"""Render blocks: the contract between agent tools and the chat canvas.

A tool that has something to show returns *blocks* rather than prose. Each block
is a small validated payload with a `kind` discriminator; the frontend keeps one
renderer per kind in a registry, so adding a visualization type means adding one
builder here and one component there — no changes to the agent, the graph, the
chat endpoint, or any existing renderer.

Three rules keep the canvas honest, and they are enforced here rather than left
to each tool:

* **Every block names its source.** `source` says how the numbers were produced
  ("COUNT(*) over samples grouped by split"). A chart whose provenance is not
  stated is indistinguishable from a chart the model invented, and this platform
  shows measured data or says it cannot.
* **Every block is built through a builder.** The builders validate, so a
  malformed block fails at the tool call with a traceback, not silently in the
  browser as an empty box.
* **Series are capped.** A local model asked to "plot the tags" can name a
  dimension with thousands of levels. `cap_points` keeps the top N and folds the
  remainder into one honest "other (N levels)" entry instead of emitting a chart
  with 4,000 bars or truncating in silence.

`drill` is what makes the canvas part of the app rather than a picture of it: it
carries a gallery query string, so clicking a bar opens exactly the slice that
bar counts. It is the same "a view produces a set" contract the map and quality
pages use, extended to charts.
"""
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

# A chart with more categories than this is unreadable, and a local model cannot
# be trusted not to ask for one. The cap is applied by `cap_points`, which folds
# the tail into a labelled remainder rather than dropping it.
MAX_POINTS = 24
MAX_TABLE_ROWS = 200
MAX_IMAGES = 24


class Point(BaseModel):
    """One categorical value. `drill` opens the slice this point counts."""
    label: str
    value: float
    drill: Optional[str] = None          # gallery query string, e.g. "split=train"


class XY(BaseModel):
    x: float
    y: float


class Series(BaseModel):
    name: str
    points: list[Point] = []


class XYSeries(BaseModel):
    name: str
    points: list[XY] = []


class _Base(BaseModel):
    title: str
    # How the numbers were obtained. Required, deliberately: see module docstring.
    source: str
    note: Optional[str] = None


class BarBlock(_Base):
    kind: Literal["bar"] = "bar"
    series: list[Series]
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    # Horizontal when the category labels are long — which for this dataset is
    # most of them ("a man in a red shirt climbing").
    horizontal: bool = False
    stacked: bool = False


class LineBlock(_Base):
    kind: Literal["line"] = "line"
    series: list[XYSeries]
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    area: bool = False


class PieBlock(_Base):
    kind: Literal["pie"] = "pie"
    points: list[Point]


class HistogramBlock(_Base):
    kind: Literal["histogram"] = "histogram"
    bins: list[dict]                     # [{lo, hi, count}]
    x_label: Optional[str] = None
    # Where a meaningful cut sits, drawn as a reference line (e.g. a QA
    # threshold). A histogram without one invites the reader to guess.
    marker: Optional[float] = None
    marker_label: Optional[str] = None
    drill_param: Optional[str] = None    # e.g. "max_agreement" — brush → gallery


class TableColumn(BaseModel):
    key: str
    label: str
    numeric: bool = False


class TableBlock(_Base):
    kind: Literal["table"] = "table"
    columns: list[TableColumn]
    rows: list[dict]
    # Per-row query string under this key turns rows into links.
    drill_key: Optional[str] = None


class StatItem(BaseModel):
    label: str
    value: str
    hint: Optional[str] = None
    drill: Optional[str] = None


class StatBlock(_Base):
    kind: Literal["stat"] = "stat"
    items: list[StatItem]


class FlowNode(BaseModel):
    id: str
    label: str
    # Colours the node by role (e.g. "agent", "tool", "store"); the renderer maps
    # groups to the shared palette so a diagram matches the rest of the app.
    group: Optional[str] = None
    drill: Optional[str] = None


class FlowEdge(BaseModel):
    src: str
    dst: str
    label: Optional[str] = None


class FlowBlock(_Base):
    kind: Literal["flow"] = "flow"
    nodes: list[FlowNode]
    edges: list[FlowEdge] = []
    # Explicit left-to-right columns. Supplying the layering rather than deriving
    # it keeps the renderer to a few dozen lines of SVG and needs no graph-layout
    # dependency: mermaid would have cost ~800 kB of bundle for this one block
    # type, and the diagrams worth drawing here are all shallow pipelines whose
    # layering the producer already knows.
    layers: list[list[str]] = []


class ImagesBlock(_Base):
    kind: Literal["images"] = "images"
    sample_ids: list[int]
    # Set when the tool had more than MAX_IMAGES, so the UI can offer the rest
    # in the gallery instead of pretending this is everything.
    total: Optional[int] = None
    drill: Optional[str] = None


class ReportSection(BaseModel):
    heading: str
    text: Optional[str] = None
    blocks: list["Block"] = []


class ReportBlock(_Base):
    kind: Literal["report"] = "report"
    sections: list[ReportSection]
    # Filled in by the chat endpoint once the report is persisted; the tool that
    # builds a report does not know the URL it will be served from.
    download_md: Optional[str] = None
    download_json: Optional[str] = None


class QAFlowResult(BaseModel):
    name: str
    status: Literal["pass", "fail", "skip"]
    checks: list[dict] = []              # [{name, ok, detail}]
    screenshot: Optional[str] = None     # URL under /media/qa/
    detail: Optional[str] = None
    duration_s: Optional[float] = None


class QABlock(_Base):
    kind: Literal["qa"] = "qa"
    status: Literal["running", "done", "failed"]
    run_id: str
    flows: list[QAFlowResult] = []
    passed: int = 0
    total: int = 0
    console_errors: list[str] = []
    deck_url: Optional[str] = None
    markdown_url: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


Block = Annotated[
    Union[BarBlock, LineBlock, PieBlock, HistogramBlock, TableBlock, StatBlock,
          FlowBlock, ImagesBlock, ReportBlock, QABlock],
    Field(discriminator="kind"),
]

ReportSection.model_rebuild()

# Every kind the union admits. Tests assert the frontend registry covers exactly
# this set, so a new block type cannot ship without a renderer.
BLOCK_KINDS = ("bar", "line", "pie", "histogram", "table", "stat", "flow",
               "images", "report", "qa")


# --------------------------------------------------------------------- builders

def cap_points(points: list[dict], limit: int = MAX_POINTS) -> tuple[list[dict], int]:
    """Keep the `limit` largest points; fold the rest into one labelled entry.

    Returns (points, dropped). The remainder is a real point rather than a
    silent truncation: a pie chart of "the top 24 tags" that omits 4,000 others
    without saying so is a lie about the distribution, and the missing mass is
    usually the interesting part of a long-tailed dataset.
    """
    if len(points) <= limit:
        return points, 0
    ranked = sorted(points, key=lambda p: -abs(float(p.get("value", 0))))
    head, tail = ranked[:limit - 1], ranked[limit - 1:]
    head.append({"label": f"other ({len(tail)} levels)",
                 "value": sum(float(p.get("value", 0)) for p in tail)})
    return head, len(tail)


def _capped_note(dropped: int, note: Optional[str]) -> Optional[str]:
    if not dropped:
        return note
    extra = (f"{dropped} smaller categories folded into “other” — the chart is "
             f"capped at {MAX_POINTS} entries.")
    return f"{note} {extra}" if note else extra


def bar(title: str, source: str, points: list[dict], *, series_name: str = "count",
        x_label: Optional[str] = None, y_label: Optional[str] = None,
        horizontal: bool = False, note: Optional[str] = None) -> BarBlock:
    kept, dropped = cap_points(points)
    return BarBlock(title=title, source=source, x_label=x_label, y_label=y_label,
                    horizontal=horizontal, note=_capped_note(dropped, note),
                    series=[Series(name=series_name,
                                   points=[Point(**p) for p in kept])])


def grouped_bar(title: str, source: str, series: list[dict], *,
                x_label: Optional[str] = None, y_label: Optional[str] = None,
                note: Optional[str] = None) -> BarBlock:
    """Several named series over the same categories (e.g. R@1/R@5/R@10 per mode)."""
    return BarBlock(
        title=title, source=source, x_label=x_label, y_label=y_label, note=note,
        series=[Series(name=s["name"], points=[Point(**p) for p in s["points"]])
                for s in series])


def line(title: str, source: str, series: list[dict], *,
         x_label: Optional[str] = None, y_label: Optional[str] = None,
         area: bool = False, note: Optional[str] = None) -> LineBlock:
    return LineBlock(
        title=title, source=source, x_label=x_label, y_label=y_label, area=area,
        note=note,
        series=[XYSeries(name=s["name"], points=[XY(**p) for p in s["points"]])
                for s in series])


def pie(title: str, source: str, points: list[dict],
        note: Optional[str] = None) -> PieBlock:
    kept, dropped = cap_points(points, limit=10)   # a 24-slice pie is unreadable
    if dropped:
        note = _capped_note(dropped, note)
    return PieBlock(title=title, source=source, note=note,
                    points=[Point(**p) for p in kept])


def histogram(title: str, source: str, bins: list[dict], *,
              x_label: Optional[str] = None, marker: Optional[float] = None,
              marker_label: Optional[str] = None,
              drill_param: Optional[str] = None,
              note: Optional[str] = None) -> HistogramBlock:
    return HistogramBlock(title=title, source=source, bins=bins, x_label=x_label,
                          marker=marker, marker_label=marker_label,
                          drill_param=drill_param, note=note)


def table(title: str, source: str, columns: list[dict], rows: list[dict], *,
          drill_key: Optional[str] = None, note: Optional[str] = None) -> TableBlock:
    if len(rows) > MAX_TABLE_ROWS:
        note = ((note + " ") if note else "") + \
            f"Showing the first {MAX_TABLE_ROWS} of {len(rows)} rows."
        rows = rows[:MAX_TABLE_ROWS]
    return TableBlock(title=title, source=source, note=note, drill_key=drill_key,
                      columns=[TableColumn(**c) for c in columns], rows=rows)


def stat(title: str, source: str, items: list[dict],
         note: Optional[str] = None) -> StatBlock:
    return StatBlock(title=title, source=source, note=note,
                     items=[StatItem(**i) for i in items])


def flow(title: str, source: str, nodes: list[dict], edges: list[dict],
         layers: list[list[str]], note: Optional[str] = None) -> FlowBlock:
    ids = {n["id"] for n in nodes}
    unknown = [e for e in edges if e["src"] not in ids or e["dst"] not in ids]
    if unknown:
        raise ValueError(f"flow edge references unknown node: {unknown[0]}")
    return FlowBlock(title=title, source=source, note=note, layers=layers,
                     nodes=[FlowNode(**n) for n in nodes],
                     edges=[FlowEdge(**e) for e in edges])


def images(title: str, source: str, sample_ids: list[int], *,
           drill: Optional[str] = None, note: Optional[str] = None) -> ImagesBlock:
    total = len(sample_ids)
    return ImagesBlock(title=title, source=source, note=note, drill=drill,
                       sample_ids=sample_ids[:MAX_IMAGES],
                       total=total if total > MAX_IMAGES else None)
