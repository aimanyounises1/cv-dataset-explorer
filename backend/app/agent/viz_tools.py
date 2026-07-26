"""Tools that answer with a chart instead of a sentence.

Every tool here returns JSON carrying a `blocks` list built through
`agent.blocks`, so the numbers are measured in SQL and the chart states where
they came from. The agent's job is to choose the dimension and explain the
result; it never supplies the values, which is what stops a 7-billion-parameter
local model from drawing a plausible chart out of nothing.

Two deliberate choices about tool *count*:

* One `plot_distribution(dimension)` rather than a tool per chart. A small local
  model picks badly from a long menu, and every dimension here answers the same
  question shape ("how is X distributed, and which slice do I click to see it").
* `build_dataset_report` assembles a fixed, curated set of analyses rather than
  letting the model compose sections from block JSON. Report structure is a
  product decision, not a per-turn generation problem, and threading block
  payloads through a local model's tool arguments is exactly where that model is
  least reliable.
"""
import json
import logging

from langchain_core.tools import tool

from .. import config, db
from . import blocks

logger = logging.getLogger(__name__)

# Dimensions `plot_distribution` understands, with the attribute groups appended
# at call time so a newly analyzed group needs no code change here.
CORE_DIMENSIONS = ("split", "caption_length", "agreement", "difficulty_axes",
                   "image_size", "tags")

_AXIS_LABEL = {"legibility": "Legibility", "rarity": "Rarity",
               "difficulty": "Difficulty", "clutter": "Clutter"}


def _attribute_groups(conn) -> list[str]:
    return [r["grp"] for r in conn.execute(
        "SELECT DISTINCT grp FROM attributes ORDER BY grp")]


def _dimension_help(conn) -> str:
    return ", ".join(list(CORE_DIMENSIONS) + _attribute_groups(conn))


def _figures(block) -> list[str]:
    """The block's own values, pre-formatted with shares.

    An 8B model asked to summarize a chart will do the arithmetic itself and get
    it wrong: on the split pie it reported "train 60%" under a chart correctly
    showing 75%. Handing it the computed shares removes the incentive to
    calculate, which is the only reliable fix — a prompt telling a small model
    not to do mental arithmetic does not stop it.
    """
    points = []
    if getattr(block, "kind", "") == "pie":
        points = block.points
    elif getattr(block, "kind", "") == "bar" and len(block.series) == 1:
        points = block.series[0].points
    if not points:
        return []
    total = sum(p.value for p in points) or 1.0
    return [f"{p.label}: {p.value:,.0f} ({p.value / total:.1%})" for p in points]


def _payload(*block_objs, **extra) -> str:
    """Serialize blocks plus any plain fields the model should read as text.

    The model sees `summary` and `figures`; the UI renders `blocks`. Keeping them
    in one payload means the prose and the picture cannot describe different
    numbers — and `figures` exists so the prose does not have to derive any.
    """
    payload = {"blocks": [b.model_dump(mode="json") for b in block_objs], **extra}
    # A caller may supply its own figures — `compare_slices` names superlatives a
    # table cannot express — and those win. Only derive them when it did not.
    if "figures" not in payload:
        figures: list[str] = []
        for b in block_objs:
            figures += _figures(b)
        if figures:
            payload["figures"] = figures
            payload["note_to_agent"] = (
                "Quote these figures verbatim if you cite any number; do not "
                "compute percentages yourself.")
    return json.dumps(payload)


# --------------------------------------------------------------- distributions

def _split_block(conn):
    rows = conn.execute(
        "SELECT split, COUNT(*) AS n FROM samples GROUP BY split ORDER BY n DESC")
    points = [{"label": r["split"], "value": r["n"], "drill": f"split={r['split']}"}
              for r in rows]
    return blocks.pie(
        "Samples per split",
        "COUNT(*) over samples grouped by split",
        points,
        note="Click a slice to open that split in the gallery."), points


def _caption_length_block(conn):
    # Word counts are computed once at ingest; binning in SQL avoids streaming
    # 40,000 rows into Python to produce 20 numbers.
    rows = conn.execute(
        "SELECT n_words AS w, COUNT(*) AS n FROM ("
        "  SELECT LENGTH(TRIM(text)) - LENGTH(REPLACE(TRIM(text), ' ', '')) + 1 AS n_words"
        "  FROM captions) GROUP BY w ORDER BY w").fetchall()
    bins = [{"lo": r["w"], "hi": r["w"] + 1, "count": r["n"]} for r in rows if r["w"] <= 40]
    total = sum(b["count"] for b in bins) or 1
    mean = sum(b["lo"] * b["count"] for b in bins) / total
    return blocks.histogram(
        "Caption length",
        "Word count per caption, counted in SQL over all captions",
        bins, x_label="words in caption", marker=round(mean, 1),
        marker_label=f"mean {mean:.1f}",
        note="Short captions carry less retrievable signal; the left tail is where "
             "keyword search has least to rank on.")


def _agreement_block(conn):
    from ..api.qa import qa_summary

    summary = qa_summary(conn)
    if not summary.available:
        return None
    lo = summary.min_agreement or 0.0
    hi = summary.max_agreement or 1.0
    # A ~1% tail is the same default the Quality page uses, so the marker on
    # this chart and the slider over there mean the same thing.
    total = sum(b["count"] for b in summary.histogram) or 1
    seen, cut = 0, hi
    for b in summary.histogram:
        seen += b["count"]
        if seen / total >= 0.01:
            cut = b["hi"]
            break
    return blocks.histogram(
        "Image–caption agreement",
        f"SigLIP cosine per caption, {summary.scored_captions:,} captions, binned in SQL",
        summary.histogram, x_label="agreement (cosine)",
        marker=cut, marker_label=f"review below {cut:.3f}",
        drill_param="max_agreement",
        note=f"Observed range {lo:.3f}–{hi:.3f}: agreement occupies a narrow slice of "
             f"[0,1], so bins are over the observed range, not the full interval. "
             f"Click a bin to open every image with a caption at or below it.")


def _axes_block(conn):
    row = conn.execute(
        "SELECT " + ", ".join(f"AVG({a}) AS {a}" for a in db.AXES) + " FROM samples"
    ).fetchone()
    if row is None or all(row[a] is None for a in db.AXES):
        return None
    # Mean of a percentile rank is ~5 by construction, so the mean alone says
    # nothing. The share scoring 8+ is the number a researcher actually wants.
    hard = {a: conn.execute(
        f"SELECT COUNT(*) FROM samples WHERE {a} >= 8").fetchone()[0] for a in db.AXES}
    return blocks.bar(
        "Hard tail per difficulty axis",
        "COUNT(*) where axis >= 8, over samples (axes are dataset-relative "
        "percentile ranks 0–10)",
        [{"label": _AXIS_LABEL[a], "value": hard[a], "drill": f"{a}_min=8"}
         for a in db.AXES],
        series_name="samples scoring 8+", y_label="samples",
        note="Percentile ranks average ~5 by construction, so the mean is not "
             "informative; this counts the hard tail instead. Click a bar to open "
             "that slice.")


def _image_size_block(conn):
    rows = conn.execute(
        "SELECT width || '×' || height AS wh, COUNT(*) AS n FROM samples "
        "GROUP BY wh ORDER BY n DESC").fetchall()
    return blocks.bar(
        "Image dimensions",
        "COUNT(*) over samples grouped by width×height",
        [{"label": r["wh"], "value": r["n"]} for r in rows],
        series_name="images", horizontal=True, y_label="images",
        note="Flickr8k is not dimension-normalized; a model trained on it sees a "
             "mixture of aspect ratios.")


def _tags_block(conn):
    rows = conn.execute(
        "SELECT t.name, COUNT(*) AS n FROM tags t "
        "JOIN sample_tags st ON st.tag_id = t.id GROUP BY t.id ORDER BY n DESC"
    ).fetchall()
    if not rows:
        return None
    return blocks.bar(
        "Curation tags",
        "COUNT(*) over sample_tags grouped by tag",
        [{"label": r["name"], "value": r["n"], "drill": f"tag={r['name']}"}
         for r in rows],
        series_name="samples", horizontal=True,
        note="Tags applied in this tool — by hand, by a map lasso, or by the "
             "assistant. Click a bar to open the tagged slice.")


def _attribute_block(conn, grp: str):
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] or 1
    rows = conn.execute(
        "SELECT label, COUNT(*) AS n FROM attributes WHERE grp = ? "
        "GROUP BY label ORDER BY n DESC", (grp,)).fetchall()
    if not rows:
        return None
    scored = sum(r["n"] for r in rows)
    points = [{"label": r["label"], "value": r["n"],
               "drill": f"attr={grp}:{r['label']}"} for r in rows]
    abstained = total - scored
    note = ("Click a bar to open that slice. Zero-shot SigLIP labels with a "
            "top1−top2 margin gate: ")
    note += (f"{abstained:,} of {total:,} samples ({abstained / total:.1%}) abstained "
             f"as too close to call." if abstained > 0
             else "every sample cleared the margin gate.")
    return blocks.bar(
        f"Attribute coverage — {grp.replace('_', ' ')}",
        f"COUNT(*) over attributes where grp='{grp}', margin-gated zero-shot labels",
        points, series_name="samples", horizontal=True, y_label="samples", note=note)


@tool
def plot_distribution(dimension: str) -> str:
    """Draw how the dataset is distributed along one dimension, as an interactive
    chart the user can hover and click through to the matching images.

    dimension must be one of: 'split', 'caption_length', 'agreement',
    'difficulty_axes', 'image_size', 'tags', or an attribute group name such as
    'time_of_day', 'setting', 'environment', 'main_subject'. Call this instead of
    describing a distribution in words."""
    conn = db.connect()
    try:
        dim = (dimension or "").strip().lower().replace(" ", "_")
        aliases = {"splits": "split", "captions": "caption_length",
                   "caption_lengths": "caption_length", "quality": "agreement",
                   "caption_quality": "agreement", "axes": "difficulty_axes",
                   "difficulty": "difficulty_axes", "size": "image_size",
                   "dimensions": "image_size", "tag": "tags",
                   "time": "time_of_day", "subject": "main_subject"}
        dim = aliases.get(dim, dim)

        block = None
        if dim == "split":
            block, _ = _split_block(conn)
        elif dim == "caption_length":
            block = _caption_length_block(conn)
        elif dim == "agreement":
            block = _agreement_block(conn)
        elif dim == "difficulty_axes":
            block = _axes_block(conn)
        elif dim == "image_size":
            block = _image_size_block(conn)
        elif dim == "tags":
            block = _tags_block(conn)
        elif dim in _attribute_groups(conn):
            block = _attribute_block(conn, dim)
        else:
            return json.dumps({
                "error": f"unknown dimension '{dimension}'",
                "available": _dimension_help(conn)})

        if block is None:
            return json.dumps({
                "error": f"'{dim}' has no computed values yet — run "
                         f"`python -m app.analyze` in the backend.",
                "available": _dimension_help(conn)})
        return _payload(block, summary=f"Rendered a chart of {dim}: {block.source}.")
    finally:
        conn.close()


@tool
def plot_retrieval_benchmark() -> str:
    """Chart measured retrieval accuracy — Recall@1/5/10 for semantic, keyword and
    hybrid search — from the caption-to-image benchmark. Use when asked how well
    search works, or to compare the three search modes. Runs a real evaluation the
    first time and is cached afterwards, so it may take a while."""
    conn = db.connect()
    try:
        from ..api.eval import retrieval_benchmark

        res = retrieval_benchmark(sample_size=1000, conn=conn)
        if not res.available:
            return json.dumps({"error": res.message or "benchmark unavailable"})
        series = [
            {"name": f"R@{k}",
             "points": [{"label": m.mode,
                         "value": round(m.recall_at.get(str(k), 0.0) * 100, 2)}
                        for m in res.results]}
            for k in (1, 5, 10)
        ]
        # The pool a mode actually ranked against. Reporting recall without it
        # invites reading 2% keyword recall as a ranking failure, when the real
        # finding is that the lexical index returned nothing to rank.
        rows = [{"mode": m.mode,
                 "R@1": f"{m.recall_at.get('1', 0.0):.1%}",
                 "R@5": f"{m.recall_at.get('5', 0.0):.1%}",
                 "R@10": f"{m.recall_at.get('10', 0.0):.1%}",
                 "MRR@10": f"{m.mrr:.3f}",
                 "median rank": (f"{m.median_rank:.0f}" if m.median_rank is not None
                                 else f"> {res.depth}"),
                 "mean candidates": f"{m.mean_candidates:,.1f}",
                 "empty queries": f"{m.empty_query_rate:.1%}"}
                for m in res.results]
        chart = blocks.grouped_bar(
            "Retrieval accuracy by mode",
            f"Caption→image retrieval over {res.sample_size:,} held-out captions "
            f"against a pool of {res.pool_size:,} images; each query caption is "
            f"excluded from its own candidate pool",
            series, y_label="recall (%)",
            note="Keyword mode is not ranking against the whole corpus: the lexical "
                 "index requires every caption term in one caption, so most queries "
                 "retrieve almost nothing. Read its recall next to the mean "
                 "candidates column, which is the honest denominator.")
        detail = blocks.table(
            "Benchmark detail", chart.source,
            [{"key": "mode", "label": "mode"},
             {"key": "R@1", "label": "R@1", "numeric": True},
             {"key": "R@5", "label": "R@5", "numeric": True},
             {"key": "R@10", "label": "R@10", "numeric": True},
             {"key": "MRR@10", "label": "MRR@10", "numeric": True},
             {"key": "median rank", "label": "median rank", "numeric": True},
             {"key": "mean candidates", "label": "mean candidates", "numeric": True},
             {"key": "empty queries", "label": "empty queries", "numeric": True}],
            rows,
            note=f"Queries are whole captions ({res.mean_query_words:.1f} words on "
                 f"average), ranked to depth {res.depth}.")
        return _payload(chart, detail,
                        summary=f"Benchmarked {res.sample_size} captions across "
                                f"{len(res.results)} modes.")
    finally:
        conn.close()


@tool
def show_images(sample_ids: list[int], caption: str = "Selected images") -> str:
    """Display specific images inline as an interactive strip. Pass the sample ids
    you want shown. Use after a search or an analysis to put the actual pictures
    in front of the user."""
    ids = [int(s) for s in (sample_ids or [])][:blocks.MAX_IMAGES]
    if not ids:
        return json.dumps({"error": "no sample ids given"})
    conn = db.connect()
    try:
        qmarks = ",".join("?" * len(ids))
        found = {r["id"] for r in conn.execute(
            f"SELECT id FROM samples WHERE id IN ({qmarks})", ids)}
        keep = [i for i in ids if i in found]
        if not keep:
            return json.dumps({"error": "none of those sample ids exist"})
        missing = [i for i in ids if i not in found]
        note = f"{len(missing)} of the requested ids are not in this dataset." if missing else None
        block = blocks.images(caption, "Sample ids resolved against the samples table",
                              keep, drill=f"ids={','.join(str(i) for i in keep)}",
                              note=note)
        return _payload(block, summary=f"Showing {len(keep)} images.",
                        sample_ids=keep)
    finally:
        conn.close()


@tool
def compare_slices(dimension: str) -> str:
    """Compare the slices of one attribute group side by side in a sortable table:
    how many samples each holds, and how they differ on caption quality and the
    difficulty axes. Use to find which slice of the dataset is hardest, not just
    which is rarest. dimension is an attribute group such as 'time_of_day'."""
    conn = db.connect()
    try:
        grp = (dimension or "").strip().lower().replace(" ", "_")
        aliases = {"time": "time_of_day", "subject": "main_subject"}
        grp = aliases.get(grp, grp)
        groups = _attribute_groups(conn)
        if grp not in groups:
            return json.dumps({"error": f"'{dimension}' is not an attribute group",
                               "available": ", ".join(groups)})
        total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] or 1
        rows = conn.execute(
            "SELECT a.label AS label, COUNT(*) AS n, "
            "  AVG(s.difficulty) AS difficulty, AVG(s.rarity) AS rarity, "
            "  AVG(s.clutter) AS clutter, AVG(s.legibility) AS legibility, "
            "  AVG(s.caption_consistency) AS consistency "
            "FROM attributes a JOIN samples s ON s.id = a.sample_id "
            "WHERE a.grp = ? GROUP BY a.label ORDER BY n DESC", (grp,)).fetchall()
        if not rows:
            return json.dumps({"error": f"no samples labelled in group '{grp}'"})
        out = []
        for r in rows:
            out.append({
                "slice": r["label"],
                "samples": r["n"],
                "share": f"{r['n'] / total:.1%}",
                "difficulty": round(r["difficulty"], 2) if r["difficulty"] is not None else None,
                "rarity": round(r["rarity"], 2) if r["rarity"] is not None else None,
                "clutter": round(r["clutter"], 2) if r["clutter"] is not None else None,
                "legibility": round(r["legibility"], 2) if r["legibility"] is not None else None,
                # `caption_consistency` is agreement BETWEEN a sample's five
                # captions, not between the image and its captions. The two live
                # on different scales here — 0.77 against 0.16 — so labelling
                # this "caption agreement" invited a reader to compare it with
                # the number on the Quality page and conclude the opposite of
                # the truth. Named for what it is.
                "caption consistency": round(r["consistency"], 3) if r["consistency"] is not None else None,
                "drill": f"attr={grp}:{r['label']}",
            })
        # The superlatives, stated rather than left to be inferred. Asked which
        # slice was rarest, the model read the table and answered "night (4.9%)"
        # when dusk sits at 2.7% two rows below — the same failure as deriving a
        # percentage, one level up: an ordering rather than an arithmetic. Naming
        # each extreme is the fix, because it removes the inference entirely.
        scored = [r for r in out if r["difficulty"] is not None]
        superlatives = [
            f"largest slice: {max(out, key=lambda r: r['samples'])['slice']}",
            f"rarest slice: {min(out, key=lambda r: r['samples'])['slice']}",
        ]
        if scored:
            superlatives += [
                f"hardest by difficulty: {max(scored, key=lambda r: r['difficulty'])['slice']}",
                f"hardest to see (legibility): "
                f"{max(scored, key=lambda r: r['legibility'] or 0)['slice']}",
            ]
        block = blocks.table(
            f"Slices of {grp.replace('_', ' ')}, compared",
            f"Per-label AVG over the difficulty axes and caption consistency, "
            f"grouped from attributes where grp='{grp}'",
            [{"key": "slice", "label": "slice"},
             {"key": "samples", "label": "samples", "numeric": True},
             {"key": "share", "label": "share", "numeric": True},
             {"key": "difficulty", "label": "difficulty", "numeric": True},
             {"key": "rarity", "label": "rarity", "numeric": True},
             {"key": "clutter", "label": "clutter", "numeric": True},
             {"key": "legibility", "label": "legibility", "numeric": True},
             {"key": "caption consistency", "label": "caption consistency",
              "numeric": True}],
            out, drill_key="drill",
            note="Axis columns are dataset-relative percentile ranks (0–10), so a "
                 "value above 5 means this slice is harder than the dataset median "
                 "on that axis. Click a row to open the slice.")
        return _payload(block,
                        summary=f"Compared {len(out)} slices of {grp}.",
                        figures=superlatives,
                        note_to_agent="Quote these superlatives verbatim; do not "
                                      "work out which slice is largest, rarest or "
                                      "hardest by reading the table yourself.")
    finally:
        conn.close()


@tool
def system_diagram() -> str:
    """Draw how this platform is built: the request path from the UI through the
    agents, tools, retrieval stack and storage. Use when the user asks how the
    system works, what the architecture is, or which components exist."""
    from ..ml.index import get_caption_index, get_index

    semantic = get_index() is not None
    nodes = [
        {"id": "ui", "label": "React UI\ngallery · map · chat", "group": "ui"},
        {"id": "api", "label": "FastAPI\nREST + agent", "group": "api"},
        {"id": "orch", "label": "Orchestrator", "group": "agent"},
        {"id": "retrieval", "label": "Retrieval\nagent", "group": "agent"},
        {"id": "insights", "label": "Insights\nagent", "group": "agent"},
        {"id": "viz", "label": "Visualization\nagent", "group": "agent"},
        {"id": "qa", "label": "QA agent", "group": "agent"},
        {"id": "synth", "label": "Synthesizer\nquality gate", "group": "agent"},
        {"id": "search", "label": "Hybrid search\nBM25 + SigLIP, RRF", "group": "engine"},
        {"id": "embed", "label": f"SigLIP 2\n{'loaded' if semantic else 'not computed'}",
         "group": "engine"},
        {"id": "sqlite", "label": "SQLite + FTS5\n8k images · 40k captions", "group": "store"},
        {"id": "npy", "label": "Embedding matrix\n.npy, exact cosine", "group": "store"},
    ]
    edges = [
        {"src": "ui", "dst": "api"},
        {"src": "api", "dst": "orch"},
        {"src": "orch", "dst": "retrieval", "label": "parallel"},
        {"src": "orch", "dst": "insights", "label": "parallel"},
        {"src": "orch", "dst": "viz", "label": "parallel"},
        {"src": "orch", "dst": "qa"},
        {"src": "retrieval", "dst": "synth"},
        {"src": "insights", "dst": "synth"},
        {"src": "viz", "dst": "synth"},
        {"src": "qa", "dst": "synth"},
        {"src": "retrieval", "dst": "search"},
        {"src": "insights", "dst": "sqlite"},
        {"src": "viz", "dst": "sqlite"},
        {"src": "search", "dst": "embed"},
        {"src": "search", "dst": "sqlite"},
        {"src": "embed", "dst": "npy"},
    ]
    layers = [["ui"], ["api"], ["orch"],
              ["retrieval", "insights", "viz", "qa"],
              ["synth", "search"], ["embed", "sqlite"], ["npy"]]
    block = blocks.flow(
        "How this platform is wired",
        "Read from the running application: agent registry, tool registry, and "
        "whether the embedding index is loaded",
        nodes, edges, layers,
        note="Every component runs on this machine: SQLite for data and full-text "
             "search, a NumPy matrix for exact nearest-neighbour search, and a "
             "local Ollama model for the agents. No hosted service is involved.")
    return _payload(block, summary="Rendered the architecture diagram.",
                    semantic_search=semantic,
                    caption_scores=get_caption_index() is not None)


# --------------------------------------------------------------------- reports

def _overview_stats(conn):
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    caps = conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0]
    scored = conn.execute(
        "SELECT COUNT(*) FROM captions WHERE agreement IS NOT NULL").fetchone()[0]
    tagged = conn.execute(
        "SELECT COUNT(DISTINCT sample_id) FROM sample_tags").fetchone()[0]
    from ..ml.index import get_index

    return blocks.stat(
        "Dataset at a glance",
        "COUNT(*) over samples, captions, sample_tags",
        [{"label": "images", "value": f"{total:,}"},
         {"label": "captions", "value": f"{caps:,}",
          "hint": f"{caps / max(1, total):.1f} per image"},
         {"label": "QA-scored captions", "value": f"{scored:,}",
          "hint": "image–caption agreement computed with SigLIP"},
         {"label": "curated (tagged)", "value": f"{tagged:,}",
          "hint": "images carrying at least one curation tag"},
         {"label": "semantic search", "value": "on" if get_index() else "off",
          "hint": f"embedding model {config.EMBED_MODEL}"}])


def _suspect_table(conn, limit: int = 10):
    rows = conn.execute(
        "SELECT c.sample_id, c.text, c.agreement FROM captions c "
        "WHERE c.agreement IS NOT NULL ORDER BY c.agreement ASC LIMIT ?",
        (limit,)).fetchall()
    if not rows:
        return None
    return blocks.table(
        "Least supported captions",
        "captions ordered by ascending SigLIP image–caption agreement",
        [{"key": "sample", "label": "sample", "numeric": True},
         {"key": "agreement", "label": "agreement", "numeric": True},
         {"key": "caption", "label": "caption"}],
        [{"sample": r["sample_id"], "agreement": round(r["agreement"], 4),
          "caption": r["text"], "drill": f"ids={r['sample_id']}"} for r in rows],
        drill_key="drill",
        note="Low agreement is a candidate annotation error, not a verdict: an "
             "unusual image scores low with a perfectly good caption. Click a row "
             "to judge it against the picture.")


REPORT_SECTIONS = ("overview", "composition", "captions", "difficulty", "retrieval")


@tool
def build_dataset_report(title: str = "Flickr8k dataset report",
                         include_benchmark: bool = True) -> str:
    """Generate a full multi-section dataset report, rendered inline and
    downloadable as Markdown or JSON. Covers scale, composition, caption quality,
    the difficulty axes and — unless include_benchmark is false — measured
    retrieval accuracy. Use when the user asks for a report, a summary, an audit,
    or an overview of the dataset. Set include_benchmark=false for a fast report:
    the benchmark can take minutes on a cold cache."""
    conn = db.connect()
    try:
        sections: list[dict] = []

        sections.append({
            "heading": "Scale and readiness",
            "text": "What is in the dataset and which derived signals have been "
                    "computed. Anything reported as off simply has not been run; "
                    "no number below is estimated.",
            "blocks": [_overview_stats(conn)]})

        composition = []
        split_block, _ = _split_block(conn)
        composition.append(split_block)
        for grp in _attribute_groups(conn):
            b = _attribute_block(conn, grp)
            if b is not None:
                composition.append(b)
        size = _image_size_block(conn)
        if size is not None:
            composition.append(size)
        sections.append({
            "heading": "Composition",
            "text": "How the corpus divides by split and by zero-shot attribute. "
                    "The long tail is the part that matters for a CV model: a "
                    "slice holding under a percent of the data is a slice the "
                    "model will rarely see in training.",
            "blocks": composition})

        caption_blocks = [_caption_length_block(conn)]
        agreement = _agreement_block(conn)
        if agreement is not None:
            caption_blocks.append(agreement)
        suspects = _suspect_table(conn)
        if suspects is not None:
            caption_blocks.append(suspects)
        sections.append({
            "heading": "Annotation quality",
            "text": "Caption length bounds how much a caption can specify; "
                    "image–caption agreement bounds how much of it is true of the "
                    "picture. Together they locate the annotation errors.",
            "blocks": caption_blocks})

        axes = _axes_block(conn)
        if axes is not None:
            sections.append({
                "heading": "Difficulty profile",
                "text": "Four dataset-relative axes, each a percentile rank in "
                        "0–10. The hard tail is what a benchmark slice should be "
                        "drawn from.",
                "blocks": [axes]})

        if include_benchmark:
            try:
                bench = json.loads(plot_retrieval_benchmark.invoke({}))
                if "blocks" in bench:
                    sections.append({
                        "heading": "Retrieval accuracy",
                        "text": "Measured caption-to-image retrieval, with each "
                                "query caption excluded from its own candidate "
                                "pool. Without that exclusion keyword search "
                                "scores 99% by finding the caption it was given.",
                        "blocks": bench["blocks"]})
            except Exception as exc:                      # pragma: no cover
                logger.warning("Report: benchmark section skipped (%s)", exc)
                sections.append({
                    "heading": "Retrieval accuracy",
                    "text": f"Skipped — the benchmark could not run ({exc}). "
                            f"Everything above is unaffected.",
                    "blocks": []})

        report = blocks.ReportBlock(
            title=title,
            source="Assembled from live SQL over the explorer database; every "
                   "section states its own measurement",
            sections=[blocks.ReportSection(**s) for s in sections])
        n_blocks = sum(len(s["blocks"]) for s in sections)
        return _payload(report,
                        summary=f"Built '{title}' with {len(sections)} sections and "
                                f"{n_blocks} visualizations.")
    finally:
        conn.close()


VIZ_TOOLS = [plot_distribution, plot_retrieval_benchmark, compare_slices,
             show_images, system_diagram, build_dataset_report]
