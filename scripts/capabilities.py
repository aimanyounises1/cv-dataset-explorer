#!/usr/bin/env python3
"""Generate docs/CAPABILITIES.md — the index of everything this system does.

Read from the running application rather than written by hand, because a
hand-written inventory of 32 endpoints, 18 agent tools and 11 QA workflows is
wrong within a week and nobody notices. Everything below comes from the live
OpenAPI schema, the agent registry, the QA flow registry, and the router table
in App.tsx.

    cd backend && .venv/bin/uvicorn app.main:app --port 8000 &
    python scripts/capabilities.py            # writes docs/CAPABILITIES.md
    python scripts/capabilities.py --check    # exit 1 if the file is stale

`--check` is the point of generating it: it turns "the docs are out of date" into
a failing command instead of something a reader discovers.
"""
import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "CAPABILITIES.md"
API = "http://127.0.0.1:8000"

# Endpoint groups, in the order a reader meets them. The prefix match is
# deliberate: a new /api/samples/... route lands in the right section by itself.
GROUPS = [
    ("Browse and inspect", ("/api/samples", "/api/export")),
    ("Search", ("/api/search",)),
    ("Statistics and map", ("/api/stats", "/api/map", "/api/attributes")),
    ("Annotation QA", ("/api/qa/summary", "/api/qa/captions", "/api/qa/consistency",
                       "/api/qa/selection")),
    ("Retrieval benchmark", ("/api/eval",)),
    ("Curation", ("/api/tags", "/api/vlm-tags", "/api/views")),
    ("Assistant", ("/api/chat", "/api/agent", "/api/reports")),
    ("Application self-QA", ("/api/qa/run", "/api/qa/flows", "/api/qa/artifact")),
    ("Operations", ("/api/health", "/api/admin")),
]


def get(path: str):
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.load(r)


def frontend_routes() -> list[tuple[str, str, str]]:
    """Routes from the router, labelled and grouped from the navigation.

    Two files, because they own different facts: `App.tsx` is the authority on
    which paths exist, and `LeftRail.tsx` is the authority on what each is
    called and which job it belongs to. Reading labels from the router would
    silently fall back to component names the moment the nav moved — which is
    exactly what happened when the top bar became a rail.
    """
    src = (ROOT / "frontend" / "src" / "App.tsx").read_text()
    routes = re.findall(r'<Route path="([^"]+)" element=\{<(\w+)', src)

    rail = (ROOT / "frontend" / "src" / "components" / "shell" / "LeftRail.tsx").read_text()
    labels: dict[str, tuple[str, str]] = {}
    group = ""
    for m in re.finditer(r'title:\s*"([^"]+)"|to:\s*"([^"]+)",\s*label:\s*"([^"]+)"', rail):
        if m.group(1):
            group = m.group(1)
        else:
            labels[m.group(2)] = (m.group(3), group)

    out = []
    for path, comp in routes:
        label, grp = labels.get(path, (comp.replace("Page", ""), ""))
        out.append((path, label, grp))
    return out


def render() -> str:
    schema = get("/openapi.json")
    agents = get("/api/agent/graph")
    flows = get("/api/qa/flows")
    health = get("/api/health")

    paths = schema["paths"]
    lines: list[str] = []
    w = lines.append

    w("# Capabilities")
    w("")
    w("Every view, endpoint, agent tool and tested workflow in this system.")
    w("")
    w("**Generated** — do not edit by hand. Regenerate with")
    w("`python scripts/capabilities.py` while the API is running; verify with")
    w("`--check`. It is built from the live OpenAPI schema, the agent registry,")
    w("the QA flow registry and the router in `App.tsx`, so it cannot describe a")
    w("capability the code does not have.")
    w("")
    w(f"- {health['samples']:,} images loaded · semantic search "
      f"**{'on' if health['semantic_search'] else 'off'}**")
    w(f"- {len(paths)} HTTP endpoints · {len(agents['specialists'])} agent "
      f"specialists · {sum(len(s['tools']) for s in agents['specialists'])} agent "
      f"tools · {len(flows['flows'])} tested workflows")
    w("")

    # ---------------------------------------------------------------- views
    w("## Views")
    w("")
    w("| Job | Route | View | What it is for |")
    w("| --- | --- | --- | --- |")
    purpose = {
        "/": "Browse and search; every filter and the paging depth live in the URL.",
        "/samples/:id": "One image: all captions with agreement scores, attributes, "
                        "tags, difficulty axes, exact nearest neighbours.",
        "/map": "UMAP projection of all embeddings. Lasso a region to hand that "
                "exact set to the gallery.",
        "/stats": "Splits, caption lengths, vocabulary, image sizes, zero-shot "
                  "attribute coverage. Bars open their slice.",
        "/quality": "Caption agreement distribution with a review threshold; the "
                    "selection can leave as a gallery filter or an export.",
        "/eval": "The tool measuring its own retrieval accuracy — R@1/5/10 for all "
                 "three search modes.",
        "/compare": "Two images under one synchronized zoom; deterministic "
                    "shared/different panel; draw a region to search or save it.",
        "/chat": "Multi-agent assistant. Answers render as interactive charts, "
                 "tables and reports, not prose about data.",
    }
    for path, label, group in frontend_routes():
        w(f"| {group or '—'} | `{path}` | {label.strip()} | {purpose.get(path, '')} |")
    w("")
    w("Navigation is grouped by job in a persistent left rail; the current "
      "selection has a permanent home in a right rail that appears whenever "
      "something is selected. Plus **⌘K** anywhere: a command palette over "
      "routes, samples, tags, attribute slices, saved views and search.")
    w("")

    # ------------------------------------------------------------ endpoints
    w("## HTTP API")
    w("")
    w(f"Interactive schema at [`{API}/docs`]({API}/docs) while the server runs.")
    w("")
    seen: set[str] = set()
    for title, prefixes in GROUPS:
        rows = []
        for path in sorted(paths):
            if path in seen or not path.startswith(prefixes):
                continue
            seen.add(path)
            for method, op in sorted(paths[path].items()):
                summary = (op.get("summary")
                           or (op.get("description") or "").split("\n")[0]
                           or "")
                rows.append(f"| `{method.upper()} {path}` | {summary.strip()} |")
        if not rows:
            continue
        w(f"### {title}")
        w("")
        w("| Endpoint | |")
        w("| --- | --- |")
        lines.extend(rows)
        w("")
    leftover = [p for p in sorted(paths) if p not in seen]
    if leftover:
        w("### Ungrouped")
        w("")
        w("| Endpoint | |")
        w("| --- | --- |")
        for path in leftover:
            for method in sorted(paths[path]):
                w(f"| `{method.upper()} {path}` | |")
        w("")

    # --------------------------------------------------------------- agents
    w("## Assistant")
    w("")
    w(f"Model `{agents['model']}` via local Ollama. The orchestrator selects up to")
    w(f"{agents['max_parallel_lanes']} specialists per request and runs them in")
    w(f"parallel; each lane is bounded at {agents['lane_timeout_s']:.0f}s.")
    w("")
    for spec in agents["specialists"]:
        cost = " *(expensive — runs alone)*" if spec["cost"] == "expensive" else ""
        w(f"### `{spec['name']}`{cost}")
        w("")
        w(f"{spec['summary'][0].upper()}{spec['summary'][1:]}.")
        w("")
        for tool in spec["tools"]:
            w(f"- `{tool}`")
        w("")
    w("Answers arrive as render blocks — bar, line, pie, histogram, table, stat,")
    w("flow, images, report, qa — each stating the SQL behind it, and each")
    w("clickable through to the gallery slice it describes.")
    w("")

    # ---------------------------------------------------------------- flows
    w("## Tested workflows")
    w("")
    w("Driven through real Chrome by `scripts/ui_smoke.py`, by `POST /api/qa/run`,")
    w("and by the assistant on request — one registry, three consumers.")
    w("")
    w("| Workflow | Budget | What it asserts |")
    w("| --- | --- | --- |")
    for f in flows["flows"]:
        doc = (f.get("doc") or "").strip().split("\n")[0]
        w(f"| {f['name']} | {f['budget_s']:.0f}s | {doc} |")
    w("")

    # --------------------------------------------------------------- optional
    w("## Optional layers and how they degrade")
    w("")
    w("| Layer | Needs | Without it |")
    w("| --- | --- | --- |")
    w("| Semantic search, map, benchmark | `requirements.txt` + `app.ingest` | "
      "Browsing, keyword search and stats still work; the UI says which features "
      "are unavailable. |")
    w("| Caption QA, attributes, difficulty axes | `app.analyze` | Those views "
      "explain the command to run. |")
    w("| Assistant | `requirements-agent.txt` + Ollama | The tab shows exact setup "
      "instructions; nothing else is affected. |")
    w("| Application self-QA | `requirements-qa.txt` (Playwright) | "
      "`POST /api/qa/run` returns 503 with setup instructions. |")
    w("| PowerPoint deck | `python-pptx` | The Markdown report is still produced "
      "and says the deck was skipped. |")
    w("| VLM tag enrichment | Ollama + a vision model | VLM tags are simply absent. |")
    w("")
    w("## Further reading")
    w("")
    w("- `README.md` — setup, feature tour, the two limits worth knowing")
    w("- `docs/TECHNICAL.md` — layer-by-layer build: schema, the real SQL and "
      "query plans, retrieval maths, frontend, measured performance")
    w("- `docs/DESIGN.md` — retrieval design, trade-offs, and the production "
      "scale path")
    w("- `docs/PRISM.md` — the retrieval-accuracy research programme and the "
      "method this project proposes")
    w("- `docs/AGENTS.md` — orchestration, the render-block contract, self-QA")
    w("- `docs/DEMO.md` — an eight-minute walkthrough")
    w("- `docs/screenshots/` — one image per view")
    w("")
    w(f"<!-- generated {datetime.now(timezone.utc):%Y-%m-%d} -->")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed file differs from the live system")
    args = ap.parse_args()

    try:
        body = render()
    except Exception as exc:
        print(f"Could not reach the API at {API}: {exc}\n"
              f"Start it with: cd backend && .venv/bin/uvicorn app.main:app --port 8000",
              file=sys.stderr)
        return 2

    # The date line is regenerated every run; comparing it would make --check
    # fail every day for no reason.
    strip = lambda s: re.sub(r"<!-- generated .*? -->", "", s).strip()  # noqa: E731

    if args.check:
        if not OUT.exists():
            print(f"{OUT.relative_to(ROOT)} does not exist — run without --check.",
                  file=sys.stderr)
            return 1
        if strip(OUT.read_text()) != strip(body):
            print(f"{OUT.relative_to(ROOT)} is out of date — "
                  f"run `python scripts/capabilities.py`.", file=sys.stderr)
            return 1
        print(f"{OUT.relative_to(ROOT)} matches the running system.")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(body.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
