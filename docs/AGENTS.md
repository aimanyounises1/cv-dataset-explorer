# The agent layer: orchestration, canvas, and self-QA

How the assistant is built, why it is built that way, and what it costs. This
document covers the optional agent layer only; `docs/TECHNICAL.md` covers the
platform it sits on and `docs/DESIGN.md` covers the retrieval stack and its
scale path.

Everything here runs on one machine. No hosted model, no managed database, no
external vector store, no paid API. If Ollama is not running, the assistant is
the only thing that stops working.

---

## 1. Architecture

```
                          ┌──────────────────────────────┐
   browser ──── POST ────▶│ /api/chat                    │
                          │  · validates render blocks   │
                          │  · persists reports          │
                          │  · reports lane failures     │
                          └───────────────┬──────────────┘
                                          ▼
                                 ┌────────────────┐
                                 │  orchestrator  │  picks 1–2 lanes
                                 └───────┬────────┘
             ┌──────────────┬────────────┼────────────┬──────────────┐
             ▼              ▼            ▼            ▼              │
       ┌──────────┐  ┌────────────┐ ┌──────────┐ ┌────────┐          │
       │retrieval │  │  insights  │ │   viz    │ │   qa   │          │
       └────┬─────┘  └─────┬──────┘ └────┬─────┘ └───┬────┘          │
            │              │             │           │               │
            └──────────────┴──────┬──────┴───────────┘               │
                    (parallel, results merged)                       │
                                  ▼                                  │
                        ┌───────────────────┐                        │
                        │    synthesizer    │──── RETRY ─────────────┘
                        │   (quality gate)  │      (once)
                        └─────────┬─────────┘
                                  ▼
                   reply + render blocks + lane report
```

Cheap lanes are `retrieval`, `insights` and `visualization`; `qa` is marked
expensive and runs alone. All of them consume the same service functions the REST
API uses — `run_search`, `retrieval_benchmark`, the QA runner — so the assistant
cannot answer differently from the rest of the application.

### Files

| File | Role |
| --- | --- |
| `backend/app/agent/registry.py` | The specialist list. The one place a new agent is declared. |
| `backend/app/agent/graph.py` | Compiles the LangGraph graph *from* the registry. |
| `backend/app/agent/blocks.py` | The render-block contract (Pydantic, discriminated union). |
| `backend/app/agent/viz_tools.py` | Tools that answer with a chart. |
| `backend/app/agent/tools.py` | Tools that search, inspect and tag. |
| `backend/app/agent/qa_tools.py` | Tools that report on the application itself. |
| `backend/app/agent/report_md.py` | Renders a report block to Markdown for download. |
| `backend/app/api/chat.py` | HTTP boundary: block validation, report persistence, lane reporting. |
| `frontend/src/api/blocks.ts` | TypeScript mirror of the block union. |
| `frontend/src/components/blocks/` | One renderer per block kind, plus the registry. |
| `backend/app/qa/` | The autonomous QA sweep: flows, runner, deck. |

---

## 2. Adding things

The extensibility claims are load-bearing, so each is tested rather than
asserted. `backend/tests/test_agent_graph.py::test_registering_a_specialist_needs_no_graph_edit`
registers a throwaway specialist and checks the compiled graph gained a node and
the routing prompt gained a line, with no other change anywhere.

### A new specialist agent

Append one entry to `SPECIALISTS` in `registry.py`:

```python
GEOMETRY = Specialist(
    name="geometry",
    summary="camera geometry questions: focal length, aspect ratio, distortion",
    prompt="You are the geometry specialist. …",
    tools=[estimate_focal_length, aspect_ratio_report],
)
SPECIALISTS = [RETRIEVAL, INSIGHTS, VISUALIZATION, GEOMETRY, QA]
```

The graph node, the routing menu, the fan-out edge and the synthesizer's view of
the lane all follow. Nothing else changes.

Two fields carry weight. `summary` is the *only* description the orchestrator
sees, so it must be phrased as the kind of request that should arrive — and, if
the lane is easy to confuse with another, what should *not*. `cost="expensive"`
excludes a lane from speculative parallel selection.

### A new visualization type

1. Add a Pydantic model and a builder to `backend/app/agent/blocks.py`, and its
   `kind` to `BLOCK_KINDS`.
2. Add the interface and union member to `frontend/src/api/blocks.ts`.
3. Add a renderer to `frontend/src/components/blocks/` and one line to
   `BLOCK_RENDERERS`.

Step 3 is not optional in the "you should remember to" sense: `BLOCK_RENDERERS`
is a mapped type over `BlockKind`, so after step 2 the frontend fails `tsc` until
the renderer exists. A `Record<string, FC>` would have compiled happily and
shipped an empty box.

### A new QA workflow

One decorated function in `backend/app/qa/flows.py`:

```python
@flow("Saved views")
def saved_views(pg, ok):
    pg.goto(f"{BASE}/?split=train", wait_until="domcontentloaded")
    ok("saved view can be stored", pg.query_selector(".saved-views button") is not None)
```

It is picked up by the CLI (`scripts/ui_smoke.py`), by `POST /api/qa/run`, and by
the assistant's status report, because all three read the same registry.

---

## 3. Render blocks: why charts and not prose

A chat agent over a dataset has an obvious failure mode: asked how captions are
distributed, an 8B model writes a confident paragraph of invented numbers. Every
tool that has something to show therefore returns *blocks* rather than text, and
three rules are enforced in `blocks.py` rather than left to each tool.

**Every block names its source.** `source` is a required field. It holds the
measurement — `"COUNT(*) over samples grouped by split"` — and the frontend frame
renders it under every chart, including an explicit "not stated by the tool that
produced this block" when it is somehow missing. A chart without provenance is
indistinguishable from one the model made up.

**Every block is built through a builder.** Builders validate, so a malformed
block fails at the tool call with a traceback rather than silently in the browser
as an empty box. `/api/chat` re-validates against the union at the boundary for
the same reason.

**Series are capped, and capping is disclosed.** `cap_points` keeps the largest
N and folds the remainder into one `other (N levels)` entry whose value is the
summed tail, then says so in the block's note. Silently showing "the top 24 tags"
of 4,000 is a lie about a long-tailed distribution, and the missing mass is
usually the interesting part.

### Interaction: a chart is a way into the data

Every block can carry `drill`, a gallery query string. Clicking a pie slice, a
bar, a table row, a stat card or a flow node navigates to `/?<drill>` — the exact
slice that element counts, with a removable filter chip. That is the same "a view
produces a set" contract the similarity map and the quality page use, extended to
the chat canvas: an answer is not a dead end.

On top of that the renderers provide hover tooltips with exact values, legend
clicks to toggle series, `<Brush>` zoom on histograms and line charts,
click-to-sort and text filtering on tables, and a labelled reference line where a
threshold matters.

### Reports

`build_dataset_report` assembles a fixed, curated set of analyses — scale,
composition, annotation quality, difficulty profile, retrieval accuracy — rather
than letting the model compose sections from block payloads. Report structure is
a product decision, not a per-turn generation problem, and threading block JSON
through a small model's tool arguments is exactly where it is least reliable. The
measured output is 5 sections and 13 visualizations.

`/api/chat` writes each report to `backend/data/reports/<id>.{json,md}` and
attaches download URLs. Markdown renders every block as the data behind it — a
bar chart becomes its table, a flow becomes its edge list — because that is the
honest translation of a chart into text.

---

## 4. Failure, bounded

A local 8B model asked a vague question will loop through tools; a stalled Ollama
will hold a socket open forever; one specialist can fail while another succeeds.
Each of those has a specific answer, and each answer has a test.

| Failure | Response | Test |
| --- | --- | --- |
| A lane raises | Recorded as failed; the turn continues with the other lanes | `test_one_failing_lane_does_not_fail_the_turn` |
| A lane hangs | Cut off at `AGENT_LANE_TIMEOUT`; reported as a timeout | `test_hanging_lane_is_cut_off` |
| Tool loop runs away | `AGENT_RECURSION_LIMIT` inside each lane | — |
| One model call stalls | `OLLAMA_TIMEOUT` on the HTTP client | — |
| A complete turn stalls | One absolute `AGENT_TURN_BUDGET` deadline covers preflight, graph, stream and response assembly | `test_blocking_response_budget_includes_ollama_preflight`, `test_streaming_response_uses_the_same_absolute_deadline` |
| The orchestrator fails | Falls back to `retrieval` (read-only, cheap) | `test_orchestrator_failure_falls_back_to_retrieval` |
| The synthesizer fails | Refuses router-identified premises; otherwise hands over the specialist answer marked unverified | `test_synthesizer_failure_still_answers`, `test_synthesizer_timeout_still_answers` |
| User supplies an unsupported premise | Structured verdict must cite an exact current-turn tool excerpt; otherwise the final answer is a refusal | `test_unverified_claim_cannot_escape_through_adversarial_synthesis`, `test_supported_claim_requires_an_exact_current_tool_excerpt` |
| Orchestrator emits junk | Prose-wrapped JSON, `route` for `routes`, unknown names, and over-long lane lists all handled | `test_parse_routes` (9 cases) |
| A lane died | Named in the synthesizer's prompt *and* in the UI above the answer | `test_synthesizer_is_told_which_lanes_failed` |

Partial failure is surfaced, not smoothed over. The alternative is a reply that
covered half the request and reads as though it covered all of it.

### Concurrency

Two findings from building this, both fixed:

**Concurrent inference on one MPS model crashes the process.** Two threads
calling SigLIP's text encoder on the same module either segfault inside
`copy_cast_kernel_mps` or deadlock inside Metal — both reproduced here. FastAPI
serves sync endpoints from a thread pool, so *two simultaneous semantic searches*
were always enough to hit it; parallel agent lanes made it routine. Inference is
now serialized on a per-instance lock, held per batch so a long image run does not
block a single query. `backend/tests/test_embedder_concurrency.py` asserts no two
forward passes overlap.

**Timeouts are LangGraph nodes, not thread wrappers.** Every lane is async and
uses `StateGraph.add_node(timeout=..., error_handler=...)`. Each lane is an
isolated subgraph so one branch can record its `NodeTimeoutError` without
failing sibling branches. The API wraps the whole `ainvoke`/`astream` call in
one turn timeout, while Ollama's `num_predict` caps server-side generation.

### Large results and duplicates

Charts cap at 24 categories (10 for pies) and disclose it; tables cap at 200 rows
and disclose it; image blocks cap at 24 and report the true total. Blocks that are
byte-identical within one turn collapse to one — a local model will call the same
tool twice, and two parallel lanes can independently chart the same thing. Both
were observed; both looked like a rendering bug.

---

## 5. The autonomous QA agent

Ask *"show me the status of the application"* and the `qa` lane drives real Chrome
over every workflow, screenshots each one, and returns a pass/fail report inline
with a downloadable deck.

The important design property is that there is **one definition of the flows**.
`backend/app/qa/flows.py` holds them as decorated callables; the CLI smoke test,
the HTTP endpoint and the assistant all execute that same registry. A separate
in-app QA suite would have drifted from the developer one within a week.

- `POST /api/qa/run` starts a sweep in a background thread and returns a run id
  immediately. Only one sweep runs at a time — two Chromes driving the same dev
  server interleave their navigations and both report nonsense — so a second
  request attaches to the in-flight run. A run whose heartbeat goes stale is
  abandoned, because a browser can hang in a way no internal timeout can see.
- `GET /api/qa/run/{id}` polls; `GET /api/qa/run` returns the latest.
- A sweep takes minutes, which a chat turn cannot. `run_app_qa` starts the
  background runner and immediately returns the run id; `app_qa_status` reads
  progress without keeping an agent lane occupied.
- Playwright and `python-pptx` are optional (`backend/requirements-qa.txt`). No
  Playwright gives a 503 with setup instructions; no `python-pptx` still produces
  the Markdown report and says the deck was skipped and how to enable it.

Artifacts land under `backend/data/qa/<run_id>/` (gitignored), served read-only at
`/media/qa/` and also at `/api/qa/artifact/{run_id}/{name}`.

---

## 6. Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `CVDE_CHAT_MODEL` | `qwen3:8b` | Ollama model. Must support tool calling. |
| `CVDE_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint. |
| `CVDE_OLLAMA_TIMEOUT` | `120` | One model call, seconds. |
| `CVDE_AGENT_LANE_TIMEOUT` | `90` | One async LangGraph specialist node, seconds. |
| `CVDE_AGENT_TURN_BUDGET` | `150` | Complete assistant turn, seconds. |
| `CVDE_AGENT_RECURSION_LIMIT` | `25` | Tool-call steps inside a lane. |
| `CVDE_QA_BASE_URL` | `http://localhost:5173` | What the QA browser drives. |

`CVDE_QA_BASE_URL` uses `localhost`, not `127.0.0.1`: the Vite dev server listens
on IPv6 only, so the numeric form connects to nothing.

---

## 7. Known limits

- **Answer quality is bounded by an 8B local model.** Routing, tool choice and
  prose are all as good as `qwen3:8b`, which is to say usually right and
  occasionally not. The mitigations are structural rather than prompt-based:
  tools return the numbers so the model never derives them (it reported "train
  60%" under a chart correctly showing 75% until the shares were precomputed and
  handed to it), the routing menu states exclusions, and every chart carries its
  own provenance so a reader can check the prose against it.
- **Fan-out is capped at two lanes.** Three would mean three more full model
  round-trips against one local GPU for a request that is usually one badly
  phrased question.
- **Ollama serializes at the model level.** Two lanes overlap in tool work and in
  SQL, and their generation steps queue. Parallelism is real (measured: lane
  execution windows overlap) but the speedup is smaller than the lane count.
- **The steps stream; the text does not.** `POST /api/chat/stream` emits
  LangGraph's own node transitions as NDJSON — `{"type":"step","node":…,"t":…}`
  per real transition, never a staged animation — and the Assistant renders
  them live, so a 50-second report names the lane it is in rather than showing
  a bare spinner. The reply itself arrives whole, in the closing
  `{"type":"final"}` line carrying the exact `POST /chat` payload. Streaming
  tokens, and streaming *blocks* as lanes finish, are still not built.
- **Reports do not survive a cleared data directory.** They are build artifacts.
- **`qwen2.5vl:7b` is not installed here,** so VLM tag enrichment is unavailable
  and the app degrades as designed.
