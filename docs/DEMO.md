# Demo script

Eight minutes, three acts: a report rendered inline, an interactive chart you can
click through into the data, and the QA agent producing a screenshot-based status
deck. Every number below was observed on this machine; where a step depends on
something optional, the fallback is stated.

## Before you start

```bash
# 1. backend  (SigLIP loads lazily on first semantic query)
cd backend && .venv/bin/uvicorn app.main:app --port 8000

# 2. frontend
cd frontend && npm run dev            # http://localhost:5173

# 3. the assistant needs a local model with tool calling
ollama serve
ollama pull qwen3:8b

# 4. optional: the in-app QA sweep needs Playwright in the server's venv
cd backend && uv pip install --python .venv/bin/python -r requirements-qa.txt
```

Only step 1 and 2 are required. Without Ollama the Assistant tab explains what to
install and every other view works. Without Playwright the QA agent returns setup
instructions instead of a sweep.

Sanity check:

```bash
curl -s localhost:8000/api/health
# {"status":"ok","samples":8000,"semantic_search":true}
```

---

## Act 1 — a report, generated and rendered inline (~2 min)

Open **Assistant** and send:

> Generate a dataset report

What to point at, in order:

1. **The trace chips** above the answer name the specialist and the tool for
   every step: `insights → dataset_overview`, `visualization → build_dataset_report`.
   Nothing about the answer is unattributed.
2. **The lane footer** under the answer reads something like
   `insights ‖ visualization · 52.9s`. Two specialists ran, in parallel, for this
   one request.
3. **The report itself** renders as collapsible sections, each holding live
   components — 5 sections and 13 visualizations: scale, composition (split pie
   plus one bar per attribute group), annotation quality (caption length, an
   agreement histogram, the least-supported captions), difficulty profile, and
   measured retrieval accuracy.
4. **Under every chart is a `source` line.** "COUNT(*) over samples grouped by
   split". The model chose the dimension; SQL produced the numbers. That line is
   a required field on the block, not a convention.
5. **The `markdown` and `json` buttons** download the report. The Markdown
   renders each chart as the table behind it, so the artifact keeps the data
   rather than a picture of it.

Expected timing: 30–60 s on a cold benchmark cache, then a few seconds.

## Act 2 — an interactive chart that is a way into the data (~3 min)

Send:

> Which time of day is hardest? Compare the slices

1. The footer again shows two lanes — `insights ‖ visualization`.
2. You get a **sortable table** of every `time_of_day` slice with its sample
   count, share, and mean value on all four difficulty axes. **Click a column
   header** to sort; **type in the filter box** to narrow it.
3. `night` holds 392 samples (4.9%) and scores highest on both difficulty (5.48)
   and legibility (8.50) — legibility being an axis where high means *hard to
   read*. It is the hardest slice, though not the rarest: `dusk` has only 213
   images. That distinction is the point of the table — "rare" and "hard" are
   different questions, and a coverage bar chart only answers the first.
4. **Click the `night` row.** You land in the gallery at
   `/?attr=time_of_day:night`, filtered to exactly those images, with a removable
   `Attribute time_of_day:night` chip. The chart was not a picture — it was a
   query you could open.

Then send:

> Plot how the dataset splits into train, validation and test

- Hover a slice for the exact count. Click a legend row to drill in.
- The prose and the chart agree — `train 6,000 (75.0%)`. They agree because the
  tool hands the model the pre-computed shares; asked to derive them it reported
  60% under a chart correctly showing 75%.

Finally, to show the topology is read from the code rather than drawn:

> How does this platform work architecturally?

A flow diagram of the live system, including whether the embedding index is
currently loaded. Rendered as hand-written SVG — a graph-layout dependency would
have cost ~800 kB for this one block type.

## Act 3 — the QA agent and its status deck (~3 min)

Send:

> Show me the status of the application

The `qa` specialist runs alone (it is marked expensive, so it is never chosen
speculatively alongside another lane). It drives real Chrome over every registered
workflow and returns a status block inline:

- a `DONE` badge and the pass count — **63/63 checks · 11/11 workflows** on the run
  this walkthrough was written against, before three further workflows were
  registered, so expect a different denominator;
- one row per workflow — Routes, Gallery, Similarity map, Statistics, Quality,
  Benchmark, Sample detail, Assistant, Graceful degradation, Command palette,
  Assistant canvas — with its check tally and duration;
- a **screenshot of each workflow**, click to enlarge;
- an expandable list of every individual check;
- **`markdown` and `deck` download buttons.** The deck is a real `.pptx`
  (12 slides, ~7 MB): a title slide with the tally, then one slide per workflow
  with its status, screenshot and checks.

Observed: 47 s wall clock, 63/63 checks, 11/11 flows, no console errors. A good
chunk of that is the *Assistant canvas* flow, which waits on the local model to
route a request and produce a chart.

One of those workflows is *Graceful degradation*: it intercepts `/api/views` and
`/api/tags`, returns 500s, and asserts the UI says so — while a 404 (an optional
router simply absent) must stay quiet. Errors it causes on purpose are filed
separately from real ones, so a test proving the app survives a failure cannot
itself report the app as broken.

If a recent sweep exists the agent reads it instead of starting another — that is
instant, and its prompt tells it to prefer that. To force a fresh one:

```bash
curl -sX POST localhost:8000/api/qa/run -H 'Content-Type: application/json' -d '{}'
# {"run_id":"20260725-181632-02c8","status":"running", ...}
curl -s localhost:8000/api/qa/run/20260725-181632-02c8 | python3 -m json.tool
```

A second `POST` during a run returns the **same** run id rather than starting a
competing browser — two Chromes driving one dev server interleave their
navigations and both report nonsense.

The same flows run from the command line, because there is one definition of
them:

```bash
cd backend && uv run --with playwright --with python-pptx \
    --python .venv/bin/python python ../scripts/ui_smoke.py
#   PASS  route /
#   ...
#   90/90 checks passed in 76s
```

---

## Worth showing if there is time

**The command palette** — `⌘K` from anywhere. Type `night` to jump to an
attribute slice, a tag, or a saved view; type a sample id to open it; type free
text to search the dataset. It is the only affordance that reaches tags,
attribute slices and saved views without navigating first.

**Sets flowing between views** — the property the visualization layer is built
around. On **Map**, shift-drag a lasso over a region, then *Inspect N in gallery*:
those exact ids become the gallery filter. On **Quality**, drag the review
threshold and press *Review N images in gallery*: the button's count and the
gallery's total are the same number because both come from the same SQL
predicate. Neither view is a dead end.

**Graceful degradation** — stop Ollama and reload the Assistant tab. It explains
exactly what to install; everything else keeps working. This is the same pattern
for the embedding stack, the VLM, Playwright and `python-pptx`.

## Things that will look imperfect, and why

- **No streaming.** A 50-second report shows a spinner for 50 seconds. Streaming
  blocks as lanes finish is the version worth building and is not built.
- **Prose quality varies.** Routing, tool choice and wording are as good as a
  local 8B model, which is usually right and occasionally not. The defences are
  structural: tools supply the numbers, the routing menu states exclusions, and
  every chart shows its own provenance so the prose can be checked against it.
- **Duplicate trace chips.** A local model sometimes calls the same tool twice.
  The trace shows both, because that is what happened; identical *blocks* are
  deduplicated, so the chart appears once.
