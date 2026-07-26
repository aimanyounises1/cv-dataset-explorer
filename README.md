# CV Dataset Explorer

A local, web-based visualization and exploration tool for image–caption datasets,
built for the [Flickr8k dataset](https://huggingface.co/datasets/jxie/flickr8k)
(8,000 images × 5 human captions — see [Data provenance](#data-provenance-and-licensing)).

Everything runs on a single developer machine: SQLite for storage, SigLIP 2 for
local embeddings, no cloud services or paid APIs.

![The gallery: a hybrid search for "a crowded street at night", with the
difficulty sparkline, the retrieval-path evidence strip, and the selection rail
on the right](docs/screenshots/1-gallery.png)

Every figure below was captured from the running app by
[`scripts/screenshots.py`](scripts/screenshots.py), which records the URL each one
came from — so any claim here can be checked by opening the same address.
Regenerate them with the API on `:8000` and the dev server on `:5173`:

```bash
cd backend && .venv/bin/python ../scripts/screenshots.py     # --headed to watch
```

## Features

- **Gallery** — browse all samples with split / tag / attribute filters. All search
  and filter state lives in the URL: shareable links, working back-button.
- **Search** — four modes, switchable in the UI:
  - *Semantic*: natural-language text-to-image search via SigLIP 2 embeddings ("dog jumping into water").
  - *Keyword*: BM25 full-text search over all 40k captions (SQLite FTS5, Porter-stemmed).
  - *Hybrid* (default): reciprocal-rank fusion of both.
  - *Boosted*: the semantic ranking replaced by **PRISM speaker models trained on this corpus** (`python -m app.train_prism --no-sigma`). Degrades gracefully to semantic when no trained artifacts exist. **Two different numbers describe this and they are not interchangeable** — see [the gain, honestly](#the-boosted-mode-gain-honestly) before quoting either.

  Every result shows *why* it matched: the best-matching caption, highlighted terms, and the relevance score.
- **Sample inspector** — full image, all 5 captions **with image-caption agreement scores**, zero-shot attributes, metadata, editable curation tags, and "similar images" via embedding nearest neighbors.
- **Embedding map** — interactive UMAP scatter of the whole dataset (zoom/pan/hover thumbnails, click-through). **Shift+drag selects a region for bulk tagging** — lasso a visual cluster, name it, filter the gallery by it.
- **Statistics** — split sizes, caption length/vocabulary distributions, **zero-shot attribute coverage** (click a bar to open that slice in the gallery — small slices are the long tail), and **near-duplicate detection**.
- **Difficulty axes** — every sample is scored 0–10 on four axes describing how *hard* it is, not what is in it: **legibility** (blur and darkness), **rarity** (rare caption vocabulary, and isolation in embedding space), **difficulty** (where image–caption agreement and inter-caption agreement are weakest), and **clutter** (how much the captions name, and how much their lengths vary). Each is a range filter, a badge on every result card, and a sort key — so "show me the hardest 300 samples in the validation split" is a query the tool can answer. See [Reading the difficulty axes](#reading-the-difficulty-axes).
- **Quality (annotation QA)** — CLIPScore-style ranking of captions least supported by their image (likely annotation errors), plus samples whose 5 captions disagree most with each other.
- **Benchmark** — the tool measures its own search quality: standard Flickr8k text→image retrieval recall@1/5/10 for every mode, using the dataset's captions as ground truth. When a trained PRISM model is present, the table adds a paired test-split comparison so the boosted mode's gain is measured, not asserted.
- **Assistant (optional)** — a chat interface backed by a **Fugu-style multi-agent orchestration** (LangGraph over local Ollama): an orchestrator routes requests to retrieval and insights specialist agents, and a synthesizer quality-gates the answer. Agents use the same service functions as the REST API and can search, inspect, analyze coverage, audit captions, and tag samples. The UI shows the agent/tool trace for every answer.
- **Curation workflow, closing both ways** — the point of a search tool over a dataset is composing a training set, so a slice has to be able to leave and come back:
  - **Out:** export the current view — filters *or* a ranked search result — as JSON, JSONL or CSV. The manifest records the query, the axis bounds and the embedding model, because a slice you cannot regenerate is not curation.
  - **In:** paste or upload a list of ids or filenames (the **Id list** panel). Both are accepted because both are things you already have — this tool's own export, or anything that touched the images on disk. It composes with every other filter rather than replacing them, and reports how many entries exist here, so a list carried over from a larger corpus tells you "412 of your 500" instead of failing.
  - Tag samples manually, in bulk by lassoing a region of the map, or via the assistant; filter by tag.
- **Named views** — save the current filter set under a name and restore it later. Stored as the URL query string, opaquely, so a view keeps working when the UI grows a filter the backend has no column for.
- **Legible filter state** — every active constraint appears as a removable chip above the results, so you never reach an empty page wondering which of five filters emptied it.
- **Optional VLM enrichment** — tag every image with a local vision-language model via [Ollama](https://ollama.com).

**Design intent — layers, not a monolith.** Browsing, keyword search, and stats
run on plain SQLite with nothing else installed; every ML capability (semantic
search, map, QA, benchmark, assistant) is an optional layer that reports its
own availability and degrades gracefully when its prerequisites are missing —
without embeddings you still get browsing/keyword search/stats, and without
the agent stack the assistant tab explains exactly how to enable it.

## What it looks like

### Describing a selection, and drilling into it

![The "what is in this selection?" panel over the 223 images that are both night
and indoor, listing over- and under-represented attributes with lift multipliers
and raw counts](docs/screenshots/2-describe.png)

`/?attr=time_of_day:night&attr=setting:indoor` — Every other view answers *given a
filter, which samples*. This runs it backwards: *given these samples, what do they
have in common*. Each row carries its raw count next to the multiplier, because ×6
over five images and ×6 over five hundred are different findings, and each is
tested against the hypergeometric distribution before it is shown.

Two things it deliberately will not do. Clicking a row **narrows** this selection
rather than replacing it — the count on the row is measured inside the current set,
so it has to. And facets from a group you already filtered by are suppressed: a
sample carries exactly one label per group, so "these night images are ×20 more
nocturnal than the corpus" is true, useless, and would otherwise be the largest
number on the page.

### Four search modes, each labelled with what produced its score

![The boosted mode: results for "a crowded street at night" re-ranked by the
trained PRISM speaker models, each card showing its candidate rank and
fit score](docs/screenshots/3-boosted.png)

`/?q=a+crowded+street+at+night&mode=boosted&sort=rarity_desc` — A score is only
interpretable next to what produced it, so every card names its basis: `cos` for a
plain cosine, `cos*` when the hubness correction re-ranked it, `rrf` for a fused
rank, `fit` for a PRISM log-likelihood. These live on different scales and must
never be read against each other. `boost 171` means this image placed 171st in the
semantic candidate pool before re-ranking — the gap is the correction doing work.

### The difficulty axes as a filter

![The gallery filtered to samples scoring 8 or above on both difficulty and
legibility](docs/screenshots/4-axes.png)

`/?difficulty_min=8&legibility_min=8` — "Show me the hardest samples" is a query
this tool can answer, because *hard* is stored rather than eyeballed. See
[Reading the difficulty axes](#reading-the-difficulty-axes) for what each of the
four measures and where they mislead.

### Embedding map

![UMAP scatter of all 8,000 images, coloured by cluster, with a hover
thumbnail](docs/screenshots/5-map.png)

`/map` — Shift+drag lassoes a region into a named tag, which then filters the
gallery like any other constraint. The projection is for looking at; every
similarity the tool acts on is computed in the full 768 dimensions.

### Caption quality

![The caption quality page: a distribution of image-caption agreement with a
brush, over the captions least supported by their image](docs/screenshots/6-quality.png)

`/quality` — Captions ranked by how little their own image supports them, which is
where annotation errors are. The threshold brush is a real filter, so a triage
selection made here can leave the page it was made on.

### Dataset profile

![Split sizes, caption length and vocabulary distributions, zero-shot attribute
coverage, and near-duplicate pairs](docs/screenshots/7-stats.png)

`/stats` — Clicking an attribute bar opens that slice in the gallery. The small
bars are the long tail, which is the point.

### The tool measuring its own search quality

![The benchmark page: recall chart and table for semantic, keyword, hybrid, and a
paired test-split comparison of semantic against boosted](docs/screenshots/8-eval.png)

`/eval` — Flickr8k's captions are ground truth, so the tool can grade itself. The
query caption is excluded from the index it searches; without that the number
measures nothing but self-retrieval, and it was 99.1%.

Read the keyword row as a property of the query rather than of BM25: these queries
are whole ~12-word captions and keyword mode requires every content term in one
caption, so for 85% of them the lexical path has nothing to rank at all. The
candidates column reports that directly rather than letting the recall figure imply
a ranking failure.

One caveat on that column, disclosed rather than quietly re-derived: **hybrid's
figure is a sum, not a set**. It adds the semantic pool to the mean lexical match
count, but every lexical match is already inside the semantic pool, so two
overlapping sets are added where they should be unioned and the honest figure is
the pool itself (8,000). The overstatement equals the lexical mean — 2.1 here.
Correcting the number would change what every cached run means, so the page says
what it is instead.

#### The boosted mode gain, honestly

The two bottom rows are the like-for-like comparison — the same test-split queries,
ranked by semantic and by boosted. This is what the page above reports:

| on the same 1,000 test queries | R@1 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|
| semantic | 57.8% | **81.0%** | **86.8%** | 0.6746 |
| boosted (PRISM) | **57.9%** | 80.3% | 86.4% | **0.6774** |

So in this app, on this protocol, boosted is **+0.1 pts R@1 and +0.003 MRR**, and
slightly *behind* on R@5 and R@10. It is a wash, and the page says so.

That is not the `+2.2 pts R@1` figure in [docs/PRISM.md](docs/PRISM.md), and the
two are not comparable. That result is from PRISM's own offline harness against a
**49.4% baseline** (49.4 → 51.6, two seeds, paired bootstrap CI `[+1.28, +3.18]`);
this page's semantic baseline on test queries is **57.8%**. The two runs use
different query samples; the candidate pool is identical (8,000 images in both).
The residual gap is under investigation, so the offline delta cannot be quoted as
the in-app gain — a reviewer who clicks **Run benchmark** gets the table above.

The offline ablation is worth reading on its own terms: it refuted the per-image
variance hypothesis it was built to test and kept only the component that survived
measurement. But the honest headline for the shipped feature is the wash, not the
ablation's win.

### Sample inspector

![A sample detail page: full image, all five captions with agreement scores,
zero-shot attributes, metadata, tags, and nearest neighbours](docs/screenshots/9-sample.png)

`/samples/1865` — All five captions with their individual agreement scores, so a
disagreement is visible rather than averaged away, plus nearest neighbours in
embedding space.

### Command palette

![The command palette filtered on "night", offering tags, attributes, saved views
and actions](docs/screenshots/10-palette.png)

`⌘K` — One place to reach any filter, saved view, or action by name.

### Assistant

![The assistant answering "show me the 12 images with the worst caption
agreement" with a tool trace and an interactive image block](docs/screenshots/11-assistant.png)

`/chat` — A LangGraph orchestrator over local Ollama routes to retrieval and
insights specialists. Two things worth noticing: the chips name every agent and
tool the turn used, and the answer is a **rendered block** the UI can make
interactive, not prose describing images you then have to go find. Answers are
generated by a local 8B model, so quality varies between runs.

## Reading the difficulty axes

Computed once by `python -m app.analyze --only axes` and stored on each sample.
Three things about them are worth knowing before you trust a number.

**They are percentile ranks, not measurements.** A Laplacian variance, a cosine
distance and an inverse document frequency live on entirely different scales
with distributions you cannot guess in advance, so a range filter over the raw
values behaves erratically — "blur ≥ 40" means something different on every
dataset and nothing at all to a person. Each axis is therefore ranked across the
dataset and bucketed 0–10, which makes `rarity ≥ 7` and `difficulty ≥ 7` both
mean "roughly the top 30% of this corpus" and lets four sliders be used
together. The cost: the scores are **dataset-relative**. A 7 here is not a 7 on
COCO, the buckets are near-uniformly populated by construction, and ingesting
more images can move a sample's bucket without its pixels changing.

**Every score carries its components.** `axis_detail` stores the raw values
behind each axis (blur, luminance, agreement, and so on), so the interface can
explain a score in place rather than asking you to trust it. Nothing here is
model-generated prose — the explanations are templated from measured numbers.

**There is no fifth axis, on purpose.** Systems of this kind usually carry a
*dynamic complexity* axis — how badly the agents in a scene are behaving. There
is no honest analogue in Flickr8k: these are still photographs, with no motion,
no agents and no rules to violate. Inventing one to round the count to five
would have made the panel look more complete and the data less true, so the
axis is absent and this paragraph is the reason.

### Two limits worth knowing before you rely on them

**Ranked results stop at 300 per query.** Reciprocal-rank fusion is computed over
the retrieved candidate lists, so its output depends on how deep those lists go:
row 300 of a 300-deep fusion is a different image from row 300 of a 350-deep one.
Widening the pool to let a user page further therefore re-ranks the tail and
repeats images across adjacent pages — measured, before this was fixed, at 4
duplicates either side of the boundary. The depth is now a hard horizon: paging
stops there and the gallery says so, rather than offering a "Load more" that
quietly runs out. Raise `CVDE_SEARCH_DEPTH` to see further, or narrow the query.

**A pasted id list is capped at 60,000 entries.** Past 10,000 the entries go into
a temporary table rather than an `IN (...)` clause, because SQLite binds each
entry as a host parameter and its default ceiling is 32,766 — a list of ~40,000
would otherwise fail with "too many SQL variables" rather than working.

## Scale: where the exact search stops being the right choice

Retrieval is exact brute-force cosine in NumPy — no approximate-nearest-neighbour
index. At 8,000 × 768 the embedding matrix is ~25 MB and a full scan takes well
under a millisecond, which is under 1% of query latency; the text encode
dominates by two orders of magnitude. An ANN index here would optimise the
fastest stage of the pipeline while adding a dependency, a build step and recall
loss, so `EmbeddingIndex` stays exact and remains the single seam where that
would change.

It stops being the right choice somewhere around **100k–500k vectors**, or
whenever the scan exceeds ~10% of end-to-end query latency, whichever comes
first. At that point the substitution is local and does not change the API:
`EmbeddingIndex.search` already takes an `allowed_ids` candidate mask, so a
FAISS `IndexIVFFlat`, `hnswlib`, or `sqlite-vec` can be dropped in behind it.
Hosted vector databases are excluded by design — everything here runs on one
machine.

## Data provenance and licensing

**What gets ingested.** The `jxie/flickr8k` copy on Hugging Face contains exactly
**8,000 rows** (6,000 train / 1,000 validation / 1,000 test), while the original
Flickr8k distribution has roughly **8,091 images**. About 90 images are absent
from this copy, with no explanation upstream — worth knowing before comparing
any number here against a published Flickr8k result. The split *counts* match the
canonical Hodosh split; the per-image *assignments* are undocumented in this copy
and are not verified by this tool.

**Licensing.** Upstream Flickr8k is distributed for **non-commercial research and
education only**, and the individual photographs remain under their original
Flickr licenses. The `jxie/flickr8k` copy specifies **no license of its own**, so
the upstream terms are the safe assumption. This repository contains **no dataset
images**: ingestion downloads them to your machine, and `data/` is gitignored.
Treat anything you export as carrying the same restrictions.

**Model weights** are downloaded from Hugging Face at first use
(SigLIP 2 base, ~1.5 GB, Apache-2.0). The optional Ollama models carry their own
licenses. No dataset or model weights are redistributed here.

**Network access.** Preparation makes two one-time downloads — the dataset and the
model weights. After they are cached, the entire system runs offline; no cloud
services, hosted APIs, or external vector databases are in the runtime path.

## Requirements

- Python 3.10+
- Node.js 18+
- ~2 GB disk for the dataset, images, and model weights
- Any OS; Apple Silicon (MPS) and NVIDIA (CUDA) are used automatically when available, otherwise CPU

## Setup

### 1. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ingest the dataset (one-time, idempotent)

```bash
python -m app.ingest
```

This downloads Flickr8k from Hugging Face, stores images + thumbnails +
captions locally, builds the FTS index, computes SigLIP 2 embeddings, fits the
UMAP projection, and runs the analysis passes (caption QA scores + zero-shot
attributes). Expect ~10–20 minutes total on an Apple Silicon laptop (dataset
download is the bulk of it).

Useful variants:

```bash
python -m app.ingest --limit 300        # quick trial run
python -m app.ingest --skip-embeddings  # browse/keyword-search only, no model download
python -m app.analyze                   # (re)run QA scores + attributes on an existing DB
```

If the API is already running, `curl -X POST localhost:8000/api/admin/reload`
picks up new embeddings/analysis without a restart.

### 3. Run the API

```bash
uvicorn app.main:app --port 8000
```

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

### 5. (Optional) Assistant — Fugu-style agent orchestration

With [Ollama](https://ollama.com) installed:

```bash
cd backend && pip install -r requirements-agent.txt
ollama pull qwen3:8b        # any Ollama chat model with tool calling
uvicorn app.main:app --port 8000   # restart the API
```

The Assistant tab then routes each question through an orchestrator → up to two
specialist agents in parallel → a quality-gate synthesizer, all running locally.
Four specialists are registered: **retrieval** (search, similar, inspect, tag),
**insights** (statistics, coverage, caption QA), **visualization** (charts,
diagrams, reports) and **qa** (drives the app itself and reports its status).

Answers come back as **interactive components, not prose about data**: charts you
can hover, sort and zoom, and whose bars, slices and rows navigate to the exact
gallery slice they count. Every chart states the SQL that produced it. Try:

- *"Plot how the dataset splits into train, validation and test"*
- *"Which time of day is hardest? Compare the slices"*
- *"Generate a dataset report"* — rendered inline, downloadable as Markdown/JSON
- *"How does this platform work architecturally?"*
- *"Show me the status of the application"* — see step 6

`docs/AGENTS.md` covers the design; `docs/DEMO.md` is an eight-minute walkthrough.

### 6. (Optional) Autonomous UI QA and status deck

```bash
cd backend && uv pip install --python .venv/bin/python -r requirements-qa.txt
```

Then ask the assistant *"show me the status of the application"*, or:

```bash
curl -sX POST localhost:8000/api/qa/run -H 'Content-Type: application/json' -d '{}'
```

A real Chrome drives all fifteen workflows, screenshots each, and compiles a
pass/fail report plus a `.pptx` deck (observed 2026-07-26: **90/90 checks, 15/15
workflows, 69 s** — including a degradation flow that injects 500s and asserts the
UI announces them).
One sweep runs at a time; a second request attaches to the one in flight.
Artifacts land in `backend/data/qa/<run_id>/` and are served at `/media/qa/`.

The same flows back the command-line smoke test — one definition, three
consumers — so the suite a developer runs cannot drift from the one the app runs:

```bash
cd backend && uv run --with playwright --with python-pptx \
    --python .venv/bin/python python ../scripts/ui_smoke.py
```

Playwright and `python-pptx` stay out of `requirements.txt` deliberately: neither
is needed to serve the app. Without them the endpoint returns setup instructions,
and without `python-pptx` alone the Markdown report is still produced.

### 7. (Optional) VLM enrichment

```bash
ollama pull qwen2.5vl:7b
cd backend && python -m app.enrich
```

Runtime is roughly 1–3 s/image locally; `--limit 500` enriches a subset. The
app is fully functional without either optional step.

## Tests

```bash
cd backend && pytest
```

The smoke tests exercise the API end-to-end on a seeded temporary database,
including the degraded-mode path (no embeddings installed). The agent suite runs
without Ollama and without a browser — the graph accepts an injected model, so
parallel fan-out, lane isolation and lane timeouts are asserted at unit-test
speed rather than assumed.

For the UI, `scripts/ui_smoke.py` drives real Chrome through every workflow (see
step 6). It is the tier that catches what `tsc` cannot: a view that renders empty,
a control that stopped filtering, a console error, a 404.

## Configuration

All via environment variables: `CVDE_DATA_DIR`, `CVDE_EMBED_MODEL`,
`CVDE_EMBED_BATCH`, `CVDE_OLLAMA_URL`, `CVDE_VLM_MODEL`, `CVDE_CHAT_MODEL`,
`CVDE_DUP_THRESHOLD`, and the agent bounds `CVDE_OLLAMA_TIMEOUT` (120 s per model
call), `CVDE_AGENT_LANE_TIMEOUT` (240 s per specialist),
`CVDE_AGENT_RECURSION_LIMIT` (25 tool steps), plus `CVDE_QA_BASE_URL` for what
the QA browser drives.

## Architecture

```
frontend/  React 18 + TypeScript + Vite
  src/pages          Gallery | Sample detail | Map | Stats | Quality | Benchmark | Assistant
  src/components     FilterBar, ImageCard, Highlight, TagEditor, canvas ScatterPlot,
                     CommandPalette (⌘K), AxisSparkline
  src/components/blocks/  One renderer per visualization kind + an exhaustive registry
  src/lib/           viz.ts (the only source of colour/axis tokens) · mapColor.ts
  (dev server proxies /api and /media to the backend; state a colleague must
   reproduce lives in the URL, ephemeral view preference in session/localStorage)

backend/   FastAPI + SQLite
  app/datasets/   Dataset adapter interface + Flickr8k adapter (pluggable)
  app/ingest.py   One-time pipeline: download → store → index → embed → project → analyze
  app/analyze.py  Caption embeddings · QA scores · zero-shot attributes
  app/enrich.py   Optional local-VLM tagging via Ollama
  app/ml/         SigLIP 2 embedder · exact search index · UMAP/KMeans · label bank
  app/agent/      registry.py (declare an agent) · graph.py (parallel orchestration)
                  blocks.py (the render-block contract) · viz_tools · tools · qa_tools
  app/qa/         Autonomous UI sweep: flows.py (declare a workflow) · runner · deck
  app/api/        samples · search · stats · map · tags · qa · qa_run · attributes ·
                  describe · leakage · views · eval · admin · chat
  data/           images/ thumbs/ explorer.db embeddings/ qa/ reports/ (gitignored)
```

**Everything this system does, in one place:**
[docs/CAPABILITIES.md](docs/CAPABILITIES.md) — every view, HTTP endpoint, agent
tool and tested workflow. It is *generated* from the live OpenAPI schema, the
agent registry, the QA flow registry and the router in `App.tsx`
(`python scripts/capabilities.py`, `--check` fails when it is stale), so it
cannot claim a capability the code does not have.

**How it is built, layer by layer** — schema, the actual SQL with real query
plans, retrieval maths, frontend architecture, measured performance and the
ceilings it will hit: [docs/TECHNICAL.md](docs/TECHNICAL.md).

Design rationale and trade-offs: see [docs/DESIGN.md](docs/DESIGN.md). The agent
layer — orchestration, the visualization canvas, and the self-QA sweep — is
documented separately in [docs/AGENTS.md](docs/AGENTS.md), with an
eight-minute walkthrough in [docs/DEMO.md](docs/DEMO.md) and one screenshot per
view in [docs/screenshots/](docs/screenshots/).
