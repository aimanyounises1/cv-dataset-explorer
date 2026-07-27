# CV Dataset Explorer

**A local tool for finding, auditing and curating slices of an image–caption
dataset — the 8,000 images and 40,000 human captions of
[Flickr8k](https://huggingface.co/datasets/jxie/flickr8k) — where every ranking,
score and measurement is labelled with what produced it.**

Working with a dataset means answering questions a file browser cannot: where are
the night scenes, which captions their own image does not support, which 300
samples are hardest, whether the held-out split is contaminated by
near-duplicates of training images. This answers those against a local corpus and
lets the answer leave as a slice you can regenerate. Everything runs on one
machine — SQLite for storage, SigLIP 2 for embeddings, no cloud services and no
paid APIs.

![The gallery: a hybrid search for "a crowded street at night", with the
difficulty sparkline, the retrieval-path evidence strip, and the selection rail
on the right](docs/screenshots/1-gallery.png)

Jump to: [what it looks like](#what-it-looks-like) ·
[architecture](docs/ARCHITECTURE.md) ·
[retrieval and evaluation](#retrieval-and-evaluation) ·
[limits](#scale-where-the-exact-search-stops-being-the-right-choice) ·
[setup](#setup) · [all documentation](#further-documentation)

## Why this is technically interesting

Four things, each checkable in this repository rather than taken on trust.

**1. Every score names its own basis, and the tool grades its own retrieval.**
Four search modes are labelled `cos`, `cos*`, `rrf` or `fit` on every card,
because a cosine, a hubness-corrected cosine, a fused rank and a log-likelihood
live on different scales and must never be read against each other. The Benchmark
page recomputes standard Flickr8k text→image recall for every mode on demand, and
excludes each query caption from the index it searches — without that exclusion
keyword recall measured 99.1%, which is self-retrieval, not retrieval.

**2. The retrieval work was measured, and most of it was refuted.** Normalising
the query text before SigLIP encodes it is worth +7.2 points of R@1 (46.0 → 53.2)
and the docs say which queries gain. A hubness correction improves MRR reliably
and R@1 only weakly (paired bootstrap CI `[+0.0048, +0.0178]` on MRR, McNemar
`p = 0.071` on R@1) and is reported that way. PRISM — a per-image speaker model
original to this project — refuted its own headline hypothesis, and in the app it
is [a wash](#the-boosted-mode-gain-honestly). Widening the lexical conjunction,
prompt templating and RRF tuning were all measured and not shipped.

**3. Layers, not a monolith.** Browsing, keyword search and statistics need
nothing but SQLite. Every ML capability is an optional layer that probes its own
availability and degrades with a message naming the command that would enable it
— never a 500, and never the same label over a different number. The
probe-and-behaviour table is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**4. Difficulty is stored, not eyeballed.** Every sample carries four
percentile-ranked axes — legibility, rarity, difficulty, clutter — with the raw
components behind each score, so "the hardest 300 samples in validation" is a
filter, a sort key and an export rather than an opinion. There is deliberately no
fifth axis: see [Reading the difficulty axes](#reading-the-difficulty-axes).

## Quick start

Two paths. The first is enough to see the product.

**Browse and keyword-search, no model download.** Downloads the dataset, skips
the 1.5 GB of model weights:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.ingest --limit 300 --skip-embeddings
uvicorn app.main:app --port 8000
```

In a second shell:

```bash
cd frontend && npm install && npm run dev      # http://localhost:5173
```

**Everything — semantic search, the embedding map, caption QA, the benchmark.**
One command, ~10–20 minutes on an Apple Silicon laptop, dominated by the dataset
and model downloads:

```bash
cd backend && python -m app.ingest
```

On macOS, once both installs are done, `./start.command` starts the API and the
dev server and opens the browser. Optional layers (the assistant, the VLM
enrichment, the browser QA sweep) and every flag are in [Setup](#setup).

## Contents

- [What it looks like](#what-it-looks-like) — one screenshot per view, each with the URL it came from
- [Core workflows](#core-workflows) — search, curation, saved views, the axes
- [Retrieval and evaluation](#retrieval-and-evaluation) — which protocol produced which number
- [Reading the difficulty axes](#reading-the-difficulty-axes) — what they mean and where they mislead
- [Scale](#scale-where-the-exact-search-stops-being-the-right-choice) — where exact search stops being right
- [Reproducibility](#reproducibility) — how to re-derive the figures and the claims
- [Data provenance and licensing](#data-provenance-and-licensing) — dataset, weights, and this repository
- [Setup](#setup) · [Tests](#tests) · [Configuration](#configuration) · [Architecture](#architecture)
- [Further documentation](#further-documentation)

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

Verdicts are tags with a convention, not a schema: record
`verdict:caption-error`, `verdict:scorer-error`, `verdict:ambiguous`,
`verdict:duplicate` or `verdict:ok` on the sample page, and the review session
becomes a filterable, exportable slice (`?tag=verdict:caption-error`) with zero
new tables.

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

## Core workflows

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

## Retrieval and evaluation

Four protocols in this repository produce retrieval numbers, and quoting one as
another is the easiest mistake available here. They are kept apart on purpose:

| Protocol | Pool | Queries | Where it runs | What it supports |
| --- | --- | --- | --- | --- |
| **In-app benchmark** | full 8,000-image corpus | 1,000 captions, fixed seed, hubness-bank captions excluded, each query caption excluded from the lexical index | the Benchmark page and `GET /api/eval/retrieval` | the per-mode table a reviewer sees, plus paired semantic-vs-boosted rows on a test-split sample |
| **Offline PRISM harness** | full 8,000-image corpus | ~5,000 test-split captions, 2 seeds, paired bootstrap | `python -m app.train_prism --eval`, [docs/PRISM.md](docs/PRISM.md) | the A0–A3 ablation ladder and the `+2.2 pts R@1` result against a 49.4% baseline |
| **Offline hubness A/B** | full 8,000-image corpus | one fixed 1,000-caption sample, held constant across arms | `python -m app.ml.hubness` | MRR 0.6280 → 0.6366 with the sample held fixed, and the R@1 result that only weakly replicates |
| **Published Flickr figures** | 1,000-image gallery | the literature protocol | cited in [docs/PRISM.md](docs/PRISM.md) | reference points only — an ~8x easier pool, never compared against the rows above |

Two consequences the repository states rather than smooths over:

- **The shipped boosted mode is a wash**, and the Benchmark page says so — see
  [the boosted mode gain, honestly](#the-boosted-mode-gain-honestly). The offline
  `+2.2 pts` came from a different protocol against a baseline eight points lower
  and cannot be quoted as the in-app gain.
- **Nothing is tuned on test.** Anything trained sees the train split; anything
  tuned is tuned on validation; the PRISM comparison rows are the only place test
  captions are used as queries, and they are used once per protocol.

What the tests pin about this, and what CI cannot check, is in
[docs/TESTING.md](docs/TESTING.md).

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
under a millisecond, which is about 2% of query latency; the text encode
dominates by two orders of magnitude. An ANN index here would optimise the
fastest stage of the pipeline while adding a dependency, a build step and recall
loss, so `EmbeddingIndex` stays exact and remains the single seam where that
would change.

It stops being the right choice somewhere around **~400k vectors** (measured by extrapolating the
scan against the encode — see [docs/TECHNICAL.md](docs/TECHNICAL.md)), or
whenever the scan exceeds ~10% of end-to-end query latency, whichever comes
first. At that point the substitution is local and does not change the API:
`EmbeddingIndex.search` already takes an `allowed_ids` candidate mask, so a
FAISS `IndexIVFFlat`, `hnswlib`, or `sqlite-vec` can be dropped in behind it.
Hosted vector databases are excluded by design — everything here runs on one
machine.

The index is one of two seams that scale would widen. The other is scoring: at
fleet scale the per-sample signals worth ranking by stop being intrinsic (blur,
rare words) and start coming from models — per-example loss, detector
confidence, ensemble disagreement. Those would enter through a `ScoreProvider`
protocol — `rank(query_vec, allowed_ids, k) → [(id, score, basis)]` — with the
hubness penalty and PRISM as its first two implementations, and every score
still arriving labelled with its basis. It is deliberately not implemented: at
8,000 images with two ranking signals, the protocol would be abstraction with
nothing to abstract over. A second dataset needs no new seam at all — it is one
adapter class in `app/datasets/`.

## Reproducibility

- **The figures.** Every screenshot in this README was captured from the running
  app by [`scripts/screenshots.py`](scripts/screenshots.py), which records the URL
  each one came from — so any claim can be checked by opening the same address.
  Regenerate them with the API on `:8000` and the dev server on `:5173`:
  `cd backend && .venv/bin/python ../scripts/screenshots.py` (`--headed` to watch).
- **The capability inventory.** [docs/CAPABILITIES.md](docs/CAPABILITIES.md) is
  generated from the live OpenAPI schema, the agent registry, the QA flow registry
  and the router in `App.tsx`. `python scripts/capabilities.py --check` fails when
  it is stale, so it cannot claim a capability the code does not have.
- **The benchmark.** Its cache key carries the protocol version, the query sample
  size, `CVDE_RRF_K`, `CVDE_SEARCH_DEPTH`, the hubness constants, whether PRISM
  artifacts exist, and the mtimes of the embeddings and the database — so a result
  computed under an older definition is never served as current. The query sample
  is drawn with a fixed seed.
- **What CI checks.** The backend suite, the frontend type-check and production
  build, and a relative-link check over the Markdown, on every pull request.
  Real model weights, ingestion, the browser sweep and the assistant do not run
  there — [docs/TESTING.md](docs/TESTING.md) lists both sides.
- **Configuration.** [`.env.example`](.env.example) lists every environment
  variable the code reads, with its default and what it changes.

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

**This repository's own source code has no licence file yet.** Under default
copyright that means no reuse rights are granted, which is a decision for the
owner rather than something to assume — it is deliberately not guessed here. It
is a separate question from the dataset terms above and from the model weights.

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

A real Chrome drives every registered workflow, screenshots each, and compiles a
pass/fail report plus a `.pptx` deck — including a degradation flow that injects
500s and asserts the UI announces them. The last full sweep (2026-07-26) reported
**90/90 checks across 15/15 workflows in 69 s**. Re-run it rather than quoting that
number: the flow registry grows, and the count is only true of the day it was taken.
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

On the light install CI uses (no torch, no transformers, no langgraph — see
[docs/TESTING.md](docs/TESTING.md)) that suite reports **173 passed, 4 skipped**.
The skips are the modules that need `torch` or `langgraph`; they run locally once
those are installed.

## Configuration

Everything is environment variables, and [`.env.example`](.env.example) lists
every one the code reads with its default, what it changes, and the
model-versus-index warning that matters most. Nothing auto-loads that file:
`app/config.py` reads the process environment, so export what you need or pass
the file to `uvicorn --env-file`.

The ones worth knowing before changing anything: `CVDE_DATA_DIR` (where every
generated artifact lives), `CVDE_EMBED_MODEL` (must be identical for indexing and
for serving), `CVDE_SEARCH_DEPTH` (the 300-row ranking horizon), `CVDE_RRF_K`
(the fusion constant, reported with every fused response) and
`CVDE_HUBNESS_BETA` (0 restores the plain cosine ranking exactly).

## Architecture

One FastAPI process, one SQLite file, one directory of images, and a React
frontend whose entire view state lives in the URL. A request does SQLite lookups
plus at most one text-encoder forward pass; everything heavy is precomputed by
idempotent batch CLIs.

```
frontend/                 React 18 + TypeScript + Vite
  src/pages/              Gallery · Sample · Map · Stats · Quality · Benchmark · Assistant
  src/components/         FilterBar · ImageCard · canvas ScatterPlot · CommandPalette (⌘K)
  src/components/blocks/  one renderer per visualization kind + an exhaustive registry
  src/lib/viz.ts          the only source of colour and axis tokens

backend/                  FastAPI + SQLite
  app/api/                samples · search · export · stats · map · tags · views · describe
                          attributes · qa · qa_run · eval · leakage · admin · chat
  app/datasets/           adapter interface + the Flickr8k adapter (pluggable)
  app/ml/                 SigLIP 2 embedder · exact index · hubness · PRISM · UMAP · labels
  app/agent/              registry · graph (parallel orchestration) · blocks · tools
  app/qa/                 flow registry · browser runner · deck
  app/ingest.py           download → store → index → embed → project
  app/analyze.py          caption embeddings · agreement · attributes · difficulty axes
  data/                   images · thumbs · explorer.db · embeddings · cache · qa · reports
                          (all gitignored; nothing generated is committed)
```

The reasoning — a rendered topology diagram, the degradation table, how model and
index consistency is enforced, how the splits stay apart, and the production scale
path — is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Further documentation

| Document | What is in it |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | runtime topology, seams, degradation boundaries, scale path |
| [docs/TECHNICAL.md](docs/TECHNICAL.md) | schema, the real SQL with query plans, retrieval maths, frontend, measured performance and the ceilings it will hit |
| [docs/DESIGN.md](docs/DESIGN.md) | who the user is, what was deliberately not built, and the trade-offs accepted |
| [docs/CAPABILITIES.md](docs/CAPABILITIES.md) | generated inventory: every view, endpoint, agent tool and tested workflow |
| [docs/TESTING.md](docs/TESTING.md) | what each test tier covers, what CI cannot see, and the known gaps |
| [docs/PRISM.md](docs/PRISM.md) | the speaker-model method, its pre-registered predictions, and the results that refuted them |
| [docs/PRISM.md](docs/PRISM.md) | the retrieval-accuracy research programme: ranked falsifiable hypotheses with noise floors and pool honesty, and the method this project proposes |
| [docs/AGENTS.md](docs/AGENTS.md) | agent orchestration, the render-block contract, the self-QA sweep |
| [docs/DEMO.md](docs/DEMO.md) | an eight-minute walkthrough |
| [docs/screenshots/](docs/screenshots/) | one image per view |
| [CONTRIBUTING.md](CONTRIBUTING.md) | the six checks a change has to pass, and the conventions that are load-bearing |
| [SECURITY.md](SECURITY.md) | the security model of a local single-user tool |
