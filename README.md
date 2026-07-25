# CV Dataset Explorer

A local, web-based visualization and exploration tool for image–caption datasets,
built for the [Flickr8k dataset](https://huggingface.co/datasets/jxie/flickr8k)
(8,000 images × 5 human captions — see [Data provenance](#data-provenance-and-licensing)).

Everything runs on a single developer machine: SQLite for storage, SigLIP 2 for
local embeddings, no cloud services or paid APIs.

## Features

- **Gallery** — browse all samples with split / tag / attribute filters. All search
  and filter state lives in the URL: shareable links, working back-button.
- **Search** — three modes, switchable in the UI:
  - *Semantic*: natural-language text-to-image search via SigLIP 2 embeddings ("dog jumping into water").
  - *Keyword*: BM25 full-text search over all 40k captions (SQLite FTS5, Porter-stemmed).
  - *Hybrid* (default): reciprocal-rank fusion of both.

  Every result shows *why* it matched: the best-matching caption, highlighted terms, and the relevance score.
- **Sample inspector** — full image, all 5 captions **with image-caption agreement scores**, zero-shot attributes, metadata, editable curation tags, and "similar images" via embedding nearest neighbors.
- **Embedding map** — interactive UMAP scatter of the whole dataset (zoom/pan/hover thumbnails, click-through). **Shift+drag selects a region for bulk tagging** — lasso a visual cluster, name it, filter the gallery by it.
- **Statistics** — split sizes, caption length/vocabulary distributions, **zero-shot attribute coverage** (click a bar to open that slice in the gallery — small slices are the long tail), and **near-duplicate detection**.
- **Difficulty axes** — every sample is scored 0–10 on four axes describing how *hard* it is, not what is in it: **legibility** (blur and darkness), **rarity** (rare caption vocabulary, and isolation in embedding space), **difficulty** (where image–caption agreement and inter-caption agreement are weakest), and **clutter** (how much the captions name, and how much their lengths vary). Each is a range filter, a badge on every result card, and a sort key — so "show me the hardest 300 samples in the validation split" is a query the tool can answer. See [Reading the difficulty axes](#reading-the-difficulty-axes).
- **Quality (annotation QA)** — CLIPScore-style ranking of captions least supported by their image (likely annotation errors), plus samples whose 5 captions disagree most with each other.
- **Benchmark** — the tool measures its own search quality: standard Flickr8k text→image retrieval recall@1/5/10 for all three modes, using the dataset's captions as ground truth.
- **Assistant (optional)** — a chat interface backed by a **Fugu-style multi-agent orchestration** (LangGraph over local Ollama): an orchestrator routes requests to retrieval and insights specialist agents, and a synthesizer quality-gates the answer. Agents use the same service functions as the REST API and can search, inspect, analyze coverage, audit captions, and tag samples. The UI shows the agent/tool trace for every answer.
- **Curation workflow** — tag samples (manually, in bulk from the map, or via the assistant), filter by tag, and export any filtered subset as a JSON manifest (`GET /api/export`).
- **Optional VLM enrichment** — tag every image with a local vision-language model via [Ollama](https://ollama.com).

**Design intent — layers, not a monolith.** Browsing, keyword search, and stats
run on plain SQLite with nothing else installed; every ML capability (semantic
search, map, QA, benchmark, assistant) is an optional layer that reports its
own availability and degrades gracefully when its prerequisites are missing —
without embeddings you still get browsing/keyword search/stats, and without
the agent stack the assistant tab explains exactly how to enable it.

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

The Assistant tab then routes your questions through an orchestrator →
specialist agents (retrieval / insights) → quality-gate synthesizer graph,
all running locally. Try: *"find dogs jumping into water"*, *"what are the
rarest slices?"*, *"tag the 5 most suspect captions as needs-review"*.

### 6. (Optional) VLM enrichment

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
including the degraded-mode path (no embeddings installed).

## Configuration

All via environment variables: `CVDE_DATA_DIR`, `CVDE_EMBED_MODEL`,
`CVDE_EMBED_BATCH`, `CVDE_OLLAMA_URL`, `CVDE_VLM_MODEL`, `CVDE_CHAT_MODEL`,
`CVDE_DUP_THRESHOLD`.

## Architecture

```
frontend/  React 18 + TypeScript + Vite
  src/pages       Gallery | Sample detail | Map | Stats | Quality | Benchmark | Assistant
  src/components  FilterBar, ImageCard, Highlight, TagEditor, canvas ScatterPlot
  (dev server proxies /api and /media to the backend; view state lives in the URL)

backend/   FastAPI + SQLite
  app/datasets/   Dataset adapter interface + Flickr8k adapter (pluggable)
  app/ingest.py   One-time pipeline: download → store → index → embed → project → analyze
  app/analyze.py  Caption embeddings · QA scores · zero-shot attributes
  app/enrich.py   Optional local-VLM tagging via Ollama
  app/ml/         SigLIP 2 embedder · exact search index · UMAP/KMeans · label bank
  app/agent/      Fugu-style LangGraph orchestration + platform tools (optional)
  app/api/        samples · search · stats · map · tags · qa · attributes · eval · admin · chat
  data/           images/ thumbs/ explorer.db embeddings/ (generated, gitignored)
```

Design rationale and trade-offs: see [docs/DESIGN.md](docs/DESIGN.md).
