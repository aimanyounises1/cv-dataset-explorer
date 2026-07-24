# CV Dataset Explorer

A local, web-based visualization and exploration tool for image–caption datasets,
built for the [Flickr8k dataset](https://huggingface.co/datasets/jxie/flickr8k)
(8,091 images × 5 human captions).

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
