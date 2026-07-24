# CV Dataset Explorer

A local, web-based visualization and exploration tool for image–caption datasets,
built for the [Flickr8k dataset](https://huggingface.co/datasets/jxie/flickr8k)
(8,091 images × 5 human captions).

Everything runs on a single developer machine: SQLite for storage, SigLIP 2 for
local embeddings, no cloud services or paid APIs.

## Features

- **Gallery** — browse all samples with split / tag filters and lazy-loaded thumbnails.
- **Search** — three modes, switchable in the UI:
  - *Semantic*: natural-language text-to-image search via SigLIP 2 embeddings ("dog jumping into water").
  - *Keyword*: BM25 full-text search over all 40k captions (SQLite FTS5).
  - *Hybrid* (default): reciprocal-rank fusion of both.
- **Sample inspector** — full image, all 5 captions, metadata, editable curation tags, and "similar images" via embedding nearest neighbors.
- **Embedding map** — interactive UMAP scatter of the whole dataset (zoom/pan/hover thumbnails, click-through), colored by cluster. Useful for spotting clusters, outliers, and coverage gaps.
- **Statistics** — split sizes, caption length/vocabulary distributions, image sizes, and **near-duplicate detection** (embedding cosine > 0.95).
- **Curation workflow** — tag samples (e.g. `edge-case`, `mislabeled`), filter by tag, and export any filtered subset as a JSON manifest (`GET /api/export`).
- **Optional VLM enrichment** — tag every image with a local vision-language model via [Ollama](https://ollama.com) (objects, scene, conditions). Tags become filterable facets and extend keyword search beyond what the human captions mention.

The app **degrades gracefully**: without embeddings it still serves browsing,
keyword search, and stats — and tells you what's disabled and how to enable it.

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
captions locally, builds the FTS index, computes SigLIP 2 embeddings, and fits
the UMAP projection. Expect ~10–20 minutes total on an Apple Silicon laptop
(dataset download is the bulk of it; embedding 8k images takes a few minutes on MPS).

Useful variants:

```bash
python -m app.ingest --limit 300        # quick trial run
python -m app.ingest --skip-embeddings  # browse/keyword-search only, no model download
```

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

### 5. (Optional) VLM enrichment

With [Ollama](https://ollama.com) installed:

```bash
ollama pull qwen2.5vl:7b
cd backend && python -m app.enrich
```

Runtime is roughly 1–3 s/image locally; `--limit 500` enriches a subset. The
app is fully functional without this step.

## Tests

```bash
cd backend && pytest
```

The smoke tests exercise the API end-to-end on a seeded temporary database,
including the degraded-mode path (no embeddings installed).

## Configuration

All via environment variables: `CVDE_DATA_DIR`, `CVDE_EMBED_MODEL`,
`CVDE_EMBED_BATCH`, `CVDE_OLLAMA_URL`, `CVDE_VLM_MODEL`, `CVDE_DUP_THRESHOLD`.

## Architecture

```
frontend/  React 18 + TypeScript + Vite
  src/pages       Gallery | Sample detail | Embedding map | Stats
  src/components  SearchBar/FilterBar, ImageCard, TagEditor, canvas ScatterPlot
  (dev server proxies /api and /media to the backend)

backend/   FastAPI + SQLite
  app/datasets/   Dataset adapter interface + Flickr8k adapter (pluggable)
  app/ingest.py   One-time pipeline: download → store → index → embed → project
  app/enrich.py   Optional local-VLM tagging via Ollama
  app/ml/         SigLIP 2 embedder · in-memory exact search index · UMAP/KMeans
  app/api/        samples · search · stats · map · tags routers
  data/           images/ thumbs/ explorer.db embeddings/ (generated, gitignored)
```

Design rationale and trade-offs: see [docs/DESIGN.md](docs/DESIGN.md).
