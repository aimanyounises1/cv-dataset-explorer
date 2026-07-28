# CV Dataset Explorer

**A local workspace for finding, auditing and curating slices of an
image–caption dataset — the 8,000 images and 40,000 human captions of
[Flickr8k](https://huggingface.co/datasets/jxie/flickr8k) — where every
ranking, score and measurement is labelled with what produced it.**

Working with a dataset means answering questions a file browser cannot: where
are the night scenes, which captions their own image does not support, which
300 samples are hardest, whether the held-out split is contaminated by
near-duplicates of training images. The through-line is **long-tail
discovery** — rare scenes, coverage gaps, and the annotations that don't hold
up are what the rarity axis, the coverage dashboard, the similarity floor and
the caption audit exist to surface. Everything runs on one machine: SQLite for
storage, local embedding models for retrieval, no cloud services, no paid
APIs, no external vector database.

## Quick start

```bash
# backend (Python 3.11+)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.ingest            # downloads Flickr8k + SigLIP 2, builds everything (~20 min)
uvicorn app.main:app --port 8000

# frontend (Node 20+), second terminal
cd frontend && npm install && npx vite --port 5173
```

Open http://localhost:5173. `python -m app.ingest --skip-embeddings` skips the
model download and runs keyword-only; every ML capability is an optional layer
that reports its own availability and names the command that enables it.

Optional layers, each honest about its absence:

| Layer | Enable with |
|---|---|
| Assistant (local agents) | [Ollama](https://ollama.com) + `ollama pull qwen3:8b`, then `pip install -r requirements-agent.txt` |
| VLM tag enrichment | `ollama pull qwen2.5vl:7b`, then `python -m app.enrich` |
| Self-QA browser sweep | `pip install -r requirements-qa.txt` (Playwright + real Chrome) |
| Qwen3-VL retrieval provider | `pip install -r requirements-qwen.txt`, then `python -m app.ingest --provider qwen3_vl` |
| Container path | `docker compose up --build` — see [docs/DEPLOY.md](docs/DEPLOY.md) |

## The model decision, measured

Retrieval defaults to **SigLIP 2** because it measures better on this corpus's
own benchmark — not because it is smaller: parameter count is not quality.
Qwen3-VL-Embedding-2B runs behind the same provider seam as an explicit
opt-in (`CVDE_EMBED_PROVIDER=qwen3_vl`); each provider keeps its own index
directory and manifest, so two embedding spaces can never mix, and the UI
names whichever provider is actually ranking, with the reason whenever a
fallback happened.

Head-to-head on this machine (M4 Max, 64 GB; 1,000-query text→image sample of
the repository's own protocol; `scripts/bench_providers.py` re-measures it):

| provider | dim | text encode p50 | image encode p50 | semantic R@1/5/10 | hybrid R@1/5/10 | index size |
|---|---|---|---|---|---|---|
| SigLIP 2 base (default) | 768 | 7.7 ms | 14.4 ms | **55.2 / 79.0 / 86.4%** | **56.0 / 79.1 / 86.3%** | 197 MB |
| Qwen3-VL-Embedding-2B (opt-in) | 2048 | 39.6 ms | 285.6 ms | 50.2 / 75.0 / 83.4% | 51.1 / 75.3 / 83.9% | 394 MB |

Ollama serves the language models only (assistant `qwen3:8b`, enrichment
`qwen2.5vl:7b`) — image embeddings are computed in-process. The map, difficulty
axes and caption agreement are ingest-time, SigLIP-derived artifacts and say
so; a provider switch changes retrieval, never rewrites stored analysis.

## What a researcher can do

### Search, three ways — and steer it

![Hybrid search for "a crowded street at night": ranked grid, per-card scores
with their basis, term statistics](docs/screenshots/1-gallery.png)

Semantic, keyword (FTS5 BM25) and hybrid (reciprocal-rank fusion) — one
ranking implementation serves the gallery, the export buttons and the agents,
so they can never disagree. Every score names its basis (`cosine`,
`cosine_adj`, `rrf`, `composed`); scores from different bases are never
compared. Search by image (drop, paste or pick a file), or steer with
reference chips: *More like this* / *Exclude* build a composed query in the
URL (`?like=76&unlike=13` plus text), so a colleague can open the same
steered search from a pasted link — and the Back button walks the trail.

![Zero-shot facet filters with a set description panel open](docs/screenshots/2-describe.png)

Facet filters come from zero-shot attribute classification over the existing
embeddings; the description panel summarizes any filtered set with counted,
never generated, statements.

### Judge difficulty, not just content

![Gallery filtered to the hardest, least legible samples](docs/screenshots/4-axes.png)

Four 0–10 axes — legibility, rarity, difficulty, clutter — computed as
percentile ranks over the corpus, each a filter, a badge and a sort key. "The
300 hardest validation samples" is a URL.

### See the corpus shape

![UMAP embedding map with cluster colors and lasso selection](docs/screenshots/5-map.png)

The embedding map projects the corpus (UMAP, ingest-time); lasso a region to
tag or open it as a gallery slice. Isolated points are coverage-gap
candidates.

### Audit the annotations

![Caption quality view: least-supported captions first](docs/screenshots/6-quality.png)

CLIPScore-style agreement between each caption and its own image, worst
first — likely annotation errors surface immediately. Per-sample caption
consistency flags images whose five captions disagree with each other.

### Trust, measured

![Dataset profile with provenance caveats and split statistics](docs/screenshots/7-stats.png)
![The self-benchmark table: recall@k for all three modes](docs/screenshots/8-eval.png)

The stats page states the dataset's provenance and known defects. The
benchmark page measures the tool's own retrieval (text→image recall@1/5/10
per mode, the dataset's captions as ground truth) with its protocol and
caveats stated inline; `GET /api/stats/leakage` reports train/test
near-duplicate contamination as a threshold ladder rather than one arbitrary
number.

### Inspect one sample deeply

![Sample inspector: image, five captions with agreement, attributes, neighbours](docs/screenshots/9-sample.png)
![Provenance banner: surfaced by hybrid search, rank 1](docs/screenshots/12-provenance.png)
![The similarity floor: two real neighbours above the divider, greyed context below](docs/screenshots/13-floor.png)

Every caption carries its agreement score. Neighbours below the corpus-derived
similarity floor (the measured 10th percentile of nearest-neighbour cosine in
the active index; `?min_sim=` overrides) grey out rather than posing as a
class — and when nothing clears it, the page says "possible coverage gap"
instead of padding. A sample reached from a search says *why* it surfaced.

### Compare two frames

![Compare canvas: synchronized zoom, shared/different panel](docs/screenshots/14-compare.png)

`/compare` puts two samples under one loupe — synchronized zoom and pan, a
counted shared/different panel, and manual rectangle regions (drawn by you, no
segmentation model) that can be saved as annotations or cropped into an
image search. Source images are never modified.

### Curate into albums, and let them leave

![An album's header: editable details and the measured analysis panel](docs/screenshots/15-album.png)
![The Share menu: copy link, email, Teams compose, downloads — everything local](docs/screenshots/16-share.png)

Albums are first-class ordered collections (provenance `manual` | `tag` —
converting a tag is explicit and keeps the tag). Drag cards onto the shelf —
every drop offers an Undo that removes exactly what the drop added — reorder
by drag or arrow keys, set a cover, edit the summary where you read it. The
Analyze panel counts what members share and where they split, and computes
coherence and outliers from the active index; a summary draft from the local
chat model is generated only on request, edited by you, saved only by you.
Share stays local-first: copy the URL, compose an email or Teams message
(nothing is uploaded), or download the slice as JSON/CSV/JSONL with a
manifest that records the query, model and embedding fingerprint that
produced it.

### Ask an assistant that shows its work

![Assistant conversation with agent trace and rendered blocks](docs/screenshots/11-assistant.png)

An optional LangGraph orchestration over local Ollama: an orchestrator routes
to retrieval/insights specialists, a synthesizer quality-gates the answer, and
the per-answer agent/tool trace is shown. Agents call the same service
functions as the REST API — retrieval stays deterministic. The assistant's one
mutation is a **proposal**: tagging waits for your click, and conversations
persist locally with rename/reopen/delete.

![Command palette over every route and action](docs/screenshots/10-palette.png)
![The front door, with the rail's local-models card](docs/screenshots/17-models.png)

⌘K reaches everything; the rail's models card says which retrieval provider,
assistant and enrichment model are real, live and local right now — with the
named reason when one is missing.

### Let other agents in, read-only

A local [MCP server](docs/MCP.md) at `POST /mcp` exposes six strictly
read-only tools (search, sample inspection, similar images, dataset stats,
caption audit, album inspection) over the same service layer — any local MCP
client, LangGraph included, can investigate the corpus and cannot curate it.

## Architecture

One FastAPI process, one SQLite file (WAL), embeddings as `.npy`, images on
disk; a React 18 + TypeScript frontend whose search and filter state lives in
the URL (an investigation is a link). Heavy work happens in idempotent batch
CLIs (`ingest`, `analyze`, `enrich`); a request does SQLite lookups plus at
most one text-encoder forward pass. Retrieval is an exact NumPy cosine scan —
measured at ~0.2 ms against the 7–8 ms encode it waits behind, with the ANN
crossover (~400k vectors) and the seam it would slot into documented rather
than built. Optional capabilities degrade with a named reason, never a 500.

The deeper documents: [ARCHITECTURE](docs/ARCHITECTURE.md) (topology, seams,
scale path) · [TECHNICAL](docs/TECHNICAL.md) (schema, query plans,
measurements) · [DESIGN](docs/DESIGN.md) (decisions and trade-offs) ·
[TESTING](docs/TESTING.md) · [AGENTS](docs/AGENTS.md) ·
[CAPABILITIES](docs/CAPABILITIES.md) (generated; CI fails if it drifts) ·
[MCP](docs/MCP.md) · [DEPLOY](docs/DEPLOY.md) · [DEMO](docs/DEMO.md).

```
frontend/src/
  pages/          Gallery · Sample · Compare · Map · Stats · Quality · Benchmark · Assistant
  components/     rail, cards, album shelf/header, share menu, render blocks
  api/            typed client — the only fetch layer
backend/app/
  api/            17 routers over one service layer (run_search is the one ranking impl)
  ml/             providers (SigLIP 2 / Qwen3-VL) · exact index · hubness correction
  agent/          LangGraph graph, tools, render-block contract
  qa/             flow registry + real-Chrome runner (one definition, three consumers)
  ingest/analyze/enrich    idempotent batch CLIs
docker/ + docker-compose.yml    the container path
scripts/        benchmarks, screenshots, capabilities/link checks
```

## Test evidence

```bash
cd backend && pytest                  # 340 passed (2026-07-28)
cd frontend && npx tsc --noEmit && npm run build
cd backend && python ../scripts/ui_smoke.py   # real-Chrome sweep, 17 workflows
```

The last full sweep at this commit: **104/104 checks across 17/17 workflows**
(run id in `backend/data/qa/`). Re-run rather than quote — the registry grows,
and every number above is only true of the day it was measured. CI runs the
light install (no torch/langgraph — those modules skip), ruff, tsc, the build,
the link check and the capabilities contract.

## Honest limitations

- Qwen3-VL underperforms SigLIP 2 on this benchmark (table above); the hubness
  constants were tuned in the SigLIP space and applied untuned in Qwen's —
  whether tuning closes any of the gap is unmeasured.
- The composed (reference-steered) ranking is unmeasured — the benchmark
  covers the text modes only; its blend weight is chosen, not tuned, and the
  code says so.
- The keyword benchmark row is flattered by construction (query captions are
  indexed); the benchmark page states this where the number appears.
- The assistant needs Ollama and a ~5 GB model; agent answers arrive after the
  full orchestration round rather than streaming.
- MCP is stateless JSON (no SSE streaming, no sessions); it binds to the same
  local server and adds no authentication — a local, single-user tool by the
  assignment's constraint.
- In-container inference is CPU-only and slower than the host path
  ([docs/DEPLOY.md](docs/DEPLOY.md) states expectations).
- The dataset copy itself is imperfect and the stats page says how (missing
  ~90 images vs the original distribution, undocumented split assignments, no
  upstream license).

## Configuration

Everything is environment variables; [.env.example](.env.example) lists each
one with its default and effect. The ones that matter first:
`CVDE_DATA_DIR`, `CVDE_EMBED_PROVIDER` (`siglip2` default | `qwen3_vl`),
`CVDE_EMBED_MODEL`, `CVDE_OLLAMA_URL`, `CVDE_CHAT_MODEL`, `CVDE_VLM_MODEL`.
