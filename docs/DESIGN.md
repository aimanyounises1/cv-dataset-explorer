# Design Notes

The assignment intentionally leaves details open, so this document records what
I chose to build, what I deliberately left out, and why — closing with the
scale path, because a local choice is only defensible if you can say what would
replace it and where.

## Who is the user?

A computer vision researcher working with an image–caption dataset. Their core
jobs-to-be-done, mirrored by the four views:

1. **Get a feel for the data** → gallery browsing, statistics, embedding map.
2. **Find specific phenomena** ("dogs in water", "low-light street scenes") → semantic/keyword/hybrid search.
3. **Inspect and audit examples** → detail view with all captions and metadata, similar-image lookup (finds label inconsistencies and near-duplicates).
4. **Curate** → tagging (e.g. `edge-case`, `mislabeled`), tag filters, subset export for downstream training/eval pipelines.

## Key decisions

### SigLIP 2 embeddings, exact brute-force search
Text-to-image retrieval is the single most useful capability for this user, and
bi-encoder embeddings are the industry-standard first stage for it (this is what
FiftyOne's semantic search does under the hood). SigLIP 2 base was chosen over
original CLIP for strictly better zero-shot retrieval at similar cost; it runs
on MPS/CUDA/CPU.

At 8k × 768 floats (~24 MB), an exact numpy matmul answers a query in ~1 ms.
A vector database or ANN index would add operational and build complexity for
zero benefit below ~400k vectors — measured, see docs/TECHNICAL.md. The `EmbeddingIndex` class is the seam where
FAISS/sqlite-vec would slot in if the dataset grew.

### Hybrid search (RRF)
Embeddings miss exact-term queries ("Frisbee"), FTS misses paraphrases ("pet
catching a disc"). BM25 over captions costs nothing (SQLite FTS5) and
reciprocal-rank fusion is a robust, tuning-free way to combine ranked lists —
so hybrid is the default mode, with the pure modes exposed for transparency.

### Optional local VLM enrichment, not an agentic retrieval engine
A local VLM (Qwen 2.5-VL via Ollama) adds real value at *index time*: structured
tags beyond what the 5 captions mention, which become filterable facets and
extra keyword-search recall. Deliberately, the *retrieval engine itself* is not
agentic: per-query LLM calls are slow and nondeterministic, and at this scale
they add latency rather than recall. Enrichment is a strictly optional pass —
the tool must not require a 7B model to be useful.

### Caption QA: annotation auditing from embeddings we already have
Each caption is scored against its own image (CLIPScore-style SigLIP cosine) —
one matrix multiply over cached embeddings. Low agreement + high sibling mean
flags a likely bad caption; uniformly low flags an unusual image. Per-sample
caption consistency (mean pairwise caption similarity) surfaces ambiguous
images. This is the image-caption analog of label-error detection in
data-centric tooling (FiftyOne mistakenness, cleanlab), and it maps directly
onto edge-case-mining workflows: find where annotations and sensor data
disagree.

### Zero-shot attribute coverage: the long-tail lens
A small label bank (setting, time of day, environment, main subject) is
classified zero-shot via text-embedding dot products against existing image
embeddings — zero extra inference cost. The coverage dashboard makes class
imbalance and rare slices ("night": <2%) visible and clickable, which is
precisely the long-tail framing an AV data platform cares about. Adding a
label group is editing one dict.

### The tool benchmarks its own search
Flickr8k's captions are retrieval ground truth: querying with a caption should
return its own image (the standard published protocol for this dataset). The
Benchmark tab computes recall@1/5/10 for semantic, keyword, and hybrid modes —
so the choice of hybrid-by-default is *measured*, not asserted. The honest
caveat is stated in the UI: keyword recall is flattered because query captions
are literally indexed; semantic recall is the generalization signal.

### The assistant: Fugu-style orchestration as a *layer*, not the engine
The optional Assistant runs a LangGraph multi-agent graph inspired by Sakana's
Fugu (a multi-agent system behind a single interface): an **orchestrator**
classifies the request and dispatches a **retrieval specialist** (search,
similar, inspect, tag) or an **insights specialist** (stats, coverage, caption
QA); a **synthesizer** verifies the result against the request and either
finalizes it or sends it back for one corrective round. Two design rules keep
it honest: (1) agents call the *same service functions as the REST API* — the
deterministic search stack stays the platform's engine, agents are a
conversational client of it; (2) the whole stack is optional (separate
requirements file, availability probe, setup instructions in the UI) because a
review machine without Ollama must still experience a complete product. The
per-answer agent/tool trace is shown in the UI — transparency over magic.

### Graceful degradation as a design principle
Every ML-dependent feature detects its own availability: no embeddings →
keyword search with an explanatory banner, similarity/map/duplicates report
what to run to enable them. Reviewers can go from `git clone` to a working app
with `--skip-embeddings` in minutes, then opt into the heavier stages.

### Generic core, dataset-specific edges
The only Flickr8k-specific code is one adapter (`app/datasets/flickr8k.py`)
that yields `(image, split, captions)` samples. Ingestion, storage, search,
stats, and the entire frontend are dataset-agnostic; a new dataset means one
new adapter class registered in `app/datasets/__init__.py`.

### SQLite + files, no services
Images and thumbnails on disk, metadata/captions/tags in SQLite (WAL mode),
embeddings as `.npy`. Zero external services to install, trivially inspectable,
and honest about the actual scale of the problem.

### Precompute at ingestion, not at request time
Embeddings, UMAP projection, clusters, and thumbnails are computed once by an
idempotent ingestion CLI. Request-time work is lookups and one text-encoder
forward pass per semantic query, keeping the UI responsive.

## Deliberately not done (each is a choice, not an omission)

- **No vector database / ANN index** — exact numpy search is correct below
  ~400k vectors; ANN would add build time and recall loss for nothing. The
  upgrade path lives behind `EmbeddingIndex` (see "Scale path" below).
- **No repository/unit-of-work pattern over SQLite** — a thin `db.py` plus
  service functions is the right ceremony level here; a repository layer with
  one implementation is abstraction without benefit.
- **No task queue (Celery/Redis)** — heavy work is one-shot CLIs, not
  request-path jobs. A queue enters at the "orchestrated DAGs" stage of the
  scale path.
- **No client state library (Redux/Zustand)** — all state is either server
  state (fetched per view) or view state (in the URL). Adding a store would
  duplicate the URL as a second source of truth.
- **Sync (`def`) endpoints, not `async`** — model inference and numpy are
  blocking; sync handlers run on Starlette's threadpool and numpy/torch
  release the GIL. `async def` here would stall the event loop — the classic
  FastAPI trap. At real scale `async` is still not the fix; embedding leaves
  the API process entirely, which is where the scale path below starts.
- **No microservices/Docker split** — one backend, one frontend, one README;
  reviewers should be running it in minutes, not composing containers.

## Trade-offs accepted

- **UMAP is precomputed, static** — re-running ingestion refreshes it; fine for a read-mostly dataset.
- **Single-process API, in-memory index** — appropriate for a local tool; the index would move behind an interface boundary for multi-worker deployment.
- **No authentication** — local single-user tool by requirement.
- **Model download on first ingestion** (~1.5 GB from Hugging Face) — unavoidable for local inference; `--skip-embeddings` avoids it entirely.

## What I would build next

1. **Composed image retrieval**: query = image embedding + text delta ("this
   image but at night") with a weight slider — training-free with the cached
   embeddings; an active research area (ZS-CIR) that maps to scenario variation.
2. **Few-shot concept classifiers**: tag N positives → logistic regression on
   embeddings ranks the rest → iterate. The canonical mining-at-scale loop.
3. **Saved views / query history** so curation sessions are resumable.
4. **Streaming assistant responses** (LangGraph streams; the UI currently waits
   for the full orchestration round).
5. **Scale path**: swap the exact index for FAISS/sqlite-vec behind
   `EmbeddingIndex`, move ingestion/analysis to a job queue, virtualized grid
   rendering, multi-worker index sharing.

## Scale path

Nothing here is built for fleet scale, but every hard-coded local choice sits
behind the seam that would become a real service — which is the only claim about
scale a laptop project is entitled to make. The closest public reference
architecture for this problem is NVIDIA's
[Cosmos Dataset Search blueprint](https://github.com/NVIDIA-Omniverse-blueprints/cosmos-dataset-search)
(ingestion → GPU embedding service → Milvus vectors + Postgres metadata →
API/UI over object storage), and this project is deliberately that shape in
miniature.

### Seam map

| Local component            | Production component                                        |
|----------------------------|-------------------------------------------------------------|
| `datasets/` adapter        | Ingestion service consuming uploads from a message queue     |
| `Embedder` (in-process)    | GPU inference service (Triton/NIM, dynamic batching)         |
| `EmbeddingIndex` (numpy)   | Vector DB shards (Milvus/Qdrant), build/serve separated      |
| SQLite metadata            | Postgres + lakehouse tables (Parquet/Iceberg/Lance)          |
| `ingest.py` / `analyze.py` | Orchestrated batch DAGs (Airflow/Dagster) over the lake      |
| `POST /api/admin/reload`   | Index versioning + blue/green index swap with validation     |
| Benchmark tab (recall@k)   | Continuous retrieval-quality SLO on golden query sets        |
| Export manifest            | Versioned dataset slices with lineage (query → slice → run)  |
| LangGraph assistant        | Agentic analysis layer calling the deterministic search API  |

### What breaks first, and the fix at each stage

**~100k images.** Exact search is still ~ms; what breaks first is request-path
embedding and in-process index reload. Fix: dedicated GPU embedding service
with dynamic batching; ANN index (HNSW); atomic blue/green index swap behind
the existing reload seam.

**~10M.** Single-node memory and index rebuild time break. Fix: vector DB with
disaggregated build/serve nodes, quantization (PQ/int8); metadata to Postgres;
embeddings to object storage (Lance/Parquet); analysis becomes orchestrated
DAGs; recall@k and latency become monitored SLOs; embedding-drift monitoring
(distribution shift vs a reference set) joins the dashboard.

**~1B+ (fleet scale, hundreds of PB).** RAM-resident indexes stop making
economic sense. Fix: disk/object-storage-based sharded indexes (DiskANN-style,
tiered storage), embeddings at multiple granularities (clip / frame / object),
aggressive metadata pre-filtering to shrink the candidate space before vector
search, result caching, and first-class data versioning so any curated slice
is a reproducible training set. At this scale the product is no longer
"search" but a curation platform with lineage.

### Where the agents sit

At every scale, LLM/VLM agents (hypothesis generation, semantic query
expansion, failure triage) remain a *client* of the deterministic retrieval
API — never inside it. Retrieval stays testable, reproducible, and
benchmarkable; agents add reasoning on top. That separation is this repo's
assistant design, and it is also how production "AI data analyst" layers are
built over search infrastructure.

### References

NVIDIA Cosmos Dataset Search ([repo](https://github.com/NVIDIA-Omniverse-blueprints/cosmos-dataset-search),
[docs](https://docs.nvidia.com/cosmos/cds/latest/introduction.html)) ·
[Milvus architecture](https://milvus.io/blog/deep-dive-1-milvus-architecture-overview.md) ·
[Lance format](https://github.com/lance-format/lance) ·
[Triton serving](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html) ·
[Hybrid search + rerank](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/) ·
[Embedding drift monitoring](https://www.evidentlyai.com/blog/embedding-drift-detection)
