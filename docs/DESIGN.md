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
FiftyOne's semantic search does under the hood). SigLIP 2 base was chosen on its
published zero-shot retrieval results and its size; no CLIP baseline was ever
run here, so this repo cannot claim a margin over CLIP. The one head-to-head it
*did* measure is SigLIP 2 against Qwen3-VL — R@1 55.2 vs 50.2 at ~5x the encode
speed (`scripts/bench_providers.py`, n=1000) — which is why SigLIP 2 is the
default. It runs on MPS/CUDA/CPU.

At 8k × 768 floats (24.6 MB), an exact numpy matmul answers a query in 0.18 ms —
against the 7–8 ms SigLIP text encode it waits behind, so the scan is not what
costs anything at this size (docs/TECHNICAL.md). A vector database or ANN index
would add operational and build complexity for no benefit anyone here has been
able to measure. The ~400k crossover quoted elsewhere is arithmetic on a scan
trend, an *estimated* crossover rather than permission to adopt: it says nothing
about recall, and a real FAISS recall/latency benchmark would have to come first.
The `EmbeddingIndex` class is the seam where FAISS/sqlite-vec would slot in if
the dataset grew.

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

### Zero-shot prompt slices: a long-tail review aid
A small prompt bank (setting, time of day, environment, main subject) is scored
via text-embedding dot products against existing image embeddings — zero extra
request-time inference. A margin gate abstains when the top two prompts are too
close. The dashboard makes the resulting distributions and small candidate
slices visible and clickable, but presents them as hypotheses to inspect, not
ground-truth classes or calibrated accuracy estimates. Adding a prompt group is
editing one dict and re-running the analysis.

### The tool benchmarks its own search
Flickr8k's captions are retrieval ground truth: querying with a caption should
return its own image (the standard published protocol for this dataset). The
Benchmark tab computes recall@1/5/10 for semantic, keyword, and hybrid modes —
so the choice of hybrid-by-default is *measured*, not asserted. The honest
caveat is stated in the UI, and it runs the other way: the query caption's own
row is excluded from the lexical scan, so keyword recall is *depressed* by a
strict conjunction that returns an empty candidate list for most queries, not
flattered by self-retrieval. Semantic recall is the generalization signal.

### The assistant: Fugu-style orchestration as a *layer*, not the engine
The optional Assistant runs a LangGraph multi-agent graph inspired by Sakana's
Fugu (a multi-agent system behind a single interface). Its registry currently
declares four lanes: **retrieval** (search, similar, inspect, tag proposals),
**insights** (statistics, coverage, caption QA), **visualization** (charts and
reports), and an expensive **QA** lane (the real-browser sweep). A
schema-constrained orchestrator selects from that registry, LangGraph fans out
to at most two cheap lanes, and a typed synthesizer either finalizes the result
or returns one bounded corrective instruction. Two design rules keep it honest:
(1) retrieval calls the same `run_search` service as the REST API, while
inspection tools are read-only SQL over the same store — the deterministic
search stack stays the platform's engine, and dataset writes remain behind the
human-operated REST UI; (2) the whole stack is optional (separate requirements
file, availability probe, setup instructions in the UI) because a review
machine without Ollama must still experience a complete product. The per-answer
agent/tool trace is shown in the UI — transparency over magic.

### Graceful degradation as a design principle
Every ML-dependent feature detects its own availability: no embeddings →
keyword search with an explanatory banner, similarity/map/duplicates report
what to run to enable them. Reviewers can go from `git clone` to a working app
with `--skip-embeddings` in minutes, then opt into the heavier stages.

### Reusable ingestion core, explicit dataset-specific analysis
`app/datasets/flickr8k.py` is the boundary that turns Flickr8k into
`(image, split, captions)` samples, and ingestion, storage, thumbnails and the
basic retrieval path reuse that contract. The finished product is intentionally
specialized beyond that boundary: the benchmark assumes image-caption
retrieval, several analyses assume caption sets and canonical splits, the
zero-shot prompt bank is chosen for this corpus, and the UI names Flickr8k.
Supporting a different image-caption dataset therefore starts with one adapter
but also requires validating its split semantics, prompt bank, analysis
calibration, benchmark protocol and dataset-facing copy. That work is explicit;
the repository does not claim a one-file port of the whole application.

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
- **Sync handlers for blocking work; async only at I/O edges** — model
  inference, NumPy and SQLite handlers run on Starlette's threadpool, while
  streaming responses, uploads and protocol adapters use `async def` where
  they await I/O. Putting blocking inference in an async handler would stall
  the event loop. At real scale `async` is still not the inference fix;
  embedding leaves the API process entirely, which is where the scale path
  below starts.
- **No microservice topology** — the optional Compose path packages the same
  FastAPI process and built React frontend; it does not introduce queues,
  service-owned databases or another retrieval implementation. The direct host
  workflow remains the shortest evaluation path.

## Trade-offs accepted

- **UMAP is precomputed, static** — re-running ingestion refreshes it; fine for a read-mostly dataset.
- **Single-process API, in-memory index** — appropriate for a local tool; the index would move behind an interface boundary for multi-worker deployment.
- **No authentication** — local single-user tool by requirement.
- **Model download on first ingestion** (~1.5 GB from Hugging Face) — unavoidable for local inference; `--skip-embeddings` avoids it entirely.

## What I would build next

Three earlier entries on this list shipped since it was written: composed image
retrieval (reference chips + steering text + negative examples, basis
`composed`), saved views, and the history view over the activity log — the
History drawer in the left rail lists the trail and reopens a recorded view
from one click. What remains, in order:

1. **Tuned composition weights**: the composed query's negative-example weight
   is a chosen constant (0.5), stated as untuned in the code; a weight slider
   plus a small ZS-CIR-style evaluation would turn a choice into a measurement.
2. **Few-shot concept classifiers**: tag N positives → logistic regression on
   embeddings ranks the rest → iterate. The canonical mining-at-scale loop.
3. **Streaming the assistant's *answer***: the node transitions already stream
   (`POST /api/chat/stream`, rendered live), but the reply text and the render
   blocks arrive whole at the end of the run. Emitting each block as its lane
   finishes is the version worth building.
4. **A review lifecycle over the annotations that now persist.** The sample
   editor can propose a Grounding DINO box, refine it with SAM2 points/boxes,
   accept a mask with model/prompt/IoU provenance, search from it, list it and
   delete it. The remaining production step is review workflow rather than
   segmentation plumbing:
   * **status and review identity** — proposed/approved/rejected plus who made
     that decision;
   * **an update endpoint** — "nudge that mask and keep it" should preserve the
     record's history rather than delete and recreate it;
   * **proposals across a set** — run the detector over an album and review the
     batch instead of invoking it one image at a time.
   Agents remain read-only: they can inspect masks and search from an accepted
   annotation, but a human owns every curation write.
5. **Scale path**: benchmark FAISS behind `EmbeddingIndex` once the corpus is
   near 100k vectors, and adopt it only when the measured recall/latency trade
   beats exact search for the intended workload. The extrapolated ~400k
   crossover is where that measurement becomes urgent, not a predetermined
   cutover. Then move ingestion/analysis to a job queue, virtualize grid
   rendering and share indexes across workers. pgvector enters only if the
   deployment becomes multi-user and server-side, which is a different product.

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
