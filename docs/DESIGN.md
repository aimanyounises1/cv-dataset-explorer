# Design Notes

The assignment intentionally leaves details open, so this document records what
I chose to build, what I deliberately left out, and why.

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
zero benefit below ~100k vectors. The `EmbeddingIndex` class is the seam where
FAISS/sqlite-vec would slot in if the dataset grew.

### Hybrid search (RRF)
Embeddings miss exact-term queries ("Frisbee"), FTS misses paraphrases ("pet
catching a disc"). BM25 over captions costs nothing (SQLite FTS5) and
reciprocal-rank fusion is a robust, tuning-free way to combine ranked lists —
so hybrid is the default mode, with the pure modes exposed for transparency.

### Optional local VLM enrichment, not an agentic pipeline
A local VLM (Qwen 2.5-VL via Ollama) adds real value at *index time*: structured
tags beyond what the 5 captions mention, which become filterable facets and
extra keyword-search recall. I deliberately did **not** build an agentic
(LLM-in-the-query-loop) search: per-query VLM calls are slow and
nondeterministic, and at this dataset scale they add latency rather than
recall. Enrichment is a strictly optional pass — the tool must not require a
7B model to be useful.

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

## Trade-offs accepted

- **UMAP is precomputed, static** — re-running ingestion refreshes it; fine for a read-mostly dataset.
- **Single-process API, in-memory index** — appropriate for a local tool; the index would move behind an interface boundary for multi-worker deployment.
- **No authentication** — local single-user tool by requirement.
- **Model download on first ingestion** (~1.5 GB from Hugging Face) — unavoidable for local inference; `--skip-embeddings` avoids it entirely.

## What I would build next

1. **Model-assisted auditing**: caption–image agreement scoring (SigLIP similarity of each caption to its own image) to surface likely annotation errors — the highest-value next feature for dataset QA.
2. **Saved views / query history** so curation sessions are resumable.
3. **Lasso selection on the embedding map** → bulk-tag a visual cluster.
4. **Text-side analysis**: caption embedding map, per-cluster vocabulary.
5. **Scale path**: swap the exact index for FAISS/sqlite-vec behind `EmbeddingIndex`, move ingestion to a job queue, virtualized grid rendering.
