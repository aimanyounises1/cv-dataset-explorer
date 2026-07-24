# Architecture — Local Design & Production Scale Path

This document covers (1) how the system is layered today, and (2) how each
piece maps to a production-scale platform — the point being that every
hard-coded local choice sits behind the seam that would become a real service.

## Local architecture

```
┌─────────────────────────── React + TS (Vite) ────────────────────────────┐
│  Gallery · Sample · Map · Stats · Quality · Benchmark · Assistant        │
│  (view state in URL · /api and /media proxied to backend)                │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │ REST
┌──────────────────────────────────▼───────────────────────────────────────┐
│ FastAPI                                                                  │
│  api/: samples search stats map tags qa attributes eval admin chat       │
│  agent/: LangGraph orchestrator → {retrieval, insights} → synthesizer    │
│          (agents call the same service functions as the REST routes)     │
│  ml/:  Embedder (SigLIP 2, MPS/CUDA/CPU) · EmbeddingIndex (exact numpy)  │
│        projection (UMAP/KMeans) · label bank (zero-shot attributes)      │
└───────┬──────────────────────┬───────────────────────┬───────────────────┘
        │                      │                       │
   SQLite (WAL)          .npy embeddings          images/ thumbs/
   samples · captions    (image + caption)        on local disk
   FTS5(porter) · tags
   attributes · QA scores
        ▲
        │  one-time / re-runnable batch CLIs
   ingest.py (download → store → index → embed → project)
   analyze.py (caption QA · zero-shot attributes)
   enrich.py  (optional VLM tagging via Ollama)
```

Key properties: request path does lookups plus at most one text-encoder
forward pass; everything heavy is precomputed by idempotent batch CLIs;
every ML-dependent feature detects its own availability and degrades.

Endpoints are deliberately sync `def` (FastAPI runs them on a threadpool;
numpy/torch release the GIL) — an `async def` endpoint calling blocking model
inference would stall the event loop. At real scale, embedding moves out of
the API process entirely (below), which is the honest fix.

## Production scale path

The closest public reference architecture for this problem is NVIDIA's
[Cosmos Dataset Search blueprint](https://github.com/NVIDIA-Omniverse-blueprints/cosmos-dataset-search)
(ingestion → GPU embedding service → Milvus vectors + Postgres metadata →
API/UI over object storage). This project is intentionally that shape in
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
