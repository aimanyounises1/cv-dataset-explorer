# Deploying with Docker

The README's direct workflow — `uvicorn` on the host, `vite` beside it — is the
reference way to run this project and is unchanged. This document covers the
containerised alternative: the same application with its Python and Node
dependencies pinned into images, for a machine where installing them directly
is not wanted.

Nothing here is required to evaluate the project. The two paths share the same
`backend/data` directory and the same ports, so they are alternatives rather
than parallel installs.

## Layout

| File | Role |
| --- | --- |
| `docker-compose.yml` | The stack: `backend`, `frontend`, optional `ollama` |
| `docker/backend.Dockerfile` | FastAPI + uvicorn on `python:3.13-slim` |
| `docker/frontend.Dockerfile` | `npm run build` on `node:22`, served by `nginx:alpine` |
| `docker/nginx.conf` | Static SPA + reverse proxy for `/api` and `/media` |
| `docker/*.Dockerfile.dockerignore` | Per-image build contexts |

Both images build from the repository root as their context. BuildKit reads the
ignore file named after the Dockerfile in preference to the context root's,
which is what lets two images share one context and still exclude different
things — without it the context would include the 2.0 GB of gitignored data
under `backend/data`. Measured contexts are 692 kB (backend) and 631 kB
(frontend).

## Running it

```bash
docker compose up --build
```

The UI is then on <http://localhost:5173> and the API on
<http://localhost:8000>. Those are the same ports the host workflow uses, on
purpose: every URL in the README and in `docs/` addresses the containerised
stack without modification, and so does the host QA sweep, whose
`CVDE_QA_BASE_URL` and `CVDE_QA_API_URL` defaults already point at them.

`frontend` waits for `backend` to report healthy before it starts, so the first
page load never races the API's schema initialisation.

To stop: `docker compose down`. State survives — it is in `backend/data` on the
host, not in the containers.

### First run: ingest

A fresh clone has no corpus, and the containers do not create one. The image
ships the same `app.ingest` module the host workflow uses:

```bash
docker compose up -d backend
docker compose exec backend python -m app.ingest
```

That downloads Flickr8k, writes images, thumbnails, `explorer.db` and the
SigLIP 2 embeddings into `/data` — which is `backend/data` on the host. It is
idempotent in the sense that matters: rerunning never duplicates a sample.
Expect tens of minutes on CPU; `--limit 200` gives a working subset quickly,
and `--skip-embeddings` gives browse and keyword search without the encode pass.

If `backend/data` is already populated from a host run, skip this entirely. The
container reads the corpus the host workflow built, and the environment
fingerprint the app records hashes the embedding files, so a mismatch between
what was ingested and what is being served is visible in the UI rather than
silent.

### With Ollama

The assistant and VLM enrichment need Ollama. Both modes work:

**Host Ollama** (the default, and the usual developer setup — the models are
already pulled):

```bash
docker compose up --build
```

`CVDE_OLLAMA_URL` defaults to `http://host.docker.internal:11434`. The compose
file maps that name to the host gateway explicitly, so it also resolves on
Linux, where Docker does not provide it by default.

**Containerised Ollama** (nothing installed on the host):

```bash
CVDE_OLLAMA_URL=http://ollama:11434 docker compose --profile ollama up --build
docker compose exec ollama ollama pull qwen3:8b       # assistant, needs tool calling
docker compose exec ollama ollama pull qwen2.5vl:7b   # optional VLM enrichment
```

The profile keeps this off by default: a second Ollama would re-download
several GB of weights that most machines running this already have.

Without either, the assistant degrades with a message naming the command that
would enable it. That is the project's documented contract for every optional
capability, and it holds in containers too — a missing Ollama does not fail the
stack.

## Health

| Endpoint | Answers |
| --- | --- |
| `GET /api/tags` | Is uvicorn serving and did the database open? |
| `GET /api/health` | ... plus corpus size and whether semantic search has an index |

The container `HEALTHCHECK` and compose's `service_healthy` gate both use
`/api/tags`, deliberately: it is a single indexed SQLite read, where
`/api/health` calls `get_index()` and pulls the embedding matrix into memory.
A health probe running every 30 s should not be the thing that decides when the
index is first read off disk. Use `/api/health` when you want the richer answer:

```bash
curl -s http://localhost:8000/api/health
# {"status":"ok","samples":8000,"semantic_search":true}
```

`docker compose ps` shows the tracked state.

## Volumes

| Mount | Contents | Why |
| --- | --- | --- |
| `./backend/data` → `/data` | `explorer.db`, images, thumbs, embeddings, caches, QA artifacts, reports | Bound to the host directory so one ingest serves both workflows |
| `hf-cache` → `/opt/hf` | Hugging Face model weights (`HF_HOME`) | Named volume: a ~1.5 GB download that should survive `down` without landing in the corpus directory |
| `ollama-models` → `/root/.ollama` | Pulled Ollama models (profile only) | Belongs to Ollama, not to this repository |

Model weights are kept out of `backend/data` on purpose. That directory is the
corpus, and the environment fingerprint is computed over the embedding files
in it; mixing a model cache into it muddles what "the data" means.

## Resource expectations

Measured on this machine where marked; otherwise labelled an estimate with its
basis. Docker Desktop's VM here has 8.2 GB of RAM and 16 CPUs, and the
measurements are `linux/arm64` on Apple Silicon.

**Images**

| Image | Size |
| --- | --- |
| `cvde-backend` | 2.52 GB measured (torch and transformers dominate) |
| `cvde-frontend` | 102 MB measured |

**Memory**

| Configuration | Expect | Basis |
| --- | --- | --- |
| Backend, booted, no model loaded | 77 MB | Measured, container idle |
| Backend, SigLIP 2 loaded on CPU | ~1.0 GB | Measured, container peak during an encode |
| Backend serving the full corpus | ~1.05 GB | Sum of two measurements, not one: the ~1.0 GB above plus the 24.6 MB image matrix — `get_index()` loads `image_embeddings.npy` alone, not the 612 MB the embeddings *directory* occupies on disk. The host process doing the same job measures 1.05 GB RSS, but on MPS, where the weights sit in shared GPU memory |
| Backend, optional `qwen3_vl` provider | +~4-5 GB | Estimate from the ~4 GB weight download plus activations; not measured |
| Frontend (nginx + static files) | 20 MB | Measured, serving 868 kB of built assets |
| Ollama container, `qwen3:8b` loaded | ~6 GB | Estimate from the model's published size; not measured |

The 8.2 GB VM fits the default stack comfortably. Adding the containerised
Ollama profile *and* the Qwen provider at once would not fit — raise Docker
Desktop's memory allocation, or keep Ollama on the host.

**Disk**

| Item | Size |
| --- | --- |
| Corpus in `backend/data` | 1.4 GB measured: images 593 MB, embeddings 612 MB, thumbs 182 MB, `explorer.db` 13 MB |
| Hugging Face cache (SigLIP 2) | 1.4 GB measured on the host cache |
| QA sweep artifacts (only if you run sweeps) | 1.1 GB on this install; disposable, which is why `backend/data` totals 2.5 GB here rather than 1.4 GB |
| Optional Qwen3-VL weights | ~4 GB; estimate, not installed here |

**Latency: containers are CPU-only.** There is no MPS inside a Linux container
and no CUDA on this host, so every forward pass runs on CPU, where the host
workflow uses Apple Silicon's GPU through MPS. The embedder reports
`device: cpu` in-container, and encoding one text query measures:

| n | min | median | p90 | max |
| --- | --- | --- | --- | --- |
| 15 | 82.6 ms | **84.9 ms** | 85.9 ms | 90.8 ms |

That is one forward pass through SigLIP 2's text tower, timed through the
application's own `Embedder.encode_texts` rather than a synthetic harness, after
a discarded warm-up pass.

This is a property of the deployment target, not a defect. Nearest-neighbour
search is unaffected — it is exact NumPy cosine over a resident matrix, no model
involved. Ingest is the pass that feels CPU-only most, because it encodes 8,000
images rather than one query.

The backend image installs torch from the `+cpu` wheel index on every
architecture. PyPI's default wheel is the CUDA build on `linux/arm64` as well
as `linux/amd64` — an earlier revision of this image assumed otherwise and shipped
`torch 2.13.0+cu130` reporting `torch.cuda.is_available() == False`, alongside
2.9 GB of NVIDIA libraries and 650 MB of triton that no container without GPU
passthrough can load. Pinning the `+cpu` index is what keeps the image at its
current size.

## What was verified

Both images build and the stack runs. Checked on `linux/arm64`, Docker 29.3.1,
against a throwaway data directory so the corpus and the host's dev servers
were untouched:

- `docker compose config -q` passes.
- Both images build from a cold cache; contexts transfer at 692 kB and 631 kB.
- `docker compose up` brings both containers to `(healthy)`, and the frontend
  starts only after the backend is healthy — the `depends_on` gate works.
  Both probes address `127.0.0.1` explicitly. An earlier revision used
  `localhost` and left the frontend permanently `unhealthy` while it served
  every request correctly: nginx listens on `0.0.0.0:80`, and busybox `wget`
  resolves `localhost` to `::1` without falling back to IPv4. `curl`'s fallback
  hides the same mismatch on the backend probe.
- `/api/tags` and `/api/health` answer, directly and through the nginx proxy.
- A client-side route (`/compare`) returns the SPA, while an unknown `/api`
  path returns the API's own JSON 404 rather than a page of HTML.
- `/media` proxies to the backend rather than falling through to the SPA.
- gzip is applied: the main bundle transfers 92 kB against 243 kB raw.
- The degradation contract holds in-container. With no embeddings present,
  `GET /api/search?q=dog&mode=semantic` returns `degraded: true`,
  `mode_used: "keyword"`, and a message naming what is missing.

**The test suite is not green inside this image**, and the reason is worth
stating rather than hiding: `docker compose exec backend python -m pytest -q`
gave **290 passed, 8 failed, 2 skipped** when this was measured — a 300-test
collection, against 383 on the host today, so re-run the command rather than
quote the tally. All 8 failures were in `tests/test_providers.py`, and all 8
passed on the host.

The application is correct here; the tests carry an environment assumption.
The image deliberately omits `requirements-qwen.txt`, so `providers.resolve()`
short-circuits at its outermost check and reports
`qwen3_vl: provider stack not installed — <install command>`. That is exactly
the documented degradation contract. Each failing test monkeypatches a *deeper*
failure — missing weights, an unbuilt index, a model mismatch — and asserts the
reason names that specific cause, which is only reachable when
`sentence-transformers` is importable. The host has it (5.6.1), so the
assumption is invisible there. Making these tests pin their own outer condition
would close the gap; that is a change to `tests/`, not to the deployment.

## What is not containerised

Named explicitly, because an unstated omission reads as an oversight:

- **The UI smoke sweep.** `scripts/ui_smoke.py` drives real Chrome via
  Playwright and runs on the host, against whichever stack is serving 5173 —
  including this one. Putting a browser in the image would add hundreds of MB
  to serve a development workflow, and the flows are defined once in
  `backend/app/qa/flows.py` regardless of who runs them.
- **The optional Qwen retrieval provider.** `requirements-qwen.txt` is not
  installed; the image would grow by the torch/sentence-transformers delta plus
  a 4 GB weight download for a capability that is opt-in by design. To use it,
  extend the image with that requirements file and set
  `CVDE_EMBED_PROVIDER=qwen3_vl`. Unset, retrieval falls back to SigLIP 2 and
  the UI names the reason.
- **Dataset download.** Ingest is a deliberate, explicit first step, not
  something a container does on boot. A stack that silently pulls a gigabyte on
  first start is worse than one that tells you to ask for it.
- **HTTPS, authentication, and multi-user concerns.** This is a local analysis
  tool with a SQLite database and no auth model. The compose file publishes to
  localhost.

## Security posture

The backend container runs as root. This is a considered tradeoff, not an
oversight: it writes continuously to a bind-mounted host directory, and a fixed
non-root UID inside the container would not match the host's file ownership on
an arbitrary machine — the common outcome is a container that cannot open its
own database. The stack publishes to localhost and is a local tool.

For a hardened deployment, run as the UID that owns the data directory:

```yaml
services:
  backend:
    user: "1000:1000"
```

The UID must match the one that owns the data directory — `stat -c '%u:%g'
backend/data` on Linux, `stat -f '%u:%g' backend/data` on macOS — or `chown`
the directory to match before starting.
