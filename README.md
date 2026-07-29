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
up are what the rarity axis, prompt-slice dashboard, similarity floor and
caption audit exist to surface. Everything runs on one machine: SQLite for
storage, local embedding models for retrieval, no cloud services, no paid
APIs, no external vector database.

## Quick start

```bash
# backend (Python 3.11+), from the repository root
cd backend
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m app.ingest           # downloads Flickr8k + SigLIP 2, builds everything (~20 min)
python3 -m uvicorn app.main:app --port 8000

# frontend (Node 20+), second terminal from the repository root
cd frontend && npm ci && npm run dev -- --port 5173
```

Open http://localhost:5173. On Windows, activate with
`.venv\Scripts\Activate.ps1`; after activation, use `python` in place of
`python3`. `python3 -m app.ingest --skip-embeddings` skips the
model download and runs keyword-only; every ML capability is an optional layer
that reports its own availability and names the command that enables it.
The Vite development proxy targets `http://localhost:8000` by default. To
verify against an isolated backend without stopping another local server, start
Vite with `CVDE_DEV_API=http://127.0.0.1:8001 npx vite --port 5174`.

Optional layers, each honest about its absence:

| Layer | Enable with |
|---|---|
| Assistant (local agents) | [Ollama](https://ollama.com) + `ollama pull qwen3:8b`, then `pip install -r requirements-agent.txt` |
| Local vision inspector | Ollama + explicit `ollama pull gemma4:12b` and/or `ollama pull qwen3.5:9b`; select aliases with `CVDE_VISION_MODELS` |
| Semantic pair inspector | The capability-tested `qwen3.5:9b` Ollama artifact; bind a revalidated alias, digest, and runtime with `CVDE_VISION_PAIR_MODEL`, `CVDE_VISION_PAIR_MODEL_DIGEST`, and `CVDE_VISION_PAIR_RUNTIME_VERSION` |
| VLM tag enrichment | `ollama pull qwen2.5vl:7b`, then `python3 -m app.enrich` |
| Self-QA browser sweep | `pip install -r requirements-qa.txt` (Playwright + real Chrome) |
| Qwen3-VL retrieval provider | `pip install -r requirements-qwen.txt`, then `python3 -m app.ingest --provider qwen3_vl` |
| Region suggestions (zero-shot detector) | `python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='IDEA-Research/grounding-dino-tiny', revision='a2bb814dd30d776dcf7e30523b00659f4f141c71')"` |
| Promptable object masks (SAM 2.1 tiny) | `python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/sam2.1-hiera-tiny', revision='de431c4043854a71d8101e17995dfe596bf101a5')"` |
| Container path | `docker compose up --build` — see [docs/DEPLOY.md](docs/DEPLOY.md) |

The detector and segmenter load only those immutable local snapshots. Override
the model and full commit together with `CVDE_DETECT_MODEL` /
`CVDE_DETECT_REVISION` or `CVDE_SEGMENT_MODEL` /
`CVDE_SEGMENT_REVISION`. Every newly accepted mask records both values; legacy
masks honestly report an unknown revision rather than inheriting the current
configuration.

Ollama's generic `vision` flag establishes single-image input, not dependable
two-image comparison. The pair adapter is therefore bound to the exact local
Qwen digest and Ollama runtime that passed the ordered-frame contract. Pulling
new weights under the same alias or upgrading the runtime disables the
capability until the contract is rerun; neither silently inherits the old
result:

```bash
backend/.venv/bin/python scripts/validate_pair_vision.py
```

The probe uses two frozen source images and validates the typed response,
proposed grounding terms, exact model digest, runtime, and adapter protocol for
both frames. Add
`--write backend/data/reports/pair-vision-validation.json` to retain the full
local evidence record under the gitignored data directory.

## The model decision, measured

Retrieval defaults to **SigLIP 2** because it measures better on this corpus's
own benchmark — not because it is smaller: parameter count is not quality.
Qwen3-VL-Embedding-2B runs behind the same provider seam as an explicit
opt-in (`CVDE_EMBED_PROVIDER=qwen3_vl`); each provider keeps its own index
directory and schema-v2 commit marker. The marker binds the exact cached
Hugging Face revision and preprocessing fingerprint to the vector dimensions,
row counts and ordered sample/caption IDs, and is written atomically only after
all four arrays are complete. A legacy or mismatched index is refused rather
than mixed with a live query encoder; the UI names whichever provider is
actually ranking and why a fallback happened.

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

![Hybrid search for "a crowded street at night": ranked grid, the fusion basis
named in the result header, matched query terms highlighted in each caption, and
the ranked/grouped toggle beside the difficulty-axis legend. Per-card scores are
hover-gated, so they are not in this still](docs/screenshots/1-gallery.png)

Semantic, keyword (FTS5 BM25) and hybrid (reciprocal-rank fusion) — one
ranking implementation serves the gallery, the export buttons and the agents,
so they can never disagree. Every score names its basis (`cosine`,
`cosine_adj`, `rrf`, `composed`); scores from different bases are never
compared. Search by image (drop, paste or pick a file), or steer with
reference chips: *More like this* / *Exclude* build a composed query in the
URL (`?like=76&unlike=13` plus text), so a colleague can open the same
steered search from a pasted link — and the Back button walks the trail.
Exclusion alone is a real direction: a negative-only query ranks the corpus
by distance from the excluded examples and says so. Recent queries drop down
inside the search field, mode, ordering and thumbnail size share one *Search
settings* popover beside it, and every committed search joins the workspace
trail the History drawer shows — each row travelling back to the exact view it
recorded.

![Zero-shot facet filters with a set description panel open](docs/screenshots/2-describe.png)

Facet filters include exploratory zero-shot assignments over hand-authored
prompt banks. Ambiguous images abstain, and the UI presents each distribution
as a review hypothesis rather than a ground-truth taxonomy; the description
panel summarizes any filtered set with counted, never generated, statements.

### Judge difficulty, not just content

![Gallery filtered to the hardest, least legible samples](docs/screenshots/4-axes.png)

Four 0–10 axes — legibility, rarity, difficulty, clutter — computed as
percentile ranks over the corpus, each a filter, a badge and a sort key. "The
300 hardest validation samples" is a URL.

### See the corpus shape

![UMAP embedding map coloured by difficulty, reporting 12 k-means clusters
computed in the original 768-D space. The working set opens empty and offers
two measured entry points — most isolated, near-duplicate pairs — beside the
neighbour statistics they derive from, above a panel stating what the
projection cannot tell you](docs/screenshots/5-map.png)

The embedding map projects the corpus (UMAP, ingest-time); lasso a region to
tag or open it as a gallery slice. Isolated points are coverage-gap
candidates.

### Audit the annotations

![Caption quality view: least-supported captions first](docs/screenshots/6-quality.png)

CLIPScore-style agreement between each caption and its own image, worst
first — likely annotation errors surface immediately. Per-sample caption
consistency flags images whose five captions disagree with each other.

### Trust, measured

![Dataset profile, split-integrity view: train/test leakage as a threshold ladder with judgeable cross-split pairs](docs/screenshots/7-stats.png)
![The self-benchmark table: recall@k for all three modes](docs/screenshots/8-eval.png)

The adapter pins the exact reviewed Flickr8k Hub commit, and the stats page
states the dataset's provenance and known defects. The
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
Mark a rectangle on the image itself, add keep/remove points, or accept a
zero-shot Grounding DINO box proposal. SAM 2.1 turns that prompt into a visible,
refinable mask; the editor keeps the leaf class and explicit parent separate
(`dog` inside `animal`), and a human click accepts the mask as an annotation.
Acceptance persists the exact reviewed PNG—SAM is not run a second time—and a
short-lived server signature binds those bytes to the source-image SHA-256,
prompt geometry, model revision, predicted IoU, and mask dimensions. Search can
use that saved object rather than the full source frame. Today the masked-object vector and
optional leaf-label vector rank the existing full-image index, and the result
header states that boundary rather than implying an object-patch index.
Rectangle search **toward or away from the region** remains available as the
fast fallback. `scripts/bench_detector.py` and `scripts/bench_sam2.py`
re-measure both optional models.

If the mask began with a detector proposal, the accepted record also preserves
Grounding DINO's exact revision, full query, original/proposed labels, score and
source box. The detector evidence reaches acceptance through a short-lived,
server-authenticated proposal token rather than trusted client JSON. A reviewer
can relabel without erasing that disagreement. Saved annotations expose
masked-object search plus one atomic ZIP export. The package contains the
accepted binary mask, a tight RGBA object cutout whose alpha channel is that
mask, and a manifest binding the source, mask, and cutout byte lengths and
SHA-256 values to the annotation, model revisions, and the documented Pillow
operations that derived it. The source frame is never modified.

The same sample page has a read-only **local vision inspector**. It runs one
typed task—scene inventory, road-scene triage, caption audit, OCR, or a focused
image question—against an explicitly configured, already-installed Ollama
vision model. Results are visibly marked as model proposals and bind the exact
decoded-source SHA-256/dimensions/mode/byte length, model digest, input SHA-256,
latency, prompt version and schema version. Proposed
classes can populate the measured Grounding-DINO → SAM2 workbench, but no result
creates a label or changes source data. Run a second local model to expose
disagreement instead of hiding it. The design decisions and deferred
OWLv2 / Florence-2 / SAM3 evidence are in
[ADR-0001](docs/adr/0001-local-vision-inspection-workbench.md) and
[ADR-0002](docs/adr/0002-segmentation-and-dataset-preparation.md).

### Compare two frames

![Compare canvas: synchronized zoom, shared/different panel](docs/screenshots/14-compare.png)

`/compare` puts two samples under one loupe. Synchronized zoom/pan and manual
rectangle search remain deterministic tools. A separate Inspection Run first
verifies and decodes both local files, then asks the capability-tested Qwen
artifact for a typed **semantic difference proposal**. The report keeps visible
pose/presence/appearance changes separate from embedding cosine, stored
attributes, dHash duplicate triage, and corruption. Proposed object phrases can
enter the existing Grounding DINO → SAM2 flow, but only a reviewer can create an
annotation. Every exported comparison binds both source SHA-256 values, decode
dimensions, mode and byte length, the exact model digest, provider/runtime
version, adapter, proposal ID, prompt/schema versions, protocol, and latency.
Source images are never modified. See
[ADR-0003](docs/adr/0003-sequential-inspection-runs-and-pair-comparison.md).

### Curate into albums, and let them leave

![An album's header: editable details and the measured analysis panel](docs/screenshots/15-album.png)
![The Share menu: copy link, email, Teams compose, downloads — everything local](docs/screenshots/16-share.png)

Albums are first-class ordered collections (provenance `manual` | `tag` —
converting a tag is explicit and keeps the tag). Pick images with the ✓ each
card carries — there is no mode to enter, so a click still opens the image —
or keep a whole ranking as an album straight from the result bar. Drag cards
onto the shelf —
every drop offers an Undo that removes exactly what the drop added — reorder
by drag or arrow keys, set a cover, edit the summary where you read it. The
Analyze panel counts what members share and where they split, and computes
coherence and outliers from the active index; a summary draft from the local
chat model is generated only on request, edited by you, saved only by you.
That stored-signal analysis is separate from **Inspect album pixels**: the
bounded visual run refreshes and freezes the ordered membership, verifies the
selected model's exact digest, and calls the same documented single-image
inspection contract sequentially for at most eight members. Progress is visible
per image; decode or structured-output failures remain beside successful
proposals; a busy accelerator or timeout stops after the current member.
Every result links back to source review and, for a scene proposal, into the
Grounding-DINO → SAM2 path. One JSON manifest exports the frozen snapshot,
ordered successes/failures, source/model provenance, and unstarted members.
It is a browser-session run, not a durable scheduler: reload discards the
on-screen run unless its manifest was downloaded, and no proposal is written
as a caption or annotation.
Share stays local-first: copy the URL, compose an email or Teams message
(nothing is uploaded), or download the slice as JSON/CSV/JSONL with a
manifest that records the query, model and embedding fingerprint that
produced it.

### Experimental assistant boundary

![Assistant conversation with agent trace and rendered blocks](docs/screenshots/11-assistant.png)

The repository also contains optional LangGraph orchestration over local Ollama:
a schema-constrained
orchestrator routes to registered retrieval, insights, visualization or QA
specialists (at most two cheap lanes in parallel), and a typed synthesizer
quality-gates the answer. **The graph's own node transitions stream into the UI
as they happen** — the progress you watch is the run, not an animation.
Retrieval tools call the same `run_search` service as the REST API; inspection
tools make read-only queries against the same SQLite store. The assistant can
inspect albums: ask about a rare-scenario album and the answer cites its
measured signals and outliers. It cannot write dataset state directly: its tag
tool returns a **proposal**, and the browser sends the tag request only after
your click. Conversations and generated reports persist locally; conversations
support rename/reopen/delete.

This assistant is **not part of the verified submission path**. It remains an
optional, experimental surface: the router and synthesizer use typed structured
outputs, but the deterministic end-to-end probe is the promotion gate for
quantitative prose. Until every fixed probe passes repeatedly, confirm numbers
against the cited tool results and live blocks. The verified product path is the
gallery, sample workbench, comparison, album inspection, audit, and export
surfaces, and none of those depend on the assistant.

![Command palette over every route and action](docs/screenshots/10-palette.png)
![The front door, with the rail's local-models card](docs/screenshots/17-models.png)

⌘K reaches everything; the rail's models card says which retrieval provider,
assistant and enrichment model are real, live and local right now — with the
named reason when one is missing.

### Let other agents in, read-only

A local [MCP server](docs/MCP.md) at `POST /mcp` exposes eight strictly
read-only tools, including saved-mask inspection and object-specific retrieval,
over the same service layer — any local MCP client, LangGraph included, can
investigate the corpus and cannot curate it.

## Architecture

One FastAPI process, one SQLite file (WAL), embeddings as `.npy`, images on
disk; a React 18 + TypeScript frontend whose search and filter state lives in
the URL (an investigation is a link). Heavy work happens in idempotent batch
CLIs (`ingest`, `analyze`, `enrich`); a request does SQLite lookups plus at
most one text-encoder forward pass. Retrieval is an exact NumPy cosine scan —
measured at ~0.2 ms against the 7–8 ms encode it waits behind, so at this size
an approximate index would trade recall to speed up the fastest stage of the
query. The seam is there rather than the machinery: `EmbeddingIndex.search`
takes a candidate mask, so an ANN index drops in behind the same signature for
text search — worth *benchmarking* around 100k vectors, with an
*estimated* crossover near 400k that is extrapolation, not permission: adoption
waits on a real FAISS recall-and-latency benchmark against the exact scan. A
hosted vector database is not
the answer to either (pgvector answers a different question — a multi-user
server deployment — and is discussed in
[ARCHITECTURE](docs/ARCHITECTURE.md)). Optional capabilities degrade with a
named reason, never a 500.

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
  api/            typed client — every route the pages call through
backend/app/
  api/            20 REST routers + one MCP router (run_search is the one ranking impl)
  ml/             providers (SigLIP 2 / Qwen3-VL) · exact index · detector · SAM2 segmenter
  agent/          LangGraph graph, tools, render-block contract
  qa/             flow registry + real-Chrome runner (one definition, three consumers)
  ingest/analyze/enrich    idempotent batch CLIs
docker/ + docker-compose.yml    the container path
scripts/        benchmarks, screenshots, capabilities/link checks
```

## Test evidence

```bash
(cd backend && .venv/bin/python -m pytest)
(cd frontend && npx tsc --noEmit && npm run build)
(cd backend && .venv/bin/python ../scripts/ui_smoke.py)
```

The real-Chrome sweep writes its run-specific check and workflow totals under
`backend/data/qa/`. Use that generated report as the evidence for the checkout
being reviewed; the registry grows, so this README deliberately does not cache
the tally. CI runs the light install (no torch/langgraph — those modules skip),
ruff, tsc, the build, the link check and the capabilities contract.

## Honest limitations

- Qwen3-VL underperforms SigLIP 2 on this benchmark (table above); the hubness
  constants were tuned in the SigLIP space and applied untuned in Qwen's —
  whether tuning closes any of the gap is unmeasured.
- The composed (reference-steered) ranking is unmeasured — the benchmark
  covers the text modes only; its blend weight is chosen, not tuned, and the
  code says so.
- The keyword benchmark row is *understated*, not flattered: the query caption's
  own row is excluded from the FTS scan (otherwise it would measure nothing but
  self-retrieval), and the strict AND conjunction then returns an empty
  candidate list for 85.3% of queries, so keyword R@10 reads 5.8% over a mean of
  2.1 candidates. Widening the conjunction raises that recall but lowers fused
  MRR overall (hybrid MRR 0.6313 → 0.5850), which is why it stays — the figures come from the
  benchmark's own cached run, and the page states the caveat where the number
  appears.
- The experimental assistant needs Ollama and a ~5 GB model. It remains
  available, but its deterministic quantitative probe—not a prompt-only
  assertion—is the release gate; the documented CV workbench remains
  independent of it.
- SAM 2.1 tiny runs here at 72 ms/mask warm (box-prompt p50; 73 ms by point
  prompt, 1.5 GB peak, 60 interleaved calls with no crash and 2 MB of drift).
  The retrieval benchmark is deliberately modest: background removal changed
  63% of the top-10 while a caption-word proxy moved only +0.009 over 16
  regions. Masks therefore ship for interaction, annotation and an explicit
  object-search mode—not as a claimed universal ranking improvement.
  `scripts/bench_sam2.py` records the full measurement.
- MCP is stateless JSON (no SSE streaming, no sessions); it binds to the same
  local server and adds no authentication — a local, single-user tool by the
  assignment's constraint.
- In-container inference is CPU-only and slower than the host path
  ([docs/DEPLOY.md](docs/DEPLOY.md) states expectations).
- The dataset copy itself is imperfect and the stats page says how (missing
  ~90 images vs the original distribution, undocumented split assignments, no
  licence metadata on the Hugging Face card; the original Flickr8k terms limit
  use to non-commercial research and education).

## Configuration

Everything is environment variables; [.env.example](.env.example) lists each
one with its default and effect. The ones that matter first:
`CVDE_DATA_DIR`, `CVDE_EMBED_PROVIDER` (`siglip2` default | `qwen3_vl`),
`CVDE_EMBED_MODEL`, `CVDE_QWEN_EMBED_MODEL`, `CVDE_OLLAMA_URL`,
`CVDE_CHAT_MODEL`, `CVDE_VLM_MODEL`.
