# Capabilities

Every view, endpoint, agent tool and tested workflow in this system.

**Generated** — do not edit by hand. Regenerate with
`python scripts/capabilities.py` while the API is running; verify with
`--check`. It is built from the live OpenAPI schema, the agent registry,
the QA flow registry and the router in `App.tsx`, so it cannot describe a
capability the code does not have.

- 8,000 images loaded · semantic search **on**
- 70 HTTP endpoints · 4 agent specialists · 22 agent tools · 17 tested workflows

## Views

| Job | Route | View | What it is for |
| --- | --- | --- | --- |
| Find | `/` | Browse | Browse and search; every filter and the paging depth live in the URL. |
| — | `/samples/:id` | Sample | One image: captions, attributes, exact neighbours, and a promptable object-mask editor, masked-object search, and typed local VLM proposals. |
| Find | `/map` | Embedding map | UMAP projection of all embeddings. Lasso a region to hand that exact set to the gallery. |
| Trust | `/stats` | Overview | Splits, caption lengths, vocabulary, image sizes, and abstaining prompt-slice review hypotheses. Bars open their slice. |
| Audit | `/quality` | Caption quality | Caption agreement distribution with a review threshold; the selection can leave as a gallery filter or an export. |
| Trust | `/eval` | Retrieval benchmark | The tool measuring its own retrieval accuracy — R@1/5/10 for all three search modes. |
| Ask | `/chat` | Assistant | Experimental local assistant. Its end-to-end claim-verification probe must pass before this route is considered release-qualified. |
| Find | `/compare` | Compare two | Two images under one synchronized zoom; typed local semantic difference proposals remain separate from stored metadata, duplicate triage and manual region search. |

Navigation is grouped by job in a persistent left rail; the current selection has a permanent home in a right rail that appears whenever something is selected. Plus **⌘K** anywhere: a command palette over routes, samples, tags, attribute slices, saved views and search.

## HTTP API

Interactive schema at [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs) while the server runs.

### Browse and inspect

| Endpoint | |
| --- | --- |
| `GET /api/export` | Export Subset |
| `GET /api/samples` | List Samples |
| `GET /api/samples/{sample_id}` | Get Sample |
| `GET /api/samples/{sample_id}/annotations` | List Annotations |
| `POST /api/samples/{sample_id}/annotations` | Add Annotation |
| `GET /api/samples/{sample_id}/segment-annotations` | List Segment Annotations |
| `POST /api/samples/{sample_id}/segment-annotations` | Accept Segment Annotation |
| `GET /api/samples/{sample_id}/similar` | Similar Samples |
| `POST /api/samples/{sample_id}/tags` | Add Tag |
| `DELETE /api/samples/{sample_id}/tags/{name}` | Remove Tag |

### Search

| Endpoint | |
| --- | --- |
| `POST /api/detect` | Detect Regions |
| `GET /api/detect/status` | Detect Status |
| `GET /api/search` | Search |
| `POST /api/search` | Search Post |
| `POST /api/search/by-annotation` | Search By Annotation |
| `POST /api/search/by-image` | Search By Image |
| `POST /api/search/by-region` | Search By Region |
| `POST /api/search/composed` | Search Composed |
| `POST /api/search/scenarios` | Search Scenarios |
| `POST /api/segment` | Preview Segment |
| `DELETE /api/segment-annotations/{annotation_id}` | Delete Segment Annotation |
| `GET /api/segment/status` | Segment Status |

### Local vision inspection

| Endpoint | |
| --- | --- |
| `POST /api/vision/compare` | Compare Images |
| `POST /api/vision/inspect` | Inspect Image |
| `GET /api/vision/models` | List Vision Models |

### Statistics and map

| Endpoint | |
| --- | --- |
| `GET /api/attributes/coverage` | Coverage |
| `GET /api/describe` | Describe Set |
| `GET /api/map` | Embedding Map |
| `GET /api/stats/captions` | Caption Stats |
| `GET /api/stats/duplicates` | Duplicates |
| `GET /api/stats/leakage` | Leakage Report |
| `GET /api/stats/leakage/contaminated` | Contaminated Ids |
| `GET /api/stats/leakage/pixel` | Pixel Report |
| `GET /api/stats/overview` | Overview |

### Annotation QA

| Endpoint | |
| --- | --- |
| `GET /api/qa/captions` | Suspect Captions |
| `GET /api/qa/consistency` | Inconsistent Samples |
| `GET /api/qa/selection` | Qa Selection |
| `GET /api/qa/summary` | Qa Summary |

### Retrieval benchmark

| Endpoint | |
| --- | --- |
| `GET /api/eval/retrieval` | Retrieval Benchmark |

### Curation

| Endpoint | |
| --- | --- |
| `DELETE /api/activity` | Clear Activity |
| `GET /api/activity` | List Activity |
| `POST /api/activity` | Add Activity |
| `DELETE /api/activity/{event_id}` | Delete Activity |
| `GET /api/albums` | List Albums |
| `POST /api/albums` | Create Album |
| `POST /api/albums/from-tag` | Album From Tag |
| `PUT /api/albums/order` | Reorder Albums |
| `DELETE /api/albums/{album_id}` | Delete Album |
| `GET /api/albums/{album_id}` | Get Album |
| `PATCH /api/albums/{album_id}` | Update Album |
| `GET /api/albums/{album_id}/analysis` | Album Analysis |
| `POST /api/albums/{album_id}/analysis/summary` | Album Summary Generate |
| `POST /api/albums/{album_id}/items` | Add Items |
| `PUT /api/albums/{album_id}/items/order` | Reorder Items |
| `DELETE /api/albums/{album_id}/items/{sample_id}` | Remove Item |
| `DELETE /api/annotations/{annotation_id}` | Delete Annotation |
| `GET /api/annotations/{annotation_id}/cutout` | Get Annotation Cutout |
| `GET /api/annotations/{annotation_id}/export` | Export Annotation Package |
| `GET /api/annotations/{annotation_id}/mask` | Get Annotation Mask |
| `GET /api/object-labels` | List Object Labels |
| `GET /api/tags` | List Tags |
| `POST /api/tags/bulk` | Bulk Tag |
| `GET /api/views` | List Views |
| `POST /api/views` | Create View |
| `DELETE /api/views/{name}` | Delete View |
| `GET /api/vlm-tags` | List Vlm Tags |

### Assistant

| Endpoint | |
| --- | --- |
| `GET /api/agent/graph` | Agent Topology |
| `POST /api/chat` | Chat |
| `GET /api/chat/status` | Chat Status |
| `POST /api/chat/stream` | Chat Stream |
| `GET /api/reports/{name}` | Download Report |

### Other agents (MCP)

| Endpoint | |
| --- | --- |
| `GET /mcp` | Mcp Info |
| `POST /mcp` | Mcp Endpoint |

### Application self-QA

| Endpoint | |
| --- | --- |
| `GET /api/qa/artifact/{run_id}/{name}` | Qa Artifact |
| `GET /api/qa/flows` | List Qa Flows |
| `GET /api/qa/run` | Latest Qa Run |
| `POST /api/qa/run` | Start Qa Run |
| `GET /api/qa/run/{run_id}` | Get Qa Run |

### Operations

| Endpoint | |
| --- | --- |
| `GET /api/admin/integrity` | Integrity |
| `POST /api/admin/reload` | Reload Indexes |
| `GET /api/health` | Health |

## Assistant

Model `qwen3:8b` via local Ollama. The orchestrator selects up to
2 specialists per request and runs them in
parallel; LangGraph bounds each lane at 90s and the API bounds the complete turn at 150s.

### `retrieval`

Finding or showing images: search, similar images, inspecting or tagging specific samples.

- `search_images`
- `find_similar`
- `find_similar_to_annotation`
- `get_sample_details`
- `list_annotations`
- `tag_samples`
- `inspect_album`

### `insights`

Dataset statistics, abstaining prompt-slice review hypotheses and coverage, rare or long-tail slices, caption quality and annotation errors, stated in numbers.

- `dataset_overview`
- `attribute_coverage`
- `rare_slice_examples`
- `suspect_captions`
- `get_sample_details`
- `inspect_album`
- `plot_distribution`

### `visualization`

Anything to plot, chart, diagram or compare visually; a written report the user wants generated; and how this platform is built (its architecture, components, or how a request flows through it).

- `plot_distribution`
- `plot_retrieval_benchmark`
- `compare_slices`
- `show_images`
- `system_diagram`
- `build_dataset_report`

### `qa` *(expensive — runs alone)*

Whether this application is currently WORKING: run its tests, check for broken screens, report pass/fail status. Only when the user asks about health, tests, QA or breakage — NOT for how the platform is built, what it can do, or anything about the dataset.

- `app_qa_status`
- `run_app_qa`

Answers arrive as render blocks — bar, line, pie, histogram, table, stat,
flow, images, report, qa — each stating the SQL behind it, and each
clickable through to the gallery slice it describes.

## Tested workflows

Driven through real Chrome by `scripts/ui_smoke.py`, by `POST /api/qa/run`,
and by the assistant on request — one registry, three consumers.

| Workflow | Budget | What it asserts |
| --- | --- | --- |
| Routes | 240s | Every route loads and paints its own content. |
| Gallery | 150s | Browse, all three search modes, axis filter, sort, density, paging, export. |
| Similarity map | 150s | Colour modes, that the canvas is actually painted, hover, and lasso → |
| Statistics | 150s | Charts render across the profile's views, and the provenance panel states |
| Quality | 150s | Agreement histogram, the review threshold, and brush → gallery. |
| Benchmark | 300s | The self-benchmark runs and reports all three retrieval modes. |
| Sample detail | 150s | A card opens its sample, which shows every caption, similar images, and |
| Assistant | 150s | The chat page states a definite state: ready, or unavailable with setup |
| Axis legend | 60s | The key for the four-bar sparkline every card carries. |
| Set description | 90s | The inversion: given a selection, what characterises it. |
| Train/test leakage | 90s | Held-out images with a training near-duplicate. |
| Data integrity | 45s | The embedding indexes and the database must still describe each other. |
| Graceful degradation | 90s | A failing data source must announce itself, and a missing one must not. |
| Command palette | 60s | ⌘K reaches anything without navigating first, and gets out of the way. |
| Assistant canvas | 300s | A charting request comes back as a live component that drills into the data. |
| Compare | 60s | Two samples under one loupe: the transform is genuinely shared, the |
| Hero journey | 180s | The workspace's spine as one continuous story: search -> inspect |

## Optional layers and how they degrade

| Layer | Needs | Without it |
| --- | --- | --- |
| Semantic search, map, benchmark | `requirements.txt` + `app.ingest` | Browsing, keyword search and stats still work; the UI says which features are unavailable. |
| Caption QA, attributes, difficulty axes | `app.analyze` | Those views explain the command to run. |
| Region suggestions | cached Grounding DINO tiny weights | Manual point/box prompting remains available; the UI names the fetch command. |
| Promptable masks | cached SAM 2.1 tiny weights | Rectangle search remains available; the UI names the fetch command and never downloads on request. |
| Local vision inspector | Ollama + an explicitly configured, installed vision model | The sample inspector remains fully usable; no model is pulled or substituted by a request. |
| Semantic pair inspector | An explicitly configured Ollama artifact whose exact digest passed the ordered-frame contract | The loupe, stored-signal comparison and manual region tools remain usable. |
| Assistant | `requirements-agent.txt` + Ollama | The tab shows exact setup instructions; nothing else is affected. |
| Application self-QA | `requirements-qa.txt` (Playwright) | `POST /api/qa/run` returns 503 with setup instructions. |
| PowerPoint deck | `python-pptx` | The Markdown report is still produced and says the deck was skipped. |
| VLM tag enrichment | Ollama + a vision model | VLM tags are simply absent. |

## Further reading

- `README.md` — setup, feature tour, the two limits worth knowing
- `docs/TECHNICAL.md` — layer-by-layer build: schema, the real SQL and query plans, retrieval maths, frontend, measured performance
- `docs/DESIGN.md` — retrieval design, trade-offs, and the production scale path
- `docs/AGENTS.md` — orchestration, the render-block contract, self-QA
- `docs/DEMO.md` — an eight-minute walkthrough
- `docs/screenshots/` — one image per view

<!-- generated 2026-07-29 -->
