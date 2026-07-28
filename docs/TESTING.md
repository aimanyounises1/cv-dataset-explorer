# Testing — what is covered, by what, and what is not

Three tiers plus two consistency checks. The point of this page is the last
section: a green CI badge on this repository means less than a reader might
assume, and it is cheaper to say so than to be found out.

## Tiers

| Tier | Command (working directory) | Needs | In CI |
| --- | --- | --- | --- |
| Backend contracts | `pytest` (`backend/`) | the light install below | yes |
| Torch-dependent unit tests | same | `torch` | skipped |
| Agent orchestration | same | `requirements-agent.txt` | skipped |
| Frontend types + build | `npm ci && npm run build` (`frontend/`) | Node 18+ | yes |
| Relative Markdown links | `python scripts/check_links.py` (repo root) | stdlib | yes |
| Capability inventory | `python scripts/capabilities.py --check` (repo root) | a running API | yes — the backend job starts one for it |
| UI sweep, real Chrome | `python ../scripts/ui_smoke.py` (`backend/`) | Chrome, both servers, an ingested corpus | no |

The install CI uses is deliberately narrow -- `fastapi httpx pydantic numpy
pillow pytest ruff tqdm uvicorn` -- because the suite is written to run
without the ML stack: a fake embedder plants synthetic vectors in the real
on-disk index format, and the degradation paths are exercised rather than
mocked. `backend/requirements.txt` remains the manifest for actually serving
the app. See [../.github/workflows/ci.yml](../.github/workflows/ci.yml).

## What CI does not test, and why

- **Real SigLIP 2 retrieval.** The weights are a ~1.5 GB download and the
  runner has no MPS or CUDA. Retrieval quality is measured locally instead --
  by the Benchmark page and the offline harnesses -- and every number in the
  README says which protocol produced it.
- **Ingestion.** `python -m app.ingest` downloads Flickr8k. Nothing in CI
  touches the network beyond installing packages.
- **The inference lock on Metal.** The crash it prevents cannot be reproduced
  on a Linux runner, and the test that pins the invariant needs `torch`, so it
  skips there. It must be run on the developer machine.
- **The assistant end to end.** That needs Ollama and a local 8B model. The
  orchestration graph itself is tested with an injected model, but only where
  the optional agent requirements are installed.
- **The UI.** The browser sweep needs Chrome, both servers and an ingested
  corpus. It is the tier that catches a view rendering empty or a control that
  stopped filtering, and it is a local command, not a CI job.
- **The PowerPoint deck.** `python-pptx` is optional even for the sweep.

So a green run says: the HTTP contracts hold, the degradation paths behave, the
ranking invariants that are pure NumPy or SQL still hold, the frontend
type-checks and builds, and no relative link in the docs is broken. It does not
say that any retrieval measurement was re-run.

## High-risk contracts and what pins them

Each row is a behaviour whose failure would be silent -- a wrong number or a
wrong label rather than an exception.

| Contract | Test |
| --- | --- |
| semantic and hybrid degrade to keyword and say so | `test_smoke.py::test_hybrid_degrades_without_embeddings` |
| filters are applied inside the ranking, never after a LIMIT | `test_smoke.py::test_keyword_search_respects_filters_in_sql`, `::test_axis_filter_is_applied_inside_the_ranking_not_after_it`, `test_with_embeddings.py::test_semantic_search_respects_filters` |
| paging partitions one ranking and stops at the fusion horizon | `test_with_embeddings.py::test_search_pages_partition_the_ranking`, `test_smoke.py::test_paging_stops_at_the_ranking_horizon_instead_of_repeating` |
| a mode that exposes no score publishes no score basis | `test_smoke.py::test_keyword_mode_publishes_no_score_basis` |
| the hubness penalty never leaks into the image-to-image paths, and a stale penalty is never applied to a rebuilt index | `test_hubness.py` |
| the benchmark excludes the query caption from the index it searches | `test_with_embeddings.py::test_benchmark_excludes_the_held_out_caption`, `::test_lexical_candidates_always_excludes_the_query_caption` |
| the benchmark measures the shipped path, and says so when it cannot | `test_with_embeddings.py::test_benchmark_encodes_queries_through_the_search_seam`, `::test_benchmark_says_so_when_it_could_not_encode_the_queries` |
| every search mode the gallery can produce survives into export | `test_mode_parity.py` |
| a pasted id list composes with every filter and respects SQLite parameter limits | `test_id_filter.py` |
| repeated attribute facets intersect, order-independently | `test_facet_composition.py` |
| optional components report their own availability rather than 500 | `test_smoke.py::test_qa_and_eval_degrade_gracefully`, `::test_chat_unavailable_is_graceful` |
| axis buckets are percentile ranks, deterministic under ties | `test_index.py` |
| agent fan-out is concurrent, lanes are isolated, a hung lane is cut off | `test_agent_graph.py` (needs `langgraph`) |
| the QA flow registry, report rendering and single-run lock | `test_qa_runner.py` |

Sources: [../backend/tests/](../backend/tests/).

## Known gaps, stated rather than closed


- **No frontend unit tier.** A deliberate trade recorded in
  [TECHNICAL.md](TECHNICAL.md): for a UI this size the failures that matter are
  "the view rendered empty" and "the control stopped filtering", which the
  browser sweep asserts and a component test would not.
- **The UI sweep is only as good as its selectors**, and it does not run in CI.
- **`ruff` is unpinned in CI.** A future release can therefore turn a run red
  without a change to this repository.

See also [CAPABILITIES.md](CAPABILITIES.md), which is generated from the live
system and fails `--check` when it drifts.
