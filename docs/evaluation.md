# Reproduce retrieval evaluation

This evaluates caption-to-image retrieval within the ingested corpus. A query
caption's own row is excluded from lexical retrieval; sibling captions remain.
The hubness-bank caption IDs are excluded from the query sample. This is an
application benchmark, not a claim about generalization to a new dataset.

## Prepare and run

Follow [local setup](../README.md#run-locally), including ingestion. For a
two-provider comparison, install the optional Qwen requirements and ingest
both providers against the same data. Model downloads and inference require
local disk, memory and processing time; the test suite does not do this.

```bash
cd backend
.venv/bin/python -m pip install -r requirements-qwen.txt
.venv/bin/python -m app.ingest --provider siglip2
.venv/bin/python -m app.ingest --provider qwen3_vl
.venv/bin/python ../scripts/bench_providers.py --sample-size 1000 \
  --output ../artifacts/retrieval-benchmark.json
```

`--output` is relative to the invoking shell and is not redirected into the
generated data directory. The default remains
`backend/data/cache/bench_providers.json`; that directory is ignored by Git.
Choose a durable path when sharing a measurement. The script can run with
only one provider present: each missing provider receives a named reason.

For the active provider only, start the API and request
`GET /api/eval/retrieval?sample_size=1000`. Valid sample sizes are 50–5000;
the actual sample is smaller if fewer aligned captions remain. If none remain,
the endpoint returns `available=false` and a reason, rather than an undefined
mean or a successful empty benchmark.

## What an artifact proves

Each provider's `retrieval` object is the complete endpoint response. Its
metadata records:

- Protocol version, fixed sampling seed, tie policy and ordered caption IDs.
- Provider/model, immutable model revision, validated artifact generation and
  the provider manifest, including index hashes and ingest library versions.
- Digest of all `(caption ID, sample ID, text)` rows used by lexical retrieval.
- Search depth, RRF constant, hubness settings, actual correction application
  and the excluded hubness-bank caption IDs.
- `live_normalized_text` or `stored_caption_vectors`, plus evaluation time.
- Evaluation runtime package versions, separately from the ingest manifest.

The outer artifact records export time, source commit, whether that working
tree was dirty, Python/platform, latency sample counts and memory observations.
`evaluated_at` may precede export time: the API caches results by protocol,
provider generation, encoding mode, runtime package versions, corpus-caption
digest and ranking settings.

Compare only matching protocol, corpus digest, query sample and encoding mode.
Model revisions and provider generations identify what changed. A dirty tree
requires retaining its diff to reconstruct the exact source. Retain data and
the named provider artifacts too: a JSON fingerprint identifies files but does
not recreate them. Numerical results can vary with runtime/library changes;
the ingest manifest records the libraries that produced the indexes and
`runtime_versions` records the libraries available when evaluation ran.

Recall is reported at 1, 5 and 10. MRR is truncated at depth 10. Median rank
is exact for semantic search and censored after depth 10 for keyword/hybrid.
Candidate counts and empty-query rates distinguish weak ranking from a path
that matched no candidates. A stored-vector fallback measures raw caption
vectors, so it must not be presented as live search-encoder quality.

Latency samples use 100 captions and 20 images (or the available smaller
corpus), one unmeasured warm-up per modality, then serial single-item encodes.
These are local microbenchmarks, not request latency or service-load results.
The evaluation may be cached; its compute time is not a reported latency.

## Verified correctness versus unmeasured quality

```bash
cd backend
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q tests/test_index.py \
  tests/test_with_embeddings.py tests/test_benchmark_export.py
```

The deterministic fixtures need no model or dataset download. They check tie
ordering, exact ranks, empty query handling, metadata, stored-vector labeling,
and preservation of the full response when exported. Four identical vectors
yield R@1 = 0.25 under protocol 11, while the earlier optimistic rank formula
yielded 1.0. This is a constructed regression, not a Flickr8k accuracy result.

No new full-corpus model comparison is committed with this protocol change.
Earlier local model-choice figures are historical and lack the provenance
required above. Re-run and retain a new artifact before making quality,
speed, or comparative performance claims.
