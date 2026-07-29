# ADR-0003: Sequential inspection runs and honest pair comparison

## Status

Accepted as the product workflow. This change implements the pair-comparison
vertical slice and a bounded browser-session album inspection run. Durable
album-run persistence and scheduling remain future work.

## Context

The application already has useful local capabilities, but they appear as
separate surfaces:

- image/caption inspection through a schema-constrained local VLM;
- text-conditioned Grounding DINO boxes;
- point/box-prompted SAM2 masks;
- masked-object retrieval and annotation export;
- album-level measurements over captions, attributes, and embeddings;
- a two-image loupe with embedding similarity and stored metadata differences.

The missing product contract is the sequence between them. In particular, the
Compare view's “Shared & different” table compares stored metadata and does not
inspect the two image contents. The new dHash leakage endpoint is a
duplicate-frame heuristic; it is not a general corruption detector or semantic
difference engine.

Three interface directions were evaluated:

1. a small provider façade with `inspect`, `ground`, and `segment`;
2. a fully generic persisted workflow DAG with arbitrary operations and events;
3. a preset Inspection Run with a fixed scope, real stages, typed artifacts,
   and human review.

The façade is a clean internal seam but does not explain progress, authority, or
review to a user. The generic DAG captures every future workflow but adds
database schema, validation, scheduling, and expert configuration before the
common case is proven.

## Decision

Use preset, scope-bound Inspection Runs:

```text
scope snapshot
  -> asset health
  -> vision proposal
  -> open-vocabulary grounding
  -> selective segmentation/isolation
  -> human review
  -> search/export
```

The common UI shows those real stages. Internally, model-specific details remain
behind the existing vision, detection, and segmentation adapters. A future
engine may represent the preset as a validated DAG, but users and agents do not
construct arbitrary graphs for ordinary curation.

For the pair vertical slice:

- the public request contains only two distinct local sample IDs; the server
  selects the capability-tested adapter and exact artifact;
- both source files must pass Pillow verification and a full pixel load before
  model inference;
- the response preserves source filenames, dimensions, image modes, byte
  lengths, SHA-256 digests, model digest, provider/runtime version, adapter,
  proposal ID, prompt/schema versions, protocol, and latency;
- semantic differences are a typed `model_proposal`, distinct from exact-byte,
  duplicate-frame, registered-pixel, or corruption evidence;
- grounding phrases may be handed to the detector, but neither the phrase nor
  its box becomes an annotation without human review;
- the workbench exports the complete proposal as JSON.

People receive neutral instance descriptions. `person-1` and `person-2` are
local geometric instance identities, not new learned classes. The VLM must not
infer gender or identity, and must not claim that an identity persists across
two frames.

## Capability validation

Ollama documents image inputs in chat messages and JSON Schema structured
outputs, but its generic `vision` capability does not establish reliable
two-image comparison for every artifact.

Live tests on the exact installed artifacts found:

- `gemma4:12b` falsely reported that two different frames were one image when
  both images were supplied in one message;
- the same Gemma artifact falsely called the frames identical when they were
  supplied in ordered messages;
- `qwen3.5:9b` correctly described pose and flame-position differences only
  with `user(image A) -> assistant acknowledgement -> user(image B + schema)`.

Pair comparison is therefore an application-level, artifact-and-runtime-specific
capability. The configured model alias is bound to the exact Ollama digest and
runtime version that passed the ordered-message contract. Pulling different
weights under that alias or upgrading Ollama disables pair comparison until the
frozen probe passes and the pin is explicitly updated. No model-name conditional
appears in the endpoint or UI.

The acceptance probe is executable:

```bash
backend/.venv/bin/python scripts/validate_pair_vision.py
```

It verifies the two frozen source hashes before inference and rejects a response
that collapses the frames, lacks a concrete typed difference, lacks grounding
handoffs for either frame, or reports a different artifact/runtime/protocol.
Pass `--write backend/data/reports/pair-vision-validation.json` to retain the
full timestamped runtime record locally; generated evidence stays in the
gitignored data directory rather than in the documentation tree.

## Consequences

Positive:

- Compare now distinguishes visual evidence from metadata similarity.
- Corrupt or unreadable inputs stop before inference with an explicit error.
- A proposal can lead directly into grounding and selective segmentation.
- Every result is reproducible against two source hashes and one exact model
  artifact.
- The same stage vocabulary extends naturally to sample and album scopes.
- Album inspection refreshes and freezes up to eight ordered members, runs the
  existing single-image contract sequentially, shows per-item progress and
  partial failures, links proposals into review/grounding, and exports one
  ordered JSON manifest.

Negative:

- Only one installed artifact currently passes the pair contract.
- Semantic comparison is explanatory, not a registered pixel-difference
  measurement.
- The bounded album run is transient: a reload discards its on-screen state
  unless the manifest was downloaded. Durable resume, a scheduler, retry
  policy, and larger-scope budgets remain future work.

## Rejected workarounds

- Do not treat dHash distance as corruption or general image difference.
- Do not parse malformed model prose with regex or silently repair JSON.
- Do not infer pair readiness from an artifact's generic `vision` flag.
- Do not hardcode rare-object dictionaries or rewrite detector labels.
- Do not segment every object in an album by default; run masks only for
  selected, rare, uncertain, or policy-triggered instances.
- Do not add a model selector showroom. Add a provider only after a frozen
  evaluation demonstrates downstream review value.

## Next model benchmark

The already-cached, immutable
`google/owlv2-base-patch16-ensemble` artifact is the next grounding benchmark.
Unlike another captioning VLM, OWLv2 adds documented text-conditioned and
image-guided detection. The accepted-exemplar path can search for visually
similar rare instances, after which the existing SAM2 adapter produces a
reviewable mask.

OWLv2 does not become the default until it passes the frozen quality, no-match,
latency, memory, provenance, and accepted-without-edit gates in
[the model research note](../research/2026-07-29-open-vocabulary-vision-models.md).

## Official references

- Ollama vision:
  <https://docs.ollama.com/capabilities/vision>
- Ollama chat API:
  <https://docs.ollama.com/api/chat>
- Ollama structured outputs:
  <https://docs.ollama.com/capabilities/structured-outputs>
- Pillow image verification and loading:
  <https://pillow.readthedocs.io/en/stable/reference/Image.html>
- Transformers OWLv2:
  <https://huggingface.co/docs/transformers/model_doc/owlv2>
- Transformers Grounding DINO:
  <https://huggingface.co/docs/transformers/model_doc/grounding-dino>
- Transformers SAM2:
  <https://huggingface.co/docs/transformers/model_doc/sam2>
