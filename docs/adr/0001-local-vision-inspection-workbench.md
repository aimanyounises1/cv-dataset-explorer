# ADR-0001: Local vision inspection is a proposal workbench

## Status

Accepted for implementation.

## Context

The product already measures retrieval quality, caption agreement, embedding
neighbourhoods, promptable boxes, and SAM2 masks. The weak point is the hand-off:
several views expose a chart or model preview without helping a CV engineer turn
it into a search slice, annotation task, review decision, or reproducible
artifact.

The assignment requires a React and Python application that runs on one
developer laptop. It rewards thoughtful CV-research utility, not the number of
models. The local machine already has two vision-capable Ollama models:
`gemma4:12b` and `qwen3.5:9b`.

Official Ollama documentation supports image inputs through the REST chat API,
JSON Schema in the `format` field, Pydantic validation of the returned content,
and `temperature: 0` for more deterministic structured output. Ollama's model
details endpoint also reports whether an installed model has the `vision`
capability.

## Decision

Add one inspector to the existing sample workbench rather than a separate model
zoo or another dashboard.

The API:

- discovers configured local models through Ollama;
- requires the live model to advertise `vision`;
- binds each result to the exact local Ollama digest and model details;
- accepts a small typed task set: scene inventory, road-scene triage, caption
  audit, OCR proposal, and a focused image question;
- passes the task's Pydantic JSON Schema to Ollama and validates the response
  with the same Pydantic model;
- disables the optional reasoning trace so the bounded generation budget is
  reserved for the schema-constrained proposal;
- serializes local vision requests so two large models cannot contend for one
  laptop at the same time;
- returns an explicit `model_proposal` status and never mutates labels,
  captions, tags, annotations, or source images.
- preserves the normalized focused question beside its answer, and rejects a
  caption audit unless it contains exactly one assessment for every supplied
  caption.

The UI:

- compares the two installed VLMs on the same image and task;
- exposes the model digest, quantization, latency, and schema version;
- turns proposed search terms into semantic-search links;
- sends proposed object classes into the existing Grounding DINO → SAM2
  annotation flow;
- downloads a self-contained JSON result for review or experiment records.

## Alternatives considered

- **Add every promising Hugging Face model now.** Florence-2, PaliGemma 2,
  Qwen3-VL-Embedding, and SAM3 have useful roles, but adding them without a
  measured task and action loop would recreate the existing shallow feature
  problem. SAM3 and some model cards are gated; extra weights also conflict with
  the one-laptop scope.
- **Replace SigLIP2 with a generative VLM.** Retrieval embeddings and
  generative inspection solve different problems. The existing Flickr8k
  benchmark currently favors SigLIP2 for retrieval, so replacement would be an
  unmeasured regression.
- **Persist model output directly as ground truth.** VLM OCR and scene details
  produced observed errors in local benchmarks. Automatic promotion would blur
  proposal and decision, making dataset state less trustworthy.
- **Create a new top-level Vision Lab route.** A route without an active sample
  adds selection ceremony and duplicates the image/annotation canvas. The
  sample page is already the inspect-and-act boundary.

## Consequences

Positive:

- A model result has immediate downstream actions instead of ending as prose.
- Every result is reproducible against an exact local model artifact.
- Model hallucinations remain visible and reviewable instead of contaminating
  source data.
- The design extends to future tasks without changing the transport contract.

Negative:

- Only configured, already-pulled Ollama models are usable; the API never
  downloads weights during a request.
- The workbench is image-first and does not pretend Flickr8k supplies video,
  tracking, calibration, depth, or multi-sensor automotive data.
- VLM output remains a proposal and therefore still needs human review.

## Failure modes and mitigations

- Ollama unavailable or model absent: status is `ready: false` with a concrete
  local setup reason.
- Model is text-only: rejected before inference using the documented
  `capabilities` field.
- Malformed structured output: explicit 502; no regex repair or hidden fallback.
- Concurrent local inference: explicit 409 busy response.
- Missing source image: explicit 503; no network fetch.
- Model alias changes to different weights: the response digest changes and the
  exported artifact records it.

## Documentation

- Ollama vision:
  <https://docs.ollama.com/capabilities/vision>
- Ollama structured outputs:
  <https://docs.ollama.com/capabilities/structured-outputs>
- Ollama model details:
  <https://docs.ollama.com/api-reference/show-model-details>
- Ollama thinking control:
  <https://docs.ollama.com/capabilities/thinking>
- Gemma 4 12B model card:
  <https://huggingface.co/google/gemma-4-12B>
- Qwen3.5 9B model card:
  <https://huggingface.co/Qwen/Qwen3.5-9B>
