# Open-vocabulary vision models: local integration decision

**Research date:** 2026-07-29
**Decision:** benchmark one additional grounder,
`google/owlv2-base-patch16-ensemble`, behind the existing proposal contract.
Do not add a model selector or a model showroom.

## Executive decision

The requested workflow is not one classification problem. It is a sequence of
different contracts:

1. validate that every image can be decoded;
2. describe an image or album and propose vocabulary;
3. ground a text phrase or an exemplar to one or more object instances;
4. refine a selected instance into a mask;
5. compare, search, export, or annotate the accepted object;
6. keep all model output as reviewable proposals.

The current local VLMs can already cover step 2, and SAM2 already covers step 4
(see [local evidence](#local-machine-and-runtime-evidence)). The most useful
new experiment is therefore **OWLv2 as a second implementation of step 3**.
OWLv2 has documented text-conditioned and image-guided detection APIs, and the
official report specifically evaluates rare classes without human box
annotations. Its exact checkpoint is already cached on this machine
([Transformers OWLv2 documentation](https://huggingface.co/docs/transformers/model_doc/owlv2),
[official checkpoint](https://huggingface.co/google/owlv2-base-patch16-ensemble)).

This does **not** mean that OWLv2 names an unknown object by itself. The VLM or
the user proposes a phrase such as `red climbing tool`; OWLv2 tests where that
phrase occurs. With an accepted object crop, image-guided detection can instead
look for visually similar instances without inventing a new class name
([Transformers OWLv2 documentation](https://huggingface.co/docs/transformers/model_doc/owlv2)).

`person-1`, `person-2`, and `person-3` are not semantic classes that a detector
should have learned. They are neutral, local instance identifiers. The correct
flow is: ground every `person`, preserve each box as a separate proposal, let a
visually evidenced referring phrase such as `person in the red coat` narrow the
proposals, then assign stable display identifiers after geometry exists. This
is a design conclusion from the detector contracts above, not a claimed learned
capability.

## Local machine and runtime evidence

The following probes were run in this repository on 2026-07-29. They are the
primary evidence for the local feasibility statements in this note.

```text
system_profiler SPHardwareDataType
  MacBook Pro Mac16,5; Apple M4 Max; 64 GB unified memory

cd backend && uv run python  # importlib.metadata + torch backend probe
  torch 2.13.0
  transformers 5.14.1
  torch.backends.mps.is_available() == True
  torch.cuda.is_available() == False

ollama list / ollama show
  runtime          0.32.0
  gemma4:12b     7.6 GB  Q4_K_M  vision
  qwen3.5:9b     6.6 GB  Q4_K_M  vision
    digest 6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7

live structured pair-comparison probes through the installed Ollama models
  gemma4:12b
    FAILED a two-image array request
    FAILED a sequential two-frame conversation
    both responses treated the input as one frame or reported identical frames
  qwen3.5:9b
    SUCCEEDED only with:
      user(image A) -> assistant acknowledgement -> user(image B + comparison)
    a direct two-image request is not an accepted local contract

backend/.venv/bin/python scripts/validate_pair_vision.py
  PASSED frozen samples 76 and 2259
  source hashes, typed differences, grounding terms for both frames, exact model digest,
  Ollama 0.32.0, adapter v1, and sequential_frames_v1 all passed
  optional full record:
    --write backend/data/reports/pair-vision-validation.json

local Hugging Face snapshots
  IDEA-Research/grounding-dino-tiny
    a2bb814dd30d776dcf7e30523b00659f4f141c71  658 MB on disk
  facebook/sam2.1-hiera-tiny
    de431c4043854a71d8101e17995dfe596bf101a5  149 MB on disk
  google/owlvit-base-patch32
    cbc355fb364588351c5d51c7f74465e8e7ec6f72  586 MB on disk
  google/owlv2-base-patch16-ensemble
    cfd3195ba4ea9592eec887ded089f4c08eff231d  593 MB on disk

not present in the local Hugging Face cache
  microsoft/Florence-2-base-ft
  google/paligemma2-3b-mix-448
  Qwen/Qwen3-VL-4B-Instruct
  facebook/sam3
  vil-uob/sam3-litetext-s0
```

The base dependency file declares `torch>=2.2` and `transformers>=4.56`; the
installed environment is newer, as measured above
([backend requirements](../../backend/requirements.txt)). The current detector
and segmenter are configured with immutable revisions
([configuration](../../backend/app/config.py),
[detector provider](../../backend/app/ml/detect.py),
[segmenter provider](../../backend/app/ml/segment.py)).

The cache sizes above are actual local disk use and are not interchangeable
with published BF16 checkpoint sizes. The latter are reported separately in
the candidate comparison.

## Candidate comparison

| Candidate | Officially documented capability | Published artifact / access | Decision for this application |
| --- | --- | --- | --- |
| **OWLv2 / OWL-ViT** | Both expose zero-shot text-conditioned boxes and image-guided detection. OWLv2 scales OWL-ViT with self-training; its report says the largest recipe used more than 1B examples and improved LVIS rare-class AP from 31.2 to 44.6 for classes with no human box annotations. | OWLv2 base ensemble is Apache-2.0 and its safetensors file is about 620 MB. Both exact base checkpoints are already in the local cache. ([OWLv2 docs](https://huggingface.co/docs/transformers/model_doc/owlv2), [OWLv2 file](https://huggingface.co/google/owlv2-base-patch16-ensemble/blob/main/model.safetensors), [OWL-ViT docs](https://huggingface.co/docs/transformers/model_doc/owlvit)) | **Benchmark OWLv2 next.** It adds an exemplar-query path and directly tests the rare/open-world localization gap while preserving SAM2 and the current annotation contract. OWL-ViT is retained only as its cached predecessor, not exposed as another product choice. |
| **Florence-2 base FT** | The processor documents captioning, dense region captioning, detection, phrase grounding, referring-expression segmentation, region segmentation/category/description, OCR, and open-vocabulary detection. Generated text is parsed by `post_process_generation()` into task outputs such as boxes and labels. | MIT; one 463 MB safetensors file at the inspected revision. ([Transformers docs](https://huggingface.co/docs/transformers/model_doc/florence2), [official file](https://huggingface.co/microsoft/Florence-2-base-ft/blob/main/model.safetensors)) | **Best compact unified follow-up, not the next benchmark.** It duplicates already-running caption/VLM and mask stages, is not cached, and introduces a generative-output parser. Benchmark it only if reducing model hand-offs becomes a measured product need. |
| **PaliGemma 2 3B mix 448** | The mix checkpoint is ready for short/long captioning, OCR, question answering, object detection, and object segmentation; it emits generated text including boxes or segmentation codewords. | Gemma license with manual access acceptance; two BF16 shards total about 6.07 GB. ([official model card](https://huggingface.co/google/paligemma2-3b-mix-448), [official files](https://huggingface.co/google/paligemma2-3b-mix-448/tree/main)) | **Do not add now.** It covers much of Florence's task surface at roughly thirteen times Florence's published weight footprint, is gated, and is not cached. It remains relevant for future task-specific fine-tuning, not as a second general-purpose UI model. |
| **Gemma 4 vision** | Google documents image description, object identification, OCR, visual QA, bounding-box object detection, variable-resolution processing, and multiple images in one prompt. | Gemma 4 12B is Apache-2.0; the official BF16 file is 23.9 GB. A 7.6 GB Q4_K_M `gemma4:12b` vision model is already installed through Ollama on this machine. ([Google vision guide](https://ai.google.dev/gemma/docs/capabilities/vision/image), [official 12B file](https://huggingface.co/google/gemma-4-12B/tree/main), [local evidence](#local-machine-and-runtime-evidence)) | **Use the installed runtime only for its locally verified single-image inspection lanes.** Despite the upstream multi-image documentation, both local pair-input shapes failed and collapsed the evidence to one or identical frames. Do not route pair comparison to Gemma until a versioned regression test passes. Its generated boxes are also not a replacement for a scored detector or mask model. |
| **Qwen VL / Qwen 3.5** | Qwen3-VL 4B documents broad visual recognition and 2D grounding; Qwen3.5 is a natively multimodal family whose vision tower reuses the Qwen3-VL encoder. The official processor contracts accept image inputs, while the model output remains generated text. | Qwen3-VL 4B is Apache-2.0 and its two BF16 shards total about 8.88 GB. The newer `qwen3.5:9b` vision model is already installed locally as a 6.6 GB Q4_K_M Ollama artifact. ([Qwen3-VL card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [Qwen3.5 docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5), [local evidence](#local-machine-and-runtime-evidence)) | **Use the existing Qwen runtime as the capability-tested pair inspector, not a geometry authority.** On this runtime, pair comparison succeeded only as `user(image A) -> assistant acknowledgement -> user(image B + comparison)`; the adapter must own that exact message construction. Geometry must still be verified by a grounder and masks by a segmenter. |
| **SAM3 / SAM3-LiteText** | SAM3 accepts text and/or image exemplars and returns instance and semantic masks for every matching concept; it also supports point/box/mask prompts and video tracking. LiteText keeps the SAM3 prompting interface while replacing the text encoder with a smaller distilled encoder. | `facebook/sam3` is manually gated, license `other`, and its safetensors file is about 3.44 GB. `vil-uob/sam3-litetext-s0` is Apache-2.0, ungated, and about 2.12 GB. Neither is cached here. ([SAM3 card](https://huggingface.co/facebook/sam3), [SAM3 docs](https://huggingface.co/docs/transformers/model_doc/sam3), [LiteText checkpoint](https://huggingface.co/vil-uob/sam3-litetext-s0/tree/main)) | **Strongest later concept-to-mask adapter, but defer it.** It could collapse grounding plus masking, yet both checkpoints are materially larger than cached SAM2 tiny and have no local latency, memory, or quality evidence on this M4 Max. |

### Why OWLv2 wins the next slot

This is a benchmark decision, not a claim that OWLv2 is already better on this
dataset.

- It is the only evaluated candidate that adds a **new interaction primitive**
  immediately: find objects from an accepted exemplar crop
  ([OWLv2 image-guided API](https://huggingface.co/docs/transformers/model_doc/owlv2)).
- Its exact immutable snapshot is already cached, so an HTTP request does not
  need to fetch weights ([local evidence](#local-machine-and-runtime-evidence)).
- It keeps the existing, measured SAM2 correction path intact
  ([segmenter provider](../../backend/app/ml/segment.py)).
- It compares one variable—grounding quality—rather than simultaneously
  replacing captioning, parsing, grounding, and segmentation.

Florence-2 remains the first unified-model comparator because its official
processor covers the most relevant tasks in the smallest published artifact
among the unified candidates above
([Florence-2 docs](https://huggingface.co/docs/transformers/model_doc/florence2),
[Florence-2 file](https://huggingface.co/microsoft/Florence-2-base-ft/blob/main/model.safetensors)).

## Minimal deep-module interface

Agents and UI code must request a capability, never a checkpoint name. One
facade can hide every candidate behind three methods:

```python
class VisionModule(Protocol):
    def inspect(
        self,
        images: Sequence[ImageInput],
        task: InspectTask,
    ) -> InspectionProposal: ...

    def ground(
        self,
        image: ImageInput,
        queries: Sequence[GroundQuery],
    ) -> Sequence[RegionProposal]: ...

    def segment(
        self,
        image: ImageInput,
        prompts: Sequence[MaskPrompt],
    ) -> Sequence[MaskProposal]: ...
```

`inspect()` covers one-image caption/OCR/QA, album batch captioning, and a
provider-gated two-image semantic difference. On this machine, the pair task
routes only to the Qwen adapter and that adapter serializes the locally proven
three-message sequence; Gemma must report the pair capability as unavailable
([local evidence](#local-machine-and-runtime-evidence)). `ground()` accepts
text, a referring phrase, or an accepted exemplar crop. `segment()` accepts a
point, box, text concept, or exemplar, although the current SAM2 provider
supports only point/box prompts
([SAM2 documentation](https://huggingface.co/docs/transformers/model_doc/sam2))
and a future SAM3 provider can support all four
([SAM3 documentation](https://huggingface.co/docs/transformers/model_doc/sam3)).

Every result must use provider-neutral geometry and include:

- source image digest and sample ID;
- capability and normalized input query;
- model ID plus immutable revision or Ollama digest;
- provider and runtime version;
- box, polygon, or mask in source-image coordinates;
- the provider's score with an explicit score kind;
- elapsed time and a proposal ID;
- `proposal` state, never implicit acceptance.

Those fields are an integration requirement. A provider that cannot supply a
calibrated mask-quality score records it as unavailable; a generation
probability must not be relabeled as IoU.

## Sequential album-agent workflow

```text
album selected
  -> deterministic integrity scan
  -> batch inspect/caption with checkpointed progress
  -> user or VLM proposes rare concepts
  -> ground concepts with current Grounding DINO and candidate OWLv2
  -> reviewer selects an instance
  -> SAM2 produces/corrects the mask
  -> accepted object becomes available to masked search, compare, and export
```

The agent may schedule and summarize these stages, but it must not silently
write accepted annotations. A failure or pause must preserve completed image
results, exact provider provenance, and the next resumable item.

Segmentation is invoked only when an object-level artifact creates value:

- isolate foreground for masked-object retrieval;
- compare the same object across two images;
- export a cutout or training mask;
- measure object area/position;
- hide a background with the shipped transparent-alpha cutout, or pursue
  background replacement only through a separately evaluated future workflow;
- create a reviewed annotation.

Running segmentation over every object in every album by default would create
cost and review debt without a user decision; this is a product conclusion,
not a model capability claim.

## Difference and corruption are separate

No VLM should decide whether an image file is corrupted. Pillow documents that
`Image.verify()` checks file contents for breakage without decoding pixels, and
that `Image.load()` performs the actual pixel load
([Pillow Image documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.verify)).
The integrity stage should therefore record open/verify/load status, dimensions,
mode, byte length, and a cryptographic digest before any learned model runs.

Pairwise comparison then has three explicit modes:

1. **Exact identity:** source-byte SHA-256 equality.
2. **Measured visual difference:** dimensions plus aligned pixel/structural
   metrics when alignment is valid; OpenCV documents PSNR and SSIM as image
   similarity measures
   ([OpenCV PSNR/SSIM tutorial](https://docs.opencv.org/4.11.0/d5/dc4/tutorial_video_input_psnr_ssim.html)).
3. **Semantic difference proposal:** call `inspect([a, b],
   InspectTask.SEMANTIC_DIFF)` through the installed Qwen adapter. The facade
   must turn that neutral call into
   `user(image A) -> assistant acknowledgement -> user(image B + comparison)`,
   because that is the only message shape that passed the local structured
   probe ([local evidence](#local-machine-and-runtime-evidence)). Do not route
   this task to the installed Gemma runtime: both its two-image array and
   sequential-frame probes falsely reduced the evidence to one or identical
   frames, even though Google's upstream guide documents multiple-image input
   ([Google Gemma image guide](https://ai.google.dev/gemma/docs/capabilities/vision/image)).

Semantic difference output is explanatory evidence, not a corruption verdict
or a pixel metric.

## OWLv2 acceptance gates

The cached checkpoint does not become the default until all gates pass.

1. **Immutable, offline artifact.** Configure
   `google/owlv2-base-patch16-ensemble@cfd3195ba4ea9592eec887ded089f4c08eff231d`;
   resolve it with `local_files_only=True`; prove that a missing snapshot
   leaves Grounding DINO fully usable. The revision is from the
   [local cache probe](#local-machine-and-runtime-evidence).
2. **Frozen evaluation set.** Version at least 100 reviewer-annotated
   query-image pairs before looking at candidate results. Include rare
   noun phrases, attribute/referring phrases, repeated people, accepted
   exemplar crops, and explicit no-match queries.
3. **Localization improvement.** Against the current Grounding DINO snapshot,
   rare-query recall at IoU >= 0.5 must improve by at least 8 percentage
   points and the paired bootstrap 95% confidence interval for the improvement
   must exclude zero.
4. **No regression on ambiguity.** Referring-instance top-1 accuracy and
   no-match false-positive rate may each regress by no more than 2 percentage
   points.
5. **End-to-end value.** After sending both providers' selected boxes through
   the same SAM2 snapshot, accepted-without-edit rate may not regress by more
   than 2 percentage points, and median correction time may not increase by
   more than 10%.
6. **Laptop envelope.** On the M4 Max probe above, record cold load, p50/p95
   warm latency, peak resident memory, failure rate, Torch/Transformers
   versions, device, and image/query batch size. Require warm p95 <= 2 seconds,
   incremental peak RSS <= 2 GB, and fewer than 1% inference failures over the
   frozen set.
7. **One contract.** Both grounders return the same `RegionProposal`; no
   OWLv2-specific conditional reaches the agent, REST schema, or UI.
8. **Truthful provenance.** The accepted annotation preserves the original
   query, original model label, exact model revision, score and score kind,
   source geometry, reviewer taxonomy, and downstream SAM2 revision as
   separate fields.

If OWLv2 fails any gate, keep the existing Grounding DINO -> SAM2 path and
publish the measured result. Do not compensate with prompt lists, regex label
rewrites, or dataset-specific thresholds.

## Sources and reproducibility

Primary external sources used:

- [Transformers: OWLv2](https://huggingface.co/docs/transformers/model_doc/owlv2)
- [Transformers: OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit)
- [Transformers: Florence-2](https://huggingface.co/docs/transformers/model_doc/florence2)
- [Microsoft Florence-2 base FT checkpoint](https://huggingface.co/microsoft/Florence-2-base-ft)
- [Google PaliGemma 2 3B mix 448 checkpoint](https://huggingface.co/google/paligemma2-3b-mix-448)
- [Google Gemma image-understanding guide](https://ai.google.dev/gemma/docs/capabilities/vision/image)
- [Qwen3-VL 4B Instruct checkpoint](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [Transformers: Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Meta SAM3 checkpoint](https://huggingface.co/facebook/sam3)
- [Transformers: SAM3](https://huggingface.co/docs/transformers/model_doc/sam3)
- [SAM3-LiteText S0 checkpoint](https://huggingface.co/vil-uob/sam3-litetext-s0)
- [Transformers: SAM2](https://huggingface.co/docs/transformers/model_doc/sam2)
- [Pillow Image API](https://pillow.readthedocs.io/en/stable/reference/Image.html)
- [OpenCV PSNR/SSIM tutorial](https://docs.opencv.org/4.11.0/d5/dc4/tutorial_video_input_psnr_ssim.html)

Hub revisions and file sizes were inspected with
`huggingface_hub.HfApi().model_info(model_id, files_metadata=True)`. Local
versions, MPS availability, Ollama manifests, and cache revisions were measured
with the commands captured in
[Local machine and runtime evidence](#local-machine-and-runtime-evidence).
