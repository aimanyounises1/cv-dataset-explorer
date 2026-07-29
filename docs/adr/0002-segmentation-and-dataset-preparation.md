# ADR-0002: Segmentation models and reversible dataset preparation

## Status

Accepted as the model-selection and integration boundary. The accepted-mask
cutout/export is shipped; SAM 3 and full transform/copy-paste workflows remain
deferred until their acceptance gates below are measured.

## Problem

The sample workbench needs to help a computer-vision engineer move from an
image-level observation to a trustworthy object-level artifact:

1. propose what may be present;
2. locate the intended instance;
3. produce and refine its mask;
4. resolve its class in an explicit taxonomy;
5. accept or reject the proposal;
6. use the accepted object for retrieval, review, and export.

A second model is useful only if it improves that loop. A model catalogue,
unreviewed auto-labels, or a transform preview with no reproducible export would
increase surface area without satisfying the assignment.

The application must also remain local, runnable on a normal developer laptop,
and honest about which artifacts are already present. Model weights are never
downloaded by an HTTP request.

## Evidence gathered

The candidates below were checked against their official Hugging Face model
cards or Transformers documentation on 2026-07-29. Artifact size is the largest
single published PyTorch/Safetensors weight file, not the sum of duplicate
serialization formats.

| Candidate | Documented capability | Largest weight | Access / license | Fit |
| --- | --- | ---: | --- | --- |
| `facebook/sam2.1-hiera-tiny` | Point/box/mask-prompted image and video segmentation | 156 MB | Open; Apache-2.0 | **Current default.** Cached locally and measured at about 72 ms per warm mask on the reference M4 Max. It refines a known instance but does not name or discover it. |
| `facebook/sam3` | Text or image-exemplar concept prompts; detects and returns semantic and instance masks for every match; point/box/mask prompts and video tracking | 3.45 GB | Manually gated; model card license is `other` | **Best future generalist adapter.** It can collapse the Grounding DINO → SAM2 proposal chain, but it is 22× the current segmenter's weight, gated, uncached, and unbenchmarked on this laptop. |
| `CIDAS/clipseg-rd64-refined` | Zero/one-shot binary segmentation from a text or image prompt | 603 MB | Open; Apache-2.0 | Useful lightweight research comparator, not an annotation default: it produces a binary concept map rather than distinct, stable object instances and is larger than the current SAM2 checkpoint. |
| `google/owlv2-base-patch16-ensemble` | Zero-shot text-conditioned boxes and image-guided one-shot detection | 620 MB | Open; Apache-2.0 | **Next local grounding benchmark.** It is already cached and can test rare phrases or an accepted exemplar crop without changing the current SAM2, annotation, or review contracts. |
| `microsoft/Florence-2-base-ft` | Captioning, detection, dense region captioning, phrase grounding, referring-expression and region-to-segmentation, region classification, and OCR | 463 MB | Open; MIT | **Best compact unified follow-up, not the next benchmark.** One documented processor keeps generated labels, boxes, and polygons in one task contract. It can propose geometry and meaning together, but remains a generative proposal model rather than a calibrated mask-quality judge. |
| `facebook/mask2former-swin-tiny-coco-instance` | Fixed-taxonomy instance segmentation; the architecture also supports semantic and panoptic tasks | 190 MB | Ungated; Hub license metadata is `other` | Useful when a project's fixed label space matches its training data. Do not treat it as permissively licensed without review; it also does not solve arbitrary Flickr8k concept prompts. |
| `shi-labs/oneformer_ade20k_swin_tiny` | Task-conditioned semantic, instance, and panoptic segmentation | 203 MB | Open; MIT | Useful for evaluating a known scene taxonomy, not for general open-vocabulary annotation. |
| `vil-uob/sam3-litetext-s0` | Text-prompted concept detection and instance masks with the SAM3 interface | 2.12 GB | Ungated; Apache-2.0 metadata | More accessible than full SAM3, but still 4.6× Florence's weight and has no official Mac/MPS benchmark. It also requires newer Transformers support than the repository's declared 4.56 floor. |

The installed Transformers 5.14.1 exposes `Sam3Model`, `Sam3Processor`, and
`Sam3TrackerModel`, so a future adapter can use the documented API on this
machine. The repository currently declares `transformers>=4.56`; SAM3 support
was added later, while Florence-2 is documented in 4.56. No Florence-2 or SAM3
artifact is present in the local Hugging Face cache.

The Hub metadata inspected for the leading candidates resolved to immutable
commits `cfd3195ba4ea9592eec887ded089f4c08eff231d`
(`google/owlv2-base-patch16-ensemble`),
`f6c1a25888ffc1d945ee8a1a77ac833c7303d46e`
(`microsoft/Florence-2-base-ft`),
`3c879f39826c281e95690f02c7821c4de09afae7` (`facebook/sam3`), and
`b09766e54f5d2eba021119ec7feff13e74c0f8fc`
(`vil-uob/sam3-litetext-s0`). These are research evidence, not configured
runtime dependencies.

## Decision

### Runtime model roles

Keep the measured local chain as the production path:

```text
local VLM proposal
  -> Grounding DINO text-conditioned boxes
  -> user selects the intended instance
  -> SAM 2.1 tiny mask
  -> user corrects label/parent and accepts
  -> persisted mask + exact model revision
  -> masked-object retrieval / review / export
```

This is not three models voting on ground truth. Each stage has one bounded
responsibility:

- the VLM proposes vocabulary and inspection hypotheses;
- Grounding DINO localizes candidate instances and returns confidence;
- SAM2 refines selected geometry into a mask and returns predicted IoU;
- the reviewer owns the taxonomy decision and promotion to an accepted
  annotation.

The accepted segment already has an operational use: the backend crops the
persisted mask, composites non-object pixels onto a neutral background, embeds
that object crop, optionally combines it with the accepted leaf-label vector,
and ranks it against the full-image index. The UI must call this **masked-object
search**, not claim that a dedicated object-patch index exists.

OWLv2 is the next provider to benchmark behind the existing region-proposal
contract. It adds a documented image-guided path: an accepted cutout can become
an exemplar for finding visually similar rare instances, after which the
existing SAM2 path produces a reviewable mask. It is not promoted until it
passes the frozen quality, no-match, latency, memory, provenance, and
accepted-without-edit gates in
[ADR-0003](0003-sequential-inspection-runs-and-pair-comparison.md) and the
[model research note](../research/2026-07-29-open-vocabulary-vision-models.md).

Florence-2-base-ft remains the later unified-provider comparator, not a
mandatory dependency. Its benchmark should bind one Flickr8k caption to its returned
phrase grounding and polygon, compare it with Grounding DINO → SAM2, and expose
box overlap, mask overlap, original phrase, canonical class, and both models'
provenance. It is accepted only when all of these gates pass:

1. an immutable 40-character Hub revision is configured and cached ahead of
   time;
2. cold load, warm latency, peak resident memory, and failure rate are measured
   on this Mac rather than inferred from CUDA/CPU examples;
3. a stratified human-reviewed corpus sample demonstrates better
   accepted-without-edit rate or shorter median correction time without worse
   mask IoU, Dice, or boundary F-score;
4. its parsed phrase, box, and polygon remain one proposal identity through the
   reviewer;
5. absence of the checkpoint leaves the current workflow fully functional.

Florence-2 does not document a calibrated mask-quality score. Its output must
therefore store mask confidence as unavailable, never copy a generation score
or reuse SAM2's `predicted_iou` field.

SAM 3 remains the strongest future generalist architecture because its concept
prompt natively discovers every matching instance. Its adapter is accepted only
when all of these additional gates pass:

1. the user has explicitly accepted the checkpoint's access and license terms;
2. an immutable 40-character Hub revision is configured and cached ahead of
   time;
3. cold load, warm latency, peak resident memory, and mask quality are measured
   on this machine;
4. a corpus sample demonstrates better instance discovery or fewer corrections
   than Grounding DINO → SAM2;
5. the existing point/box correction workflow and provenance contract remain
   available;
6. absence of the checkpoint leaves the current workflow fully functional.

CLIPSeg may be benchmarked only as a deliberately simpler binary concept-mask
comparator. SAM3-LiteText may be benchmarked after Florence-2 only if the
concept-to-instance behavior justifies its larger local footprint and newer
Transformers floor.

### Annotation responsibility

“Agent responsibility” means a visible, replayable proposal chain, not silent
mutation. Every automated result must expose:

- task and input sample;
- model ID and immutable revision/digest;
- prompt or concept query;
- proposal geometry and confidence;
- mask prompt and segmenter-predicted IoU;
- taxonomy leaf and parent chosen at acceptance;
- the shipped boundary: an inference result is visibly a transient `proposal`,
  and only the reviewer's **Accept & save** action creates an `accepted`
  annotation. That action persists the exact reviewed PNG; it never reruns
  SAM2 and substitutes a new inference result.

The database preserves the accepted SAM2 mask, prompt, model revision, taxonomy
node, and predicted IoU. When the mask began from a detector box it also
preserves Grounding DINO's exact revision, query, original label, canonical
proposal label, confidence, and source geometry. Accepted taxonomy remains a
separate reviewer-owned field, so disagreement survives in the exported record.
Manual point/box masks correctly have no detector source. Detector evidence is
not accepted from client-authored JSON: `/api/detect` returns a short-lived,
server-authenticated proposal token, and mask acceptance resolves the persisted
evidence from that token after checking its sample and box.
`POST /api/segment` likewise returns a short-lived authenticated preview token
covering the exact PNG digest, source-image digest, prompt, dimensions, model
revision, predicted IoU, bounding box, and area fraction. Acceptance requires
that token and the reviewed data URL, rejects changed pixels, prompt geometry,
source bytes, expired evidence, or a different sample, and then stores those
reviewed bytes without another model call.

Post-acceptance `needs_fix` / `rejected` mask states are deliberately deferred;
the current editor supports correction before acceptance and explicit deletion
afterwards.
An accepted mask exports as one consistently named ZIP package. It contains
the original binary mask, a tight RGBA object cutout, and a JSON manifest. The
cutout is derived with Pillow's documented `Image.getbbox`, `Image.crop`, and
`Image.putalpha` APIs; the accepted mask is the alpha channel and remains the
geometry authority. The manifest binds the source-image, mask, and cutout byte
lengths and SHA-256 values to the stored annotation and model provenance.
Neither export nor search mutates the source image. Masked search deliberately
keeps its neutral-background RGB composite because that input was designed for
the existing embedding model; transparent cutout export is a separate artifact
contract.

### Reframing and transforms

Dataset preparation is a separate, non-destructive workflow. It must never
overwrite Flickr8k source images or transform pixels without applying the same
geometry to boxes and masks.

The smallest useful transform recipe is:

- resize with explicit `contain`, `cover`, or `stretch` policy;
- crop or pad to a target aspect ratio;
- rotate or flip;
- choose and record interpolation;
- optionally apply a documented photometric operation such as CLAHE;
- preview before export;
- export derived images, transformed annotations, and a manifest containing the
  source checksum, ordered operations, parameters, random seed where relevant,
  library versions, and output checksum.

The accepted-mask cutout is now the first non-destructive preparation artifact.
It can later be placed into another frame by a deterministic, separately
reviewed copy-paste recipe. That future recipe must preserve source/destination
digests, placement, scale, interpolation, alpha policy, and transformed
annotations. It is not called generative replacement.

OpenCV documents `resize`, `warpAffine`, `warpPerspective`, histogram
equalization, and CLAHE. It also documents that camera undistortion requires a
camera matrix and distortion coefficients. Flickr8k does not provide those
calibration inputs, so a generic “undistort” button would be scientifically
invalid and is excluded.

Stretching is allowed only as an explicit policy with an aspect-ratio
distortion warning. For most training exports, letterbox/pad or crop is the
safer default.

Adding `opencv-python-headless` merely to duplicate Pillow resize is rejected.
OpenCV becomes an optional dependency only when the shipped recipe uses an
OpenCV-specific operation and produces the complete derived-data artifact
described above.

OpenCV also documents `seamlessClone` for blending a selected source region into
a destination and `inpaint` for reconstructing a masked region from nearby
boundary pixels. Those are distinct contracts: seamless cloning can support
controlled copy-paste augmentation, while classical inpainting is useful for
small restoration/removal defects and must not be presented as semantic object
generation. Photorealistic prompt-based object addition or replacement remains
a future, separately evaluated generative provider.

## UX consequence

The professional interaction is a deterministic workbench rail:

```text
Inspect -> Propose -> Localize -> Segment -> Review -> Search / Export
```

Each stage displays real state, model provenance, and a next action. Decorative
agents, jumping blobs, or random dynamic flowcharts are excluded because they
make asynchronous work look active without explaining what changed. Animation
is appropriate only for an actual transition such as model loading, inference,
or export progress.

## Why this meets the assignment

- **Useful to CV researchers:** object-level search, correction, taxonomy, and
  reproducible derived-data preparation are actionable tasks.
- **Usable:** proposals are previewed on the image and require one explicit
  acceptance decision.
- **Sound frontend/backend design:** typed contracts separate proposal,
  inference, acceptance, and export.
- **Generic and expandable:** a future `sam3` provider can replace discovery
  and mask generation without changing annotation or review semantics.
- **Local and reasonable:** the current path uses cached models and a measured
  laptop budget; speculative checkpoints stay optional.

## Official references

- SAM 3 Transformers documentation:
  <https://huggingface.co/docs/transformers/en/model_doc/sam3>
- SAM 3 model card:
  <https://huggingface.co/facebook/sam3>
- SAM 2 Transformers documentation:
  <https://huggingface.co/docs/transformers/en/model_doc/sam2>
- CLIPSeg Transformers documentation:
  <https://huggingface.co/docs/transformers/model_doc/clipseg>
- Florence-2 model card:
  <https://huggingface.co/microsoft/Florence-2-base-ft>
- Florence-2 Transformers 4.56 documentation:
  <https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/florence2>
- SAM3-LiteText Transformers documentation:
  <https://huggingface.co/docs/transformers/model_doc/sam3_lite_text>
- Mask2Former Transformers documentation:
  <https://huggingface.co/docs/transformers/model_doc/mask2former>
- OneFormer Transformers documentation:
  <https://huggingface.co/docs/transformers/main/model_doc/oneformer>
- OpenCV geometric transforms:
  <https://docs.opencv.org/4.13.0/da/d6e/tutorial_py_geometric_transformations.html>
- Pillow image crop, bounds, and alpha APIs:
  <https://pillow.readthedocs.io/en/stable/reference/Image.html>
- OpenCV seamless cloning:
  <https://docs.opencv.org/4.11.0/df/da0/group__photo__clone.html>
- OpenCV inpainting:
  <https://docs.opencv.org/5.0/main_modules/photo_inpaint.html>
- OpenCV histogram equalization:
  <https://docs.opencv.org/master/d4/d1b/tutorial_histogram_equalization.html>
- OpenCV CLAHE:
  <https://docs.opencv.org/master/d2/d74/tutorial_js_histogram_equalization.html>
- OpenCV camera calibration:
  <https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html>
