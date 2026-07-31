"""Independent verification of detector proposals.

Grounding DINO *grounds* a phrase: given an image and a phrase it returns the
region that best matches that phrase. It carries no background class, so it
answers "where is X" and can never answer "X is not here". Measured on sample
1723 — a dog jumping into water — it scored `an astronaut` 0.851, `a
helicopter` 0.738 and `a pizza` 0.737 against 0.718 for the true `a dog`: every
phrase gets a confident box, and the truth placed fourth.

SigLIP 2 scores the same crop the other way round, image against phrases, and
on that box ranked the truth first by an order of magnitude (`a dog` +0.1062,
`a helicopter` +0.0115). The two models fail differently, so the second is
worth asking: the detector proposes a region, the embedder says what is
actually in it.

A proposal is never dropped on the verifier's say-so — it is labelled. Dropping
would replace one model's opinion with another's, and this application's rule
is that model output stays a proposal until a person accepts it. What changes
is that an unverified proposal now arrives marked, with the phrase the verifier
preferred and both scores, so a reviewer can see the disagreement rather than
read `a helicopter 74%` as a finding.
"""
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# What a crop is scored against. These are deliberately broad, and they are the
# content this corpus is actually made of: the README's provenance section
# records that Flickr8k images came from a handful of Flickr hobby groups, so
# people, dogs and outdoor action dominate. A contrast set that did not cover
# the common cases would let any phrase win by default, which is the failure
# being fixed. Extending this list makes verification stricter, never looser.
CONTRAST_PHRASES: tuple[str, ...] = (
    "a dog",
    "a person",
    "a child",
    "a group of people",
    "an animal",
    "a bird",
    "a vehicle",
    "a building",
    "water",
    "grass",
    "snow",
    "sand",
    "a tree",
    "the sky",
    "a road",
    "an indoor room",
)

# The smallest crop worth encoding, in pixels per side. Below this the resize
# into the encoder's input size is mostly interpolation, and the score stops
# describing the region. Such a proposal is returned unscored rather than
# scored badly.
MIN_CROP_PX = 8


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def verify_regions(image, boxes: list[dict], phrase: str,
                   encoder=None) -> list[dict]:
    """Annotate each proposed box with what an independent encoder sees in it.

    `boxes` carry normalized x/y/w/h, as `_Detector.detect` returns them. Each
    box gains:
      `verified`          — the queried phrase beat every contrast phrase
      `verified_score`    — cosine of the crop against the queried phrase
      `best_alternative`  — the contrast phrase that scored highest
      `alternative_score` — that phrase's cosine
    A box the verifier could not judge (no encoder, degenerate crop, or a
    failed encode) gets `verified: None` and no scores, which the API and the
    UI must present as "not checked" rather than as either verdict.
    """
    if not boxes:
        return boxes
    if encoder is None:
        from . import providers

        encoder = providers.get_encoder()
    if encoder is None:      # embeddings absent: the honest answer is "unknown"
        for box in boxes:
            box["verified"] = None
        return boxes

    query = " ".join(phrase.split()).strip(" .")
    # A phrase already in the bank must not compete with itself.
    contrast = [c for c in CONTRAST_PHRASES if c.lower() != query.lower()]
    if not query:
        for box in boxes:
            box["verified"] = None
        return boxes

    width, height = image.size
    crops, indexed = [], []
    for index, box in enumerate(boxes):
        left = int(round(box["x"] * width))
        top = int(round(box["y"] * height))
        right = int(round((box["x"] + box["w"]) * width))
        bottom = int(round((box["y"] + box["h"]) * height))
        if right - left < MIN_CROP_PX or bottom - top < MIN_CROP_PX:
            box["verified"] = None
            continue
        crops.append(image.crop((left, top, right, bottom)))
        indexed.append(index)

    if not crops:
        return boxes

    try:
        image_vectors = _unit(np.asarray(encoder.encode_images(crops), dtype=np.float32))
        text_vectors = _unit(
            np.asarray(encoder.encode_texts([query, *contrast]), dtype=np.float32))
    except Exception:
        # The verifier is an addition to the answer, not the answer. If it
        # cannot run, every box says so explicitly; nothing is silently passed
        # off as verified.
        logger.exception("Region verification failed; proposals returned unchecked")
        for box in boxes:
            box["verified"] = None
        return boxes

    similarities = image_vectors @ text_vectors.T
    for row, index in enumerate(indexed):
        scores = similarities[row]
        query_score = float(scores[0])
        alternative_scores = scores[1:]
        box = boxes[index]
        if alternative_scores.size == 0 or not math.isfinite(query_score):
            box["verified"] = None
            continue
        best = int(np.argmax(alternative_scores))
        best_score = float(alternative_scores[best])
        box["verified"] = bool(query_score > best_score)
        box["verified_score"] = round(query_score, 4)
        box["best_alternative"] = contrast[best]
        box["alternative_score"] = round(best_score, 4)
    return boxes
