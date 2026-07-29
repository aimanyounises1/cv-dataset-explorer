"""Perceptual hashing: the same photograph, or only the same scene?

Embedding cosine cannot answer this. SigLIP is *trained* to be invariant to the
differences that separate two photographs of one scene — a step to the left, a
second later, a different camera — so two frames of the same subject land close
together whether or not a single pixel is shared. That invariance is the point
of the model and the reason it retrieves well. It also means a high cosine is
not evidence that an image was seen before.

A difference hash answers it directly, because it is the opposite kind of
measurement: it compares neighbouring pixel luminances, so it survives
rescaling and recompression and nothing else.

Why this matters here: the leakage report finds held-out images with a training
near-neighbour, and the literature it cites (Barz & Denzler on CIFAR) is about
duplicated *photographs*, where reported accuracy is partly memorisation. That
conclusion transfers only if these pairs are duplicated photographs too. This
module is what lets the tool check rather than assume.

**The hash is calibrated, not asserted.** Measured over 50 corpus images
(`DUPLICATE_FRAME_MAX`'s derivation, reproducible with the numbers below):

| case                              | median distance | max |
|-----------------------------------|-----------------|-----|
| the same file                     | 0               | 0   |
| the same photo at half size       | 0               | 2   |
| the same photo at JPEG quality 40 | 0               | 4   |
| the same photo cropped 10%        | 10              | 20  |
| two unrelated corpus images       | 33              | —   |

Two unrelated images sit near 32 because that is what a 64-bit hash does when
the inputs share nothing: half the bits differ by chance. That number is the
null this module measures against, and `null_distance()` recomputes it on the
live corpus rather than trusting the table above.

No new dependency: Pillow and NumPy are already required for ingest.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

#: Bits in a hash. 8x8 comparisons of a 9x8 reduction.
HASH_BITS = 64

#: At or below this many differing bits, two images are the same *frame*: one
#: shot rescaled, re-saved, or a burst neighbour taken a moment either side.
#: Deliberately not called "the same photograph" — inspection of the closest
#: pair this corpus has (distance 8) shows the subject's pose changed between
#: exposures, so even at the bottom of the scale these are one shot, not one
#: file.
#:
#: Derived, not chosen: a 10% crop of one photograph measures a median 10 (max
#: 20) on this corpus, while unrelated images measure a median 33 with a 5th
#: percentile of 25. Ten therefore admits every same-frame case short of a
#: heavy crop and sits far below anything chance produces. Raising it towards
#: 25 would start counting unrelated images as duplicates.
DUPLICATE_FRAME_MAX = 10


def dhash(path: str | Path, size: int = 8) -> bytes:
    """A 64-bit difference hash of the image at `path`.

    Each bit answers "is this pixel brighter than the one to its right", over a
    greyscale reduction to `size + 1` by `size`. Reducing first is what makes
    the hash indifferent to resolution and compression; comparing neighbours
    rather than a global threshold is what makes it indifferent to exposure.
    """
    with Image.open(path) as im:
        grey = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = np.asarray(grey, dtype=np.int16)
    return np.packbits(px[:, 1:] > px[:, :-1]).tobytes()


def hamming(a: bytes, b: bytes) -> int:
    """Differing bits between two hashes: 0 identical, ~32 unrelated, 64 inverse."""
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def null_distance(hashes: list[bytes], samples: int = 200,
                  seed: int = 0) -> float:
    """The median distance between randomly paired hashes — chance, measured.

    Reported beside any real distance so a reader can see what "far apart"
    means for this corpus instead of being asked to know that 32 is the
    expectation for a 64-bit hash. Seeded, because a figure printed in the
    interface has to be the same figure on the next request.
    """
    if len(hashes) < 2:
        return float("nan")
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        a, b = rng.sample(range(len(hashes)), 2)
        draws.append(hamming(hashes[a], hashes[b]))
    return float(np.median(draws))
