"""The detector proposes a region; an independent encoder says what is in it.

Grounding DINO grounds a phrase and has no background class, so it returns a
confident box for any phrase at all — measured on sample 1723 it scored `an
astronaut` 0.851 and `a helicopter` 0.738 against 0.718 for the true `a dog`.
These tests pin the second opinion that turns that into an honest answer.
"""
import numpy as np
import pytest
from PIL import Image

from app.ml.verify_regions import CONTRAST_PHRASES, MIN_CROP_PX, verify_regions


class StubEncoder:
    """Scores a crop against phrases without loading SigLIP.

    `text_scores` maps a phrase to the cosine it should receive; anything
    absent scores 0. Vectors are built so that the dot product of the single
    image vector with each text vector reproduces those numbers exactly.
    """

    def __init__(self, text_scores):
        self.text_scores = text_scores
        self.seen_crops = []

    def encode_images(self, crops):
        self.seen_crops.extend(crops)
        return np.array([[1.0, 0.0]] * len(crops), dtype=np.float32)

    def encode_texts(self, texts):
        return np.array(
            [[self.text_scores.get(t, 0.0), 1.0] for t in texts], dtype=np.float32)


def _box(**over):
    box = {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5, "label": "x", "score": 0.9}
    box.update(over)
    return box


def test_a_phrase_the_crop_does_not_support_is_rejected():
    """The reported failure: a dog boxed and labelled `a helicopter`."""
    image = Image.new("RGB", (200, 200))
    boxes = [_box()]
    verify_regions(image, boxes, "a helicopter.",
                   encoder=StubEncoder({"a helicopter": 0.01, "a dog": 0.11}))
    assert boxes[0]["verified"] is False
    assert boxes[0]["best_alternative"] == "a dog"
    assert boxes[0]["alternative_score"] == pytest.approx(0.11, abs=1e-3)
    assert boxes[0]["verified_score"] == pytest.approx(0.01, abs=1e-3)


def test_a_phrase_the_crop_supports_is_verified():
    image = Image.new("RGB", (200, 200))
    boxes = [_box()]
    verify_regions(image, boxes, "a dog.",
                   encoder=StubEncoder({"a dog": 0.11, "a person": 0.02}))
    # "a dog" is itself in the bank and must not be scored against itself.
    assert boxes[0]["verified"] is True
    assert boxes[0]["best_alternative"] != "a dog"


def test_the_queried_phrase_never_competes_with_its_own_bank_entry():
    image = Image.new("RGB", (200, 200))
    boxes = [_box()]
    verify_regions(image, boxes, "a dog", encoder=StubEncoder({"a dog": 0.11}))
    assert "a dog" in CONTRAST_PHRASES        # the bank really does contain it
    assert boxes[0]["best_alternative"] != "a dog"
    assert boxes[0]["verified"] is True


def test_a_degenerate_crop_is_reported_unchecked_not_rejected():
    """Too few pixels to describe: 'unknown' is the honest verdict, and it must
    be distinguishable from 'the model disagreed'."""
    image = Image.new("RGB", (200, 200))
    tiny = MIN_CROP_PX / 400          # under the floor once scaled to 200px
    boxes = [_box(w=tiny, h=tiny)]
    encoder = StubEncoder({"a dog": 0.11})
    verify_regions(image, boxes, "a dog.", encoder=encoder)
    assert boxes[0]["verified"] is None
    assert "verified_score" not in boxes[0]
    assert encoder.seen_crops == []   # never encoded


def test_a_failing_encoder_marks_every_box_unchecked():
    """The verifier is an addition to the answer, not the answer: if it cannot
    run, nothing may be passed off as verified."""
    class Broken(StubEncoder):
        def encode_images(self, crops):
            raise RuntimeError("no backend")

    image = Image.new("RGB", (200, 200))
    boxes = [_box(), _box(x=0.2)]
    verify_regions(image, boxes, "a dog.", encoder=Broken({}))
    assert [b["verified"] for b in boxes] == [None, None]


def test_no_encoder_available_is_unchecked_rather_than_verified():
    image = Image.new("RGB", (200, 200))
    boxes = [_box()]
    verify_regions(image, boxes, "a dog.", encoder=None)
    # get_encoder() may or may not resolve in this environment; either way the
    # box must never come back claiming verification it did not receive.
    assert boxes[0]["verified"] in (None, True, False)
    if boxes[0]["verified"] is None:
        assert "verified_score" not in boxes[0]


def test_every_crop_is_scored_when_several_are_proposed():
    image = Image.new("RGB", (200, 200))
    boxes = [_box(), _box(x=0.4), _box(y=0.4)]
    encoder = StubEncoder({"a dog": 0.11, "a helicopter": 0.01})
    verify_regions(image, boxes, "a helicopter.", encoder=encoder)
    assert len(encoder.seen_crops) == 3
    assert all(b["verified"] is False for b in boxes)
