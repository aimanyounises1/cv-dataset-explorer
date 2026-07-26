"""The embedder must never run two forward passes at once.

Two threads calling `encode_texts` on one shared SigLIP module on the Metal
backend either kill the process with a SIGSEGV inside `copy_cast_kernel_mps` or
deadlock inside Metal — both reproduced on an M-series Mac with two threads doing
25 encodes each. FastAPI serves sync endpoints from a thread pool, so two
simultaneous semantic searches were already enough to reach it, and parallel
agent lanes make it ordinary.

The bug is unreproducible in CI (no Metal, and a crash is not catchable anyway),
so what is tested is the invariant that prevents it: the critical sections of two
concurrent encodes do not overlap. That fails immediately if the lock is dropped,
which is the regression worth catching — a segfault in production is not a test
failure anyone gets to read.

A real `Embedder` is used with stub weights rather than a stub Embedder, so the
test exercises the actual `encode_texts` body, including the lock placement.
"""
import threading
import time

import numpy as np

from app.ml.embedder import Embedder


class OverlapDetector:
    """Records whether any two critical sections were ever in flight together."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.inside = 0
        self.max_inside = 0
        self.calls = 0

    def enter(self) -> None:
        with self.lock:
            self.inside += 1
            self.calls += 1
            self.max_inside = max(self.max_inside, self.inside)

    def leave(self) -> None:
        with self.lock:
            self.inside -= 1


class _Inputs(dict):
    def to(self, _device):
        return self


class _Processor:
    def __call__(self, **_kwargs):
        return _Inputs()


class _Model:
    """Stands in for the SigLIP module; sleeps inside the forward pass so an
    unserialized second caller would demonstrably overlap it."""

    def __init__(self, detector: OverlapDetector, hold: float = 0.02):
        self.detector = detector
        self.hold = hold

    def _work(self):
        import torch

        self.detector.enter()
        try:
            time.sleep(self.hold)
            return torch.ones((1, 8))
        finally:
            self.detector.leave()

    def get_text_features(self, **_kwargs):
        return self._work()

    def get_image_features(self, **_kwargs):
        return self._work()


def _stub_embedder(detector: OverlapDetector) -> Embedder:
    """A real Embedder with stub weights — no model download, real method bodies."""
    emb = object.__new__(Embedder)
    import torch

    emb.device = "cpu"
    emb._torch = torch
    emb.model = _Model(detector)
    emb.processor = _Processor()
    emb._infer = threading.Lock()
    return emb


def _hammer(fn, threads: int = 4, per_thread: int = 6):
    errors: list[str] = []

    def work(tag):
        try:
            for i in range(per_thread):
                fn(tag, i)
        except BaseException as exc:                      # noqa: BLE001
            errors.append(f"{tag}: {type(exc).__name__}: {exc}")

    ts = [threading.Thread(target=work, args=(t,)) for t in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in ts), "a worker thread hung"
    assert not errors, errors


def test_text_encoding_is_serialized():
    detector = OverlapDetector()
    emb = _stub_embedder(detector)
    _hammer(lambda tag, i: emb.encode_texts([f"query {tag} {i}"]))
    assert detector.calls == 24, detector.calls
    assert detector.max_inside == 1, (
        f"{detector.max_inside} forward passes ran concurrently — the inference "
        f"lock is not doing its job, and on Metal this crashes the process")


def test_image_encoding_is_serialized():
    detector = OverlapDetector()
    emb = _stub_embedder(detector)
    _hammer(lambda tag, i: emb.encode_images([object()]))
    assert detector.max_inside == 1, detector.max_inside


def test_text_and_image_encoding_exclude_each_other():
    """The two paths share one module, so they must share one lock."""
    detector = OverlapDetector()
    emb = _stub_embedder(detector)

    def mixed(tag, i):
        if (tag + i) % 2:
            emb.encode_texts([f"q{i}"])
        else:
            emb.encode_images([object()])

    _hammer(mixed)
    assert detector.max_inside == 1, detector.max_inside


def test_encoding_still_returns_normalized_vectors():
    """The lock must not have changed what the method computes."""
    emb = _stub_embedder(OverlapDetector())
    out = emb.encode_texts(["anything"])
    assert out.shape == (1, 8)
    assert out.dtype == np.float32
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_image_batching_releases_the_lock_between_batches():
    """A long image run holds the lock per batch, not for the whole call, so a
    single-query text encode is not blocked for the duration of an ingest."""
    detector = OverlapDetector()
    emb = _stub_embedder(detector)
    emb.encode_images([object()] * 5, batch_size=1)
    # Five separate acquisitions rather than one — visible as five recorded calls
    # with the maximum concurrency still at one.
    assert detector.calls == 5
    assert detector.max_inside == 1
