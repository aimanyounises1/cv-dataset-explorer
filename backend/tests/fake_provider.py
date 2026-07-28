"""Deterministic embedding stand-in for tests: same input, same unit vector,
no torch. Lives under tests/ on purpose — production code has no mock path;
tests inject this explicitly where an encoder is needed."""
import hashlib

import numpy as np


class MockEncoder:
    DIM = 32

    def _vec(self, payload: bytes) -> np.ndarray:
        digest = hashlib.sha256(payload).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        return rng.standard_normal(self.DIM)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (x / norms).astype(np.float32)

    def encode_texts(self, texts, kind: str = "query") -> np.ndarray:
        return self._normalize(np.stack([self._vec(t.encode()) for t in texts]))

    def encode_images(self, images) -> np.ndarray:
        return self._normalize(np.stack([self._vec(im.tobytes()) for im in images]))
