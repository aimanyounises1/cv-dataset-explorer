"""Retrieval embedding providers: one active provider supplies BOTH the query
encoder and the vector index, and the two are never mixed across providers.

- ``qwen3_vl`` — Qwen/Qwen3-VL-Embedding-2B through sentence-transformers,
  in-process on MPS/CUDA/CPU. Ollama serves the language models only; it cannot
  host a multimodal embedding model, so this never goes near it.
- ``siglip2`` — the original SigLIP 2 stack (`app.ml.embedder`), byte-identical
  behavior, keeping its original flat ``data/embeddings/`` layout so an
  existing install keeps working untouched.

Resolution is lazy and visible: the preferred provider (CVDE_EMBED_PROVIDER,
default qwen3_vl) is probed cheaply — imports, cached weights, index manifest —
and falls back to siglip2 with a *named* reason that the status API surfaces.
A provider whose model later fails to load flips the same way at query time.

Encoders duck-type ``app.ml.embedder.Embedder``: ``encode_texts(texts)`` and
``encode_images(images)`` returning L2-normalized float32 rows, one lock per
model because Metal cannot run one module concurrently (see embedder.py).
The optional ``kind`` kwarg ("query" | "document") exists for providers whose
model cards prescribe different prompts per side; Qwen3-VL-Embedding ships a
single default prompt ("Represent the user's input.") applied to every input,
so both kinds encode identically there, and SigLIP has no prompt at all.
"""
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
# The single prompt Qwen3-VL-Embedding's sentence-transformers config applies to
# every input. Hashed into the manifest so an index encoded under a different
# prompt can never serve queries encoded under this one.
QWEN_PROMPT = "Represent the user's input."
PROMPT_VERSION = hashlib.sha1(QWEN_PROMPT.encode()).hexdigest()[:8]

_INSTALL_HINT = "uv pip install --python .venv/bin/python -r requirements-qwen.txt"
_INGEST_HINT = "python -m app.ingest --provider qwen3_vl"

# A failed model load is cached briefly so every request doesn't retry a
# multi-GB load — same policy and constant as app.ml.embedder.
_RETRY_AFTER_S = 120.0


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


# -- encoders -----------------------------------------------------------------

class QwenEncoder:
    """Qwen3-VL-Embedding through the official sentence-transformers path."""

    def __init__(self, model_id: str):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading %s (in-process, sentence-transformers)", model_id)
        self.model_id = model_id
        self.model = SentenceTransformer(model_id)
        # One lock per model module on Metal — see app.ml.embedder's docstring
        # for the segfault this prevents.
        self._infer = threading.Lock()

    def _encode(self, batch, batch_size: int) -> np.ndarray:
        with self._infer:
            out = self.model.encode(
                batch, batch_size=batch_size,
                convert_to_numpy=True, show_progress_bar=False)
        return _normalize(np.asarray(out))

    def encode_texts(self, texts, kind: str = "query") -> np.ndarray:
        return self._encode(list(texts), batch_size=32)

    def encode_images(self, images, batch_size: Optional[int] = None) -> np.ndarray:
        return self._encode(list(images), batch_size=batch_size or config.QWEN_EMBED_BATCH)


# -- manifests ----------------------------------------------------------------

def read_manifest(emb_dir: Path) -> Optional[dict]:
    try:
        return json.loads((emb_dir / MANIFEST_NAME).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_manifest(emb_dir: Path, **fields) -> None:
    """Written last, atomically: its presence with status=complete is the
    commit marker for the whole index build."""
    tmp = emb_dir / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(fields, indent=1))
    os.replace(tmp, emb_dir / MANIFEST_NAME)


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """np.save via a temp file + rename so an interrupted build can never leave
    a half-written array that later loads."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _index_files_present(emb_dir: Path) -> bool:
    return ((emb_dir / "sample_ids.npy").exists()
            and (emb_dir / "image_embeddings.npy").exists())


def manifest_problem(emb_dir: Path, model_id: str) -> Optional[str]:
    """Why this provider's index cannot be served, or None when it can."""
    if not _index_files_present(emb_dir):
        return f"index not built — run `{_INGEST_HINT}`"
    m = read_manifest(emb_dir)
    if m is None or m.get("status") != "complete":
        return (f"index incomplete (interrupted build) — re-run `{_INGEST_HINT}`")
    if m.get("model_id") != model_id:
        return (f"index was built with {m.get('model_id')}; config wants "
                f"{model_id} — re-run `{_INGEST_HINT}`")
    if m.get("prompt_version") != PROMPT_VERSION:
        return (f"index was encoded under a different prompt "
                f"({m.get('prompt_version')} vs {PROMPT_VERSION}) — re-run "
                f"`{_INGEST_HINT}`")
    return None


# -- per-provider descriptors -------------------------------------------------

def provider_model_id(name: str) -> str:
    return {"qwen3_vl": config.QWEN_EMBED_MODEL,
            "siglip2": config.EMBED_MODEL}[name]


def _weights_cached(model_id: str) -> bool:
    from huggingface_hub.constants import HF_HUB_CACHE

    return (Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}").exists()


def _probe(name: str) -> Optional[str]:
    """Cheap availability check — no model load, safe on any request path."""
    emb_dir = config.emb_dir_for(name)
    if name == "siglip2":
        # Original semantics: the flat index existing is what "semantic search
        # available" has always meant; encoder failures still degrade at query
        # time exactly as before.
        if not _index_files_present(emb_dir):
            return "embeddings not built — run `python -m app.ingest`"
        return None
    if name == "qwen3_vl":
        try:
            import sentence_transformers  # noqa: F401
            import torchvision  # noqa: F401
        except ImportError:
            return f"provider stack not installed — `{_INSTALL_HINT}`"
        if not _weights_cached(config.QWEN_EMBED_MODEL):
            return (f"model weights not downloaded — run `{_INGEST_HINT}` "
                    "(first run downloads ~4 GB, local afterwards)")
        return manifest_problem(emb_dir, config.QWEN_EMBED_MODEL)
    return f"unknown provider {name!r}"


def _chain(preferred: str) -> list[str]:
    return [preferred] + (["siglip2"] if preferred != "siglip2" else [])


def _dim_of(name: str, emb_dir: Path) -> Optional[int]:
    m = read_manifest(emb_dir)
    if m and isinstance(m.get("dim"), int):
        return m["dim"]
    try:
        return int(np.load(emb_dir / "image_embeddings.npy", mmap_mode="r").shape[1])
    except (OSError, ValueError):
        return None


# -- resolution state ---------------------------------------------------------

@dataclass
class ProviderState:
    preferred: str
    active: Optional[str]            # None => keyword-only degradation
    model_id: Optional[str]
    dim: Optional[int]
    index_ready: bool
    fallback_reason: Optional[str]   # why the preferred provider is not active
    sim_floor: Optional[float] = None
    reasons: dict = field(default_factory=dict)


_lock = threading.Lock()
_state: Optional[ProviderState] = None
_encoders: dict[str, object] = {}
_hard_failed: dict[str, tuple[float, str]] = {}   # name -> (monotonic, reason)


def _resolve_locked() -> ProviderState:
    preferred = config.EMBED_PROVIDER
    reasons: dict[str, str] = {}
    active = None
    for name in _chain(preferred):
        failed = _hard_failed.get(name)
        if failed and time.monotonic() - failed[0] < _RETRY_AFTER_S:
            reasons[name] = failed[1]
            continue
        problem = _probe(name)
        if problem is None:
            active = name
            break
        reasons[name] = problem
    fallback_reason = None
    if active != preferred:
        parts = [f"{n}: {r}" for n, r in reasons.items()]
        fallback_reason = "; ".join(parts) if parts else None
    emb_dir = config.emb_dir_for(active) if active else None
    manifest = read_manifest(emb_dir) if emb_dir else None
    return ProviderState(
        preferred=preferred,
        active=active,
        model_id=provider_model_id(active) if active else None,
        dim=_dim_of(active, emb_dir) if active else None,
        index_ready=active is not None,
        fallback_reason=fallback_reason,
        sim_floor=(manifest or {}).get("sim_floor_p10"),
        reasons=reasons,
    )


def resolve() -> ProviderState:
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                _state = _resolve_locked()
                logger.info("Retrieval provider: preferred=%s active=%s%s",
                            _state.preferred, _state.active,
                            f" ({_state.fallback_reason})"
                            if _state.fallback_reason else "")
    return _state


def load_encoder_for(name: str):
    """Force-load a specific provider's encoder (ingest / benchmarks).
    Raises on failure — callers there want the real error, not a fallback."""
    if name in _encoders:
        return _encoders[name]
    if name == "qwen3_vl":
        enc = QwenEncoder(config.QWEN_EMBED_MODEL)
    elif name == "siglip2":
        from .embedder import Embedder
        enc = Embedder()
    else:
        raise ValueError(f"unknown provider {name!r}")
    _encoders[name] = enc
    return enc


def get_encoder():
    """The active provider's encoder, or None (=> keyword-only degradation).

    Drop-in for ``app.ml.embedder.get_embedder`` — same duck type, same
    None-on-unavailable contract — but provider-aware. A model that fails to
    load here flips resolution to the fallback with a named reason and
    invalidates the index cache so vector spaces cannot mix mid-flip.
    """
    global _state
    st = resolve()
    if st.active is None:
        return None
    if st.active == "siglip2":
        from .embedder import get_embedder
        return get_embedder()
    try:
        return load_encoder_for(st.active)
    except Exception as exc:
        reason = f"{st.active} failed to load: {exc}"
        logger.warning("%s — falling back", reason)
        with _lock:
            _hard_failed[st.active] = (time.monotonic(), reason)
            _state = None
        from .index import invalidate_index
        invalidate_index()
        return get_encoder()


# -- conveniences used by index/hubness/fingerprints/status -------------------

def active_provider() -> Optional[str]:
    return resolve().active


def active_emb_dir() -> Path:
    st = resolve()
    return config.emb_dir_for(st.active) if st.active else config.EMB_DIR


def active_model_id() -> str:
    """Provenance value for saved views and export manifests. Under siglip2
    this is exactly the historical CVDE_EMBED_MODEL string, so views saved
    before providers existed stay recognized."""
    return resolve().model_id or config.EMBED_MODEL


def invalidate_providers() -> None:
    """Force re-resolution on next access (admin/reload, tests)."""
    global _state
    with _lock:
        _state = None
        _encoders.clear()
        _hard_failed.clear()
