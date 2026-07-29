"""Retrieval embedding providers: one active provider supplies BOTH the query
encoder and the vector index, and the two are never mixed across providers.

- ``qwen3_vl`` — Qwen/Qwen3-VL-Embedding-2B through sentence-transformers,
  in-process on MPS/CUDA/CPU. Ollama serves the language models only; it cannot
  host a multimodal embedding model, so this never goes near it.
- ``siglip2`` — the original SigLIP 2 stack (`app.ml.embedder`), keeping its
  flat ``data/embeddings/`` layout but requiring the same immutable manifest
  contract as every other provider. Legacy arrays are never trusted in place.

Resolution is lazy and visible: the preferred provider (CVDE_EMBED_PROVIDER,
default siglip2) is probed cheaply — imports, cached weights, index manifest —
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
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .. import config

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 2
# The single prompt Qwen3-VL-Embedding's sentence-transformers config applies to
# every input. Hashed into the manifest so an index encoded under a different
# prompt can never serve queries encoded under this one.
QWEN_PROMPT = "Represent the user's input."
PROMPT_VERSION = hashlib.sha1(QWEN_PROMPT.encode()).hexdigest()[:8]

_INSTALL_HINT = "uv pip install --python .venv/bin/python -r requirements-qwen.txt"
_QWEN_INGEST_HINT = "python3 -m app.ingest --provider qwen3_vl"
_SIGLIP_INGEST_HINT = "python3 -m app.ingest --provider siglip2"

# The Hub commit binds every byte in a snapshot, including the weights. This
# second digest is intentionally narrower and human-auditable: it says which
# architecture, tokenizer and image preprocessing configuration produced the
# vectors, without hashing multi-GB weight files on every process start.
_CONFIG_SUFFIXES = {
    ".json", ".txt", ".model", ".py", ".jinja", ".tiktoken",
}
_FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# A failed model load is cached briefly so every request does not retry a
# multi-GB load.
_RETRY_AFTER_S = 120.0


def require_full_commit_revision(revision: str, setting: str) -> str:
    """Require an immutable 40-character Hugging Face commit selector.

    ``snapshot_download`` accepts branches and tags, but a moving selector does
    not bind a future process restart to the bytes reviewed today. Optional
    model wrappers call this before resolving their local snapshot so an
    invalid selector degrades that capability instead of silently moving it.
    """
    if not _FULL_COMMIT_RE.fullmatch(revision):
        raise ValueError(
            f"{setting} must be a full 40-character Hugging Face commit")
    return revision


@dataclass(frozen=True)
class ModelSnapshot:
    """An immutable, locally cached Hugging Face model snapshot."""

    model_id: str
    revision: str
    snapshot_path: Path
    processor_config_fingerprint: str


def processor_config_fingerprint(snapshot_path: Path) -> str:
    """Hash model/processor configuration files in a cached snapshot.

    The snapshot path itself is supplied by ``huggingface_hub``. We inspect
    files beneath that supported path rather than reconstructing Hub cache
    internals. Weight files are excluded because the immutable Hub revision
    already binds them and hashing them would turn every server start into a
    multi-GB read.
    """
    root = Path(snapshot_path)
    files = sorted(
        (
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in _CONFIG_SUFFIXES
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise RuntimeError(
            f"cached snapshot {root} has no processor/configuration files")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def resolve_model_snapshot(
    model_id: str,
    revision: str | None = None,
    local_files_only: bool = True,
) -> ModelSnapshot:
    """Resolve a model through the supported Hugging Face cache APIs.

    ``local_files_only=True`` is the mandatory request/probe path and cannot
    perform a download. Ingest may pass ``False`` for its explicit one-time
    model preparation. ``scan_cache_dir`` supplies the full commit hash; no
    cache-directory naming convention is inferred here.
    """
    from huggingface_hub import scan_cache_dir, snapshot_download

    requested_revision = revision or "main"
    resolved = snapshot_download(
        repo_id=model_id,
        revision=requested_revision,
        local_files_only=local_files_only,
    )
    if not isinstance(resolved, str):
        raise RuntimeError(f"unexpected snapshot result for {model_id}")
    snapshot_path = Path(resolved).resolve()
    cache_info = scan_cache_dir()
    matching = [
        cached_revision
        for repo in cache_info.repos
        if repo.repo_type == "model" and repo.repo_id == model_id
        for cached_revision in repo.revisions
        if cached_revision.snapshot_path.resolve() == snapshot_path
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"could not map cached snapshot for {model_id} to one full commit")
    commit = matching[0].commit_hash
    if not _FULL_COMMIT_RE.fullmatch(commit):
        raise RuntimeError(
            f"cached revision for {model_id} is not a full 40-character commit")
    return ModelSnapshot(
        model_id=model_id,
        revision=commit,
        snapshot_path=snapshot_path,
        processor_config_fingerprint=processor_config_fingerprint(snapshot_path),
    )


def ordered_ids_sha256(ids: np.ndarray | list[int]) -> str:
    """Hash ordered IDs in one portable representation."""
    values = np.asarray(ids, dtype="<i8")
    if values.ndim != 1:
        raise ValueError("ordered IDs must be one-dimensional")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


# -- encoders -----------------------------------------------------------------

class QwenEncoder:
    """Qwen3-VL-Embedding through the official sentence-transformers path."""

    def __init__(self, snapshot: ModelSnapshot):
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading %s@%s from %s (in-process, sentence-transformers)",
            snapshot.model_id,
            snapshot.revision,
            snapshot.snapshot_path,
        )
        self.model_id = snapshot.model_id
        self.revision = snapshot.revision
        self.model = SentenceTransformer(
            str(snapshot.snapshot_path),
            local_files_only=True,
        )
        get_dimension = getattr(
            self.model, "get_sentence_embedding_dimension", None)
        self.dimension = get_dimension() if callable(get_dimension) else None
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

def _read_manifest(emb_dir: Path) -> tuple[dict | None, str | None]:
    """Read a manifest and distinguish malformed JSON from a non-object.

    ``json.loads`` accepts arrays, strings, numbers, booleans and null. None of
    those provide the manifest mapping contract, and allowing them through to
    ``dict.get`` turned a corrupt artifact into an unhandled request error.
    """
    try:
        value = json.loads((emb_dir / MANIFEST_NAME).read_text())
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable: {exc}"
    if not isinstance(value, dict):
        return None, (
            "must be a JSON object "
            f"(found {type(value).__name__})")
    return value, None


def read_manifest(emb_dir: Path) -> Optional[dict]:
    """Return a manifest mapping, or ``None`` for any invalid representation."""
    manifest, _ = _read_manifest(emb_dir)
    return manifest


def write_manifest(emb_dir: Path, **fields) -> None:
    """Written last, atomically: its presence with status=complete is the
    commit marker for the whole index build."""
    tmp = emb_dir / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(fields, indent=1))
    os.replace(tmp, emb_dir / MANIFEST_NAME)


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """np.save via a temp file + rename so an interrupted build can never leave
    a half-written array that later loads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


_INDEX_FILES = (
    "sample_ids.npy",
    "image_embeddings.npy",
    "caption_ids.npy",
    "caption_embeddings.npy",
)
_ARTIFACT_FILES = (MANIFEST_NAME, *_INDEX_FILES)


@dataclass(frozen=True)
class ValidatedProviderArtifact:
    """One immutable, validated provider generation.

    The generation covers the manifest and all four arrays. Runtime acquisition
    rechecks this cheap stat identity after loading the arrays; validation's
    expensive ID hashing and processor/config hashing only repeat when one of
    those files changes.
    """

    provider: str
    model_id: str
    emb_dir: Path
    manifest: dict
    snapshot: ModelSnapshot
    generation: str
    stamp: tuple


_validation_lock = threading.Lock()
_validated_artifacts: dict[tuple[str, str, str, str | None],
                           ValidatedProviderArtifact] = {}


def _artifact_stamp(emb_dir: Path) -> tuple:
    """Cheap identity used to invalidate cached artifact validation."""
    entries = []
    for name in _ARTIFACT_FILES:
        try:
            stat = (emb_dir / name).stat()
            entries.append(
                (name, stat.st_size, stat.st_mtime_ns, stat.st_ino))
        except OSError:
            entries.append((name, None, None, None))
    return (str(emb_dir.resolve()), *entries)


def _index_files_present(emb_dir: Path, *, include_captions: bool = False) -> bool:
    names = _INDEX_FILES if include_captions else _INDEX_FILES[:2]
    return all((emb_dir / name).exists() for name in names)


def manifest_problem(emb_dir: Path, model_id: str) -> Optional[str]:
    """Validate the Qwen provider's complete artifact contract."""
    return provider_manifest_problem(
        emb_dir,
        provider="qwen3_vl",
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
    )


def _array_contract_problem(emb_dir: Path, manifest: dict) -> str | None:
    """Validate all four arrays against a committed manifest."""
    if not _index_files_present(emb_dir, include_captions=True):
        missing = [name for name in _INDEX_FILES if not (emb_dir / name).exists()]
        return f"index arrays missing {', '.join(missing)}"
    try:
        sample_ids = np.load(
            emb_dir / "sample_ids.npy", mmap_mode="r", allow_pickle=False)
        image_embeddings = np.load(
            emb_dir / "image_embeddings.npy", mmap_mode="r", allow_pickle=False)
        caption_ids = np.load(
            emb_dir / "caption_ids.npy", mmap_mode="r", allow_pickle=False)
        caption_embeddings = np.load(
            emb_dir / "caption_embeddings.npy", mmap_mode="r", allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        return f"index array unreadable: {exc}"

    if sample_ids.ndim != 1 or caption_ids.ndim != 1:
        return "sample and caption ID arrays must be one-dimensional"
    if image_embeddings.ndim != 2 or caption_embeddings.ndim != 2:
        return "image and caption embedding arrays must be two-dimensional"
    if sample_ids.dtype.kind not in "iu" or caption_ids.dtype.kind not in "iu":
        return "sample and caption ID arrays must contain integers"
    if image_embeddings.dtype != np.float32 or caption_embeddings.dtype != np.float32:
        return "embedding dtype must be float32"
    if manifest.get("dtype") != "float32":
        return f"manifest dtype is {manifest.get('dtype')!r}, expected 'float32'"

    corpus_count = manifest.get("corpus_count")
    caption_count = manifest.get("caption_count")
    if corpus_count != len(sample_ids) or corpus_count != len(image_embeddings):
        return (
            "corpus count mismatch: manifest "
            f"{corpus_count}, IDs {len(sample_ids)}, vectors {len(image_embeddings)}")
    if caption_count != len(caption_ids) or caption_count != len(caption_embeddings):
        return (
            "caption count mismatch: manifest "
            f"{caption_count}, IDs {len(caption_ids)}, "
            f"vectors {len(caption_embeddings)}")

    sample_hash = ordered_ids_sha256(sample_ids)
    if manifest.get("sample_ids_sha256") != sample_hash:
        return "sample ID hash does not match the committed manifest"
    caption_hash = ordered_ids_sha256(caption_ids)
    if manifest.get("caption_ids_sha256") != caption_hash:
        return "caption ID hash does not match the committed manifest"

    dim = manifest.get("dim")
    if (
        not isinstance(dim, int)
        or image_embeddings.shape[1] != dim
        or caption_embeddings.shape[1] != dim
    ):
        return (
            "embedding dimension mismatch: manifest "
            f"{dim}, images {image_embeddings.shape[1]}, "
            f"captions {caption_embeddings.shape[1]}")
    return None


def _validate_provider_artifact_uncached(
    emb_dir: Path,
    *,
    provider: str,
    model_id: str,
    prompt_version: str | None = None,
) -> tuple[str | None, dict | None, ModelSnapshot | None]:
    rebuild_hint = (
        _SIGLIP_INGEST_HINT if provider == "siglip2" else _QWEN_INGEST_HINT)
    if not _index_files_present(emb_dir):
        return f"index not built — run `{rebuild_hint}`", None, None
    manifest, manifest_problem = _read_manifest(emb_dir)
    if manifest_problem == "missing":
        return (
            "index incomplete or legacy (no complete commit marker) — "
            f"rebuild with `{rebuild_hint}`"), None, None
    if manifest_problem is not None:
        return (
            f"manifest {manifest_problem} — rebuild with `{rebuild_hint}`"
        ), None, None
    assert manifest is not None
    if manifest.get("status") != "complete":
        return (
            "index incomplete (manifest is not a complete commit marker) — "
            f"rebuild with `{rebuild_hint}`"), None, None
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return (
            f"manifest schema is {manifest.get('schema_version')!r}, expected "
            f"{MANIFEST_SCHEMA_VERSION} — rebuild with `{rebuild_hint}`"
        ), None, None
    if manifest.get("provider") != provider:
        return (
            f"index manifest names provider {manifest.get('provider')!r} — "
            f"rebuild with `{rebuild_hint}`"), None, None
    if manifest.get("model_id") != model_id:
        return (
            f"index was built with {manifest.get('model_id')}; config wants "
            f"{model_id} — rebuild with `{rebuild_hint}`"), None, None
    if (
        prompt_version is not None
        and manifest.get("prompt_version") != prompt_version
    ):
        return (
            "index was encoded under a different prompt "
            f"({manifest.get('prompt_version')} vs {prompt_version}) — "
            f"rebuild with `{rebuild_hint}`"), None, None
    if manifest.get("normalized") is not True:
        return (
            "manifest does not bind L2-normalized vectors — "
            f"rebuild with `{rebuild_hint}`"), None, None
    if not isinstance(manifest.get("versions"), dict):
        return (
            "manifest has no library versions — "
            f"rebuild with `{rebuild_hint}`"), None, None

    revision = manifest.get("revision")
    if not isinstance(revision, str) or not _FULL_COMMIT_RE.fullmatch(revision):
        return (
            "manifest has no full cached Hugging Face commit revision — "
            f"rebuild with `{rebuild_hint}`"), None, None
    try:
        snapshot = resolve_model_snapshot(
            model_id, revision=revision, local_files_only=True)
    except Exception as exc:
        return (
            f"manifest-bound model snapshot is unavailable locally: {exc} — "
            f"rebuild with `{rebuild_hint}`"), None, None
    if snapshot.revision != revision:
        return (
            f"cached snapshot resolved to {snapshot.revision}, manifest binds "
            f"{revision} — rebuild with `{rebuild_hint}`"), None, None
    fingerprint = manifest.get("processor_config_fingerprint")
    if fingerprint != snapshot.processor_config_fingerprint:
        return (
            "processor/config fingerprint differs from the committed snapshot "
            f"({fingerprint} vs {snapshot.processor_config_fingerprint}) — "
            f"rebuild with `{rebuild_hint}`"), None, None

    arrays_problem = _array_contract_problem(emb_dir, manifest)
    if arrays_problem is not None:
        return (
            f"{arrays_problem} — rebuild with `{rebuild_hint}`"
        ), None, None
    return None, manifest, snapshot


def _validated_provider_artifact(
    emb_dir: Path,
    *,
    provider: str,
    model_id: str,
    prompt_version: str | None = None,
) -> tuple[ValidatedProviderArtifact | None, str | None]:
    key = (provider, str(emb_dir.resolve()), model_id, prompt_version)
    with _validation_lock:
        stamp = _artifact_stamp(emb_dir)
        cached = _validated_artifacts.get(key)
        if cached is not None and cached.stamp == stamp:
            return cached, None

        problem, manifest, snapshot = _validate_provider_artifact_uncached(
            emb_dir,
            provider=provider,
            model_id=model_id,
            prompt_version=prompt_version,
        )
        if problem is not None:
            _validated_artifacts.pop(key, None)
            return None, problem
        assert manifest is not None and snapshot is not None

        final_stamp = _artifact_stamp(emb_dir)
        if final_stamp != stamp:
            _validated_artifacts.pop(key, None)
            return None, (
                "provider artifact changed while it was being validated — "
                "retry after the index build completes")
        generation_payload = {
            "provider": provider,
            "model_id": model_id,
            "revision": snapshot.revision,
            "processor_config_fingerprint": (
                snapshot.processor_config_fingerprint),
            "stamp": final_stamp,
        }
        generation = hashlib.sha256(
            json.dumps(
                generation_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        artifact = ValidatedProviderArtifact(
            provider=provider,
            model_id=model_id,
            emb_dir=emb_dir,
            manifest=dict(manifest),
            snapshot=snapshot,
            generation=generation,
            stamp=final_stamp,
        )
        _validated_artifacts[key] = artifact
        return artifact, None


def provider_manifest_problem(
    emb_dir: Path,
    *,
    provider: str,
    model_id: str,
    prompt_version: str | None = None,
) -> str | None:
    """Why a provider index cannot be served, or ``None`` when coherent."""
    _, problem = _validated_provider_artifact(
        emb_dir,
        provider=provider,
        model_id=model_id,
        prompt_version=prompt_version,
    )
    return problem


def siglip_manifest_problem(emb_dir: Path, model_id: str) -> str | None:
    """Validate the SigLIP provider's complete artifact contract."""
    return provider_manifest_problem(
        emb_dir,
        provider="siglip2",
        model_id=model_id,
    )


def remove_manifest(emb_dir: Path) -> None:
    """Remove the commit marker before any array in an index is rewritten."""
    (emb_dir / MANIFEST_NAME).unlink(missing_ok=True)


def remove_all_provider_manifests() -> None:
    """Invalidate every provider before the corpus changes.

    Provider builds are corpus-wide. Adding one database row makes every
    previous provider manifest stale, including providers that are not being
    rebuilt in the current command.
    """
    for provider in ("siglip2", "qwen3_vl"):
        remove_manifest(config.emb_dir_for(provider))
    with _validation_lock:
        _validated_artifacts.clear()
    invalidate_providers()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except Exception:
        return None


def _sim_floor_p10(
    embeddings: np.ndarray,
    chunk: int = 1024,
) -> float | None:
    """Measure the self-excluded nearest-neighbour tenth percentile."""
    if len(embeddings) < 2:
        return None
    top1 = np.empty(len(embeddings), dtype=np.float32)
    for start in range(0, len(embeddings), chunk):
        block = embeddings[start:start + chunk] @ embeddings.T
        for block_index in range(block.shape[0]):
            block[block_index, start + block_index] = -np.inf
        top1[start:start + block.shape[0]] = block.max(axis=1)
    return round(float(np.percentile(top1, 10)), 4)


def finalize_provider_manifest(
    conn: sqlite3.Connection,
    snapshot: ModelSnapshot,
    provider: str,
    emb_dir: Path | None = None,
    prompt_version: str | None = None,
    image_encode_seconds: float | None = None,
    caption_encode_seconds: float | None = None,
) -> dict:
    """Validate completed arrays and atomically commit their manifest.

    The database order is checked against both ID arrays before the marker is
    written. A failed or interrupted build therefore remains unservable rather
    than blessing a partially updated mix of image and caption vectors.
    """
    target = emb_dir or config.emb_dir_for(provider)
    expected_model_id = provider_model_id(provider)
    if snapshot.model_id != expected_model_id:
        raise RuntimeError(
            f"cannot finalize {provider}: snapshot is {snapshot.model_id}, "
            f"expected {expected_model_id}")
    if not _FULL_COMMIT_RE.fullmatch(snapshot.revision):
        raise RuntimeError(
            f"cannot finalize {provider}: snapshot revision is not a full commit")
    sample_ids = np.asarray(
        [row["id"] for row in conn.execute("SELECT id FROM samples ORDER BY id")],
        dtype=np.int64,
    )
    caption_ids = np.asarray(
        [row["id"] for row in conn.execute("SELECT id FROM captions ORDER BY id")],
        dtype=np.int64,
    )
    try:
        stored_sample_ids = np.load(
            target / "sample_ids.npy", mmap_mode="r", allow_pickle=False)
        image_embeddings = np.load(
            target / "image_embeddings.npy", mmap_mode="r", allow_pickle=False)
        stored_caption_ids = np.load(
            target / "caption_ids.npy", mmap_mode="r", allow_pickle=False)
        caption_embeddings = np.load(
            target / "caption_embeddings.npy", mmap_mode="r", allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot finalize incomplete {provider} arrays: {exc}") from exc
    if not np.array_equal(stored_sample_ids, sample_ids):
        raise RuntimeError(
            "cannot finalize: sample ID array differs from database order")
    if not np.array_equal(stored_caption_ids, caption_ids):
        raise RuntimeError(
            "cannot finalize: caption ID array differs from database order")
    if image_embeddings.ndim != 2 or caption_embeddings.ndim != 2:
        raise RuntimeError("cannot finalize: embedding arrays must be two-dimensional")
    if image_embeddings.shape[1] != caption_embeddings.shape[1]:
        raise RuntimeError(
            "cannot finalize: image/caption embedding dimensions differ")

    current_fingerprint = processor_config_fingerprint(snapshot.snapshot_path)
    if current_fingerprint != snapshot.processor_config_fingerprint:
        raise RuntimeError(
            "cannot finalize: cached processor/config changed during the build")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "provider": provider,
        "model_id": snapshot.model_id,
        "revision": snapshot.revision,
        "processor_config_fingerprint": current_fingerprint,
        "dim": int(image_embeddings.shape[1]),
        "dtype": "float32",
        "normalized": True,
        "corpus_count": len(sample_ids),
        "caption_count": len(caption_ids),
        "sample_ids_sha256": ordered_ids_sha256(sample_ids),
        "caption_ids_sha256": ordered_ids_sha256(caption_ids),
        "sim_floor_p10": _sim_floor_p10(image_embeddings),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            package: _package_version(package)
            for package in (
                "huggingface-hub",
                "numpy",
                "sentence-transformers",
                "torch",
                "transformers",
            )
        },
        "status": "complete",
    }
    if prompt_version is not None:
        manifest["prompt_version"] = prompt_version
    diagnostics = {
        "image_encode_seconds": image_encode_seconds,
        "caption_encode_seconds": caption_encode_seconds,
    }
    for name, value in diagnostics.items():
        if value is None:
            continue
        measured = float(value)
        if not np.isfinite(measured) or measured < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
        manifest[name] = round(measured, 3)
    arrays_problem = _array_contract_problem(target, manifest)
    if arrays_problem is not None:
        raise RuntimeError(
            f"cannot finalize {provider} manifest: {arrays_problem}")
    write_manifest(target, **manifest)
    return manifest


def finalize_siglip_manifest(
    conn: sqlite3.Connection,
    snapshot: ModelSnapshot,
    emb_dir: Path | None = None,
) -> dict:
    """Commit a complete SigLIP manifest."""
    return finalize_provider_manifest(
        conn,
        snapshot,
        provider="siglip2",
        emb_dir=emb_dir,
    )


def validated_provider_snapshot(provider: str) -> ModelSnapshot:
    """Return the exact local snapshot bound by a valid provider manifest."""
    model_id = provider_model_id(provider)
    emb_dir = config.emb_dir_for(provider)
    artifact, problem = _validated_provider_artifact(
        emb_dir,
        provider=provider,
        model_id=model_id,
        prompt_version=(PROMPT_VERSION if provider == "qwen3_vl" else None),
    )
    if problem is not None:
        raise RuntimeError(problem)
    assert artifact is not None
    return artifact.snapshot


def validated_siglip_snapshot() -> ModelSnapshot:
    """Return the exact local snapshot bound by a valid SigLIP manifest."""
    return validated_provider_snapshot("siglip2")


def validated_provider_artifact(provider: str) -> ValidatedProviderArtifact:
    """Return a provider's cached, generation-bound artifact contract."""
    if provider not in {"siglip2", "qwen3_vl"}:
        raise ValueError(f"unknown provider {provider!r}")
    artifact, problem = _validated_provider_artifact(
        config.emb_dir_for(provider),
        provider=provider,
        model_id=provider_model_id(provider),
        prompt_version=(PROMPT_VERSION if provider == "qwen3_vl" else None),
    )
    if problem is not None:
        raise RuntimeError(problem)
    assert artifact is not None
    return artifact


# -- per-provider descriptors -------------------------------------------------

def provider_model_id(name: str) -> str:
    return {"qwen3_vl": config.QWEN_EMBED_MODEL,
            "siglip2": config.EMBED_MODEL}[name]


def _weights_cached(model_id: str) -> bool:
    """Whether any valid revision of ``model_id`` is in the official HF cache."""
    try:
        from huggingface_hub import scan_cache_dir

        return any(
            repo.repo_type == "model"
            and repo.repo_id == model_id
            and bool(repo.revisions)
            for repo in scan_cache_dir().repos
        )
    except Exception:
        return False


def _qwen_stack_problem() -> str | None:
    """Report missing optional Qwen runtime packages without loading a model."""
    try:
        import sentence_transformers  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        return f"provider stack not installed — `{_INSTALL_HINT}`"
    return None


def _probe(name: str) -> Optional[str]:
    """Cheap availability check — no model load, safe on any request path."""
    emb_dir = config.emb_dir_for(name)
    if name == "siglip2":
        return siglip_manifest_problem(emb_dir, config.EMBED_MODEL)
    if name == "qwen3_vl":
        stack_problem = _qwen_stack_problem()
        if stack_problem is not None:
            return stack_problem
        if not _weights_cached(config.QWEN_EMBED_MODEL):
            return (f"model weights not downloaded — run `{_QWEN_INGEST_HINT}` "
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
    except (EOFError, OSError, ValueError):
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


@dataclass(frozen=True)
class RetrievalIndexBundle:
    """Both indexes from one validated provider/model generation."""

    provider: str
    model_id: str
    emb_dir: Path
    generation: str
    image_index: Any
    caption_index: Any
    manifest: dict


@dataclass(frozen=True)
class RetrievalBundle(RetrievalIndexBundle):
    """A validated index bundle extended with its matching encoder."""

    encoder: Any


_lock = threading.RLock()
_state: Optional[ProviderState] = None
_encoders: dict[str, object] = {}
_hard_failed: dict[str, tuple[float, str]] = {}   # name -> (monotonic, reason)
_index_bundle: RetrievalIndexBundle | None = None
_runtime_bundle: RetrievalBundle | None = None


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


def load_encoder_for(
    name: str,
    *,
    snapshot: ModelSnapshot | None = None,
):
    """Force-load a specific provider's encoder (ingest / benchmarks).
    Raises on failure — callers there want the real error, not a fallback."""
    if name not in {"siglip2", "qwen3_vl"}:
        raise ValueError(f"unknown provider {name!r}")
    if snapshot is not None and snapshot.model_id != provider_model_id(name):
        raise ValueError(
            f"{name} expects {provider_model_id(name)}, "
            f"got snapshot for {snapshot.model_id}")
    bound_snapshot = snapshot or validated_provider_snapshot(name)
    cache_key = (
        f"{name}:{bound_snapshot.revision}:"
        f"{bound_snapshot.processor_config_fingerprint}")
    # Construction happens under the same lock as publication. The second
    # lookup is what makes simultaneous first requests construct one model,
    # not one model per request.
    with _lock:
        if cache_key in _encoders:
            return _encoders[cache_key]
        if name == "qwen3_vl":
            enc = QwenEncoder(bound_snapshot)
        else:
            from .embedder import Embedder

            enc = Embedder(
                bound_snapshot.snapshot_path,
                model_id=bound_snapshot.model_id,
                revision=bound_snapshot.revision,
                local_files_only=True,
            )
        _encoders[cache_key] = enc
        return enc


def _encoder_dimension(encoder: Any) -> int | None:
    for name in ("dimension", "embedding_dimension", "dim"):
        value = getattr(encoder, name, None)
        if isinstance(value, (int, np.integer)):
            return int(value)
    model = getattr(encoder, "model", None)
    getter = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(getter):
        value = getter()
        if isinstance(value, (int, np.integer)):
            return int(value)
    model_config = getattr(model, "config", None)
    value = getattr(model_config, "projection_dim", None)
    return int(value) if isinstance(value, (int, np.integer)) else None


def _mark_runtime_failure(provider: str, exc: Exception) -> None:
    global _index_bundle, _runtime_bundle, _state
    reason = f"{provider} failed to load a bound retrieval runtime: {exc}"
    logger.warning("%s — falling back", reason)
    with _lock:
        _hard_failed[provider] = (time.monotonic(), reason)
        _state = None
        _index_bundle = None
        _runtime_bundle = None
    from . import hubness

    hubness.invalidate()


def get_retrieval_index_bundle() -> RetrievalIndexBundle | None:
    """Acquire both indexes from one validated provider generation.

    This path intentionally does not construct the model. It lets diagnostics
    use stored caption vectors when the exact generation's encoder is
    unavailable, without weakening search's encoder/index pairing.
    """
    global _index_bundle
    attempted: set[str] = set()
    while True:
        state = resolve()
        provider = state.active
        if provider is None or provider in attempted:
            return None
        attempted.add(provider)
        try:
            artifact = validated_provider_artifact(provider)
            with _lock:
                if (
                    _index_bundle is not None
                    and _index_bundle.generation == artifact.generation
                ):
                    return _index_bundle
                expected_dim = artifact.manifest["dim"]

                # Local import keeps index -> providers one-way for the
                # lightweight status/index-only path.
                from .index import EmbeddingIndex

                image_index = EmbeddingIndex.load(
                    "image", emb_dir=artifact.emb_dir)
                caption_index = EmbeddingIndex.load(
                    "caption", emb_dir=artifact.emb_dir)
                if image_index is None or caption_index is None:
                    raise RuntimeError(
                        "validated image/caption index could not be loaded")
                if (
                    image_index.embeddings.shape[1] != expected_dim
                    or caption_index.embeddings.shape[1] != expected_dim
                ):
                    raise RuntimeError(
                        "loaded index dimensions differ from the manifest")
                if _artifact_stamp(artifact.emb_dir) != artifact.stamp:
                    raise RuntimeError(
                        "provider artifact changed during index acquisition")

                _index_bundle = RetrievalIndexBundle(
                    provider=provider,
                    model_id=artifact.model_id,
                    emb_dir=artifact.emb_dir,
                    generation=artifact.generation,
                    image_index=image_index,
                    caption_index=caption_index,
                    manifest=dict(artifact.manifest),
                )
                return _index_bundle
        except Exception as exc:
            _mark_runtime_failure(provider, exc)


def bind_retrieval_bundle(
    indexes: RetrievalIndexBundle,
) -> RetrievalBundle:
    """Extend an index bundle with the encoder for that exact generation.

    Raises instead of silently selecting another provider. Search catches this
    and retries its whole acquisition chain; eval can retain ``indexes`` and
    report its documented stored-caption fallback.
    """
    global _runtime_bundle
    artifact = validated_provider_artifact(indexes.provider)
    if artifact.generation != indexes.generation:
        raise RuntimeError(
            "provider generation changed before encoder acquisition")
    with _lock:
        if (
            _runtime_bundle is not None
            and _runtime_bundle.generation == indexes.generation
        ):
            return _runtime_bundle
        encoder = load_encoder_for(
            indexes.provider, snapshot=artifact.snapshot)
        encoder_dim = _encoder_dimension(encoder)
        expected_dim = indexes.manifest["dim"]
        if encoder_dim is not None and encoder_dim != expected_dim:
            raise RuntimeError(
                f"encoder dimension {encoder_dim} does not match "
                f"manifest dimension {expected_dim}")
        if _artifact_stamp(indexes.emb_dir) != artifact.stamp:
            raise RuntimeError(
                "provider artifact changed during encoder acquisition")
        _runtime_bundle = RetrievalBundle(
            provider=indexes.provider,
            model_id=indexes.model_id,
            emb_dir=indexes.emb_dir,
            generation=indexes.generation,
            image_index=indexes.image_index,
            caption_index=indexes.caption_index,
            manifest=dict(indexes.manifest),
            encoder=encoder,
        )
        return _runtime_bundle


def get_retrieval_bundle() -> RetrievalBundle | None:
    """Acquire one provider-bound encoder + image index + caption index.

    A construction failure discards the entire acquisition and retries the
    fallback provider, so no request can retain one space's matrix and receive
    another space's encoder.
    """
    attempted: set[str] = set()
    while True:
        indexes = get_retrieval_index_bundle()
        if indexes is None or indexes.provider in attempted:
            return None
        attempted.add(indexes.provider)
        try:
            return bind_retrieval_bundle(indexes)
        except Exception as exc:
            _mark_runtime_failure(indexes.provider, exc)


def get_encoder():
    """The active bound provider encoder, or ``None`` on clean degradation."""
    bundle = get_retrieval_bundle()
    return bundle.encoder if bundle is not None else None


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
    global _index_bundle, _runtime_bundle, _state
    with _lock:
        _state = None
        _index_bundle = None
        _runtime_bundle = None
        _encoders.clear()
        _hard_failed.clear()
    with _validation_lock:
        _validated_artifacts.clear()
