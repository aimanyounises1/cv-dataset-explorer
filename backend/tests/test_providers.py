"""Retrieval provider resolution: preferred-to-fallback chains with named
reasons, manifest integrity as the index commit marker, provider-scoped
fingerprints so vector spaces can never mix, the deterministic mock encoder's
contract, and the additive status fields the UI renders.

The rest of the suite runs pinned to siglip2 (conftest.py); these tests flip
providers deliberately and restore state after every test.

    cd backend && pytest tests/test_providers.py
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, db
from app.main import app
from app.ml import index as index_mod
from app.ml import providers


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Fresh EMB_DIR per test; provider + index caches reset on both sides."""
    monkeypatch.setattr(config, "EMB_DIR", tmp_path)
    # Resolution tests exercise manifest/weight branches without importing the
    # optional 4 GB Qwen runtime stack in the documented light CI install.
    monkeypatch.setattr(providers, "_qwen_stack_problem", lambda: None)
    providers.invalidate_providers()
    index_mod.invalidate_index()
    yield
    providers.invalidate_providers()
    index_mod.invalidate_index()


def _write_index(d, dim=4, n=3, ids=None, *, kind="image"):
    d.mkdir(parents=True, exist_ok=True)
    ids = np.array(ids if ids is not None else range(1, n + 1), dtype=np.int64)
    vecs = np.eye(len(ids), dim, dtype=np.float32)
    prefix = "sample" if kind == "image" else "caption"
    np.save(d / f"{prefix}_ids.npy", ids)
    np.save(d / f"{kind}_embeddings.npy", vecs)
    return ids, vecs


def _snapshot(
    path: Path,
    *,
    model_id=config.EMBED_MODEL,
    fingerprint="processor-fingerprint",
):
    return providers.ModelSnapshot(
        model_id=model_id,
        revision="a" * 40,
        snapshot_path=path,
        processor_config_fingerprint=fingerprint,
    )


def _siglip_manifest(d, snapshot, *, sample_ids=None, caption_ids=None, **over):
    sample_ids = np.asarray(
        sample_ids if sample_ids is not None else [1, 2, 3], dtype=np.int64)
    caption_ids = np.asarray(
        caption_ids if caption_ids is not None else [101, 102, 103], dtype=np.int64)
    fields = dict(
        schema_version=providers.MANIFEST_SCHEMA_VERSION,
        provider="siglip2",
        model_id=config.EMBED_MODEL,
        revision=snapshot.revision,
        processor_config_fingerprint=snapshot.processor_config_fingerprint,
        dim=4,
        dtype="float32",
        normalized=True,
        corpus_count=len(sample_ids),
        caption_count=len(caption_ids),
        sample_ids_sha256=providers.ordered_ids_sha256(sample_ids),
        caption_ids_sha256=providers.ordered_ids_sha256(caption_ids),
        versions={"transformers": "test"},
        status="complete",
        sim_floor_p10=0.31,
    )
    fields.update(over)
    (d / providers.MANIFEST_NAME).write_text(json.dumps(fields))


def _write_siglip_index(d, snapshot, *, dim=4, n=3, ids=None):
    sample_ids, vecs = _write_index(d, dim=dim, n=n, ids=ids)
    caption_ids = np.arange(101, 101 + n, dtype=np.int64)
    _write_index(d, dim=dim, n=n, ids=caption_ids, kind="caption")
    _siglip_manifest(
        d, snapshot, sample_ids=sample_ids, caption_ids=caption_ids, dim=dim)
    return sample_ids, vecs


def _qwen_manifest(
    d, snapshot, *, sample_ids=None, caption_ids=None, **over,
):
    sample_ids = np.asarray(
        sample_ids if sample_ids is not None else [1, 2, 3], dtype=np.int64)
    caption_ids = np.asarray(
        caption_ids if caption_ids is not None else [101, 102, 103], dtype=np.int64)
    fields = dict(
        schema_version=providers.MANIFEST_SCHEMA_VERSION,
        provider="qwen3_vl",
        model_id=config.QWEN_EMBED_MODEL,
        revision=snapshot.revision,
        processor_config_fingerprint=snapshot.processor_config_fingerprint,
        dim=4,
        dtype="float32",
        prompt_version=providers.PROMPT_VERSION,
        normalized=True,
        corpus_count=len(sample_ids),
        caption_count=len(caption_ids),
        sample_ids_sha256=providers.ordered_ids_sha256(sample_ids),
        caption_ids_sha256=providers.ordered_ids_sha256(caption_ids),
        versions={"sentence-transformers": "test"},
        status="complete",
        sim_floor_p10=0.31,
    )
    fields.update(over)
    (d / providers.MANIFEST_NAME).write_text(json.dumps(fields))


def _write_qwen_index(d, snapshot, *, dim=4, n=3, ids=None):
    sample_ids, vecs = _write_index(d, dim=dim, n=n, ids=ids)
    caption_ids = np.arange(101, 101 + n, dtype=np.int64)
    _write_index(d, dim=dim, n=n, ids=caption_ids, kind="caption")
    _qwen_manifest(
        d, snapshot, sample_ids=sample_ids, caption_ids=caption_ids, dim=dim)
    return sample_ids, vecs


def _patch_snapshots(monkeypatch, *snapshots):
    by_model = {snapshot.model_id: snapshot for snapshot in snapshots}

    def resolve(model_id, revision=None, local_files_only=True):
        snapshot = by_model[model_id]
        assert revision in (None, snapshot.revision)
        assert local_files_only is True
        return snapshot

    monkeypatch.setattr(providers, "resolve_model_snapshot", resolve)


def _set_provider(monkeypatch, name):
    monkeypatch.setattr(config, "EMBED_PROVIDER", name)
    providers.invalidate_providers()
    index_mod.invalidate_index()


# -- mock encoder contract ----------------------------------------------------

def test_mock_encoder_is_deterministic_normalized_float32():
    from tests.fake_provider import MockEncoder
    enc = MockEncoder()
    a = enc.encode_texts(["a dog", "a cat"])
    b = enc.encode_texts(["a dog", "a cat"])
    assert np.array_equal(a, b)
    assert a.dtype == np.float32 and a.shape == (2, MockEncoder.DIM)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    # different inputs, different vectors
    assert not np.allclose(a[0], a[1])
    img = Image.new("RGB", (8, 8), (200, 30, 40))
    iv1, iv2 = enc.encode_images([img]), enc.encode_images([img])
    assert np.array_equal(iv1, iv2)
    assert np.allclose(np.linalg.norm(iv1, axis=1), 1.0, atol=1e-5)


# -- resolution matrix --------------------------------------------------------

def test_legacy_siglip_flat_index_without_manifest_is_refused(monkeypatch):
    _write_index(config.EMB_DIR)
    _set_provider(monkeypatch, "siglip2")
    st = providers.resolve()
    assert st.active is None
    assert "incomplete" in st.fallback_reason
    assert "python3 -m app.ingest --provider siglip2" in st.fallback_reason


def test_complete_siglip_manifest_is_accepted(monkeypatch, tmp_path):
    snapshot = _snapshot(tmp_path / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    _patch_snapshots(monkeypatch, snapshot)
    _set_provider(monkeypatch, "siglip2")
    st = providers.resolve()
    assert st.active == "siglip2" and st.fallback_reason is None
    assert st.model_id == config.EMBED_MODEL
    assert st.dim == 4
    assert st.sim_floor == 0.31


def test_siglip_model_and_processor_mismatches_are_refused(monkeypatch, tmp_path):
    snapshot = _snapshot(tmp_path / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    _patch_snapshots(monkeypatch, snapshot)

    _siglip_manifest(
        config.EMB_DIR, snapshot, model_id="google/different-siglip")
    _set_provider(monkeypatch, "siglip2")
    assert "google/different-siglip" in providers.resolve().fallback_reason

    _siglip_manifest(
        config.EMB_DIR, snapshot,
        processor_config_fingerprint="different-processor")
    _set_provider(monkeypatch, "siglip2")
    assert "processor/config fingerprint" in providers.resolve().fallback_reason


@pytest.mark.parametrize("tamper", ["ids", "dim"])
def test_siglip_tampered_ids_or_dimension_are_refused(
    monkeypatch, tmp_path, tamper,
):
    snapshot = _snapshot(tmp_path / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    _patch_snapshots(monkeypatch, snapshot)
    if tamper == "ids":
        np.save(
            config.EMB_DIR / "sample_ids.npy",
            np.array([1, 9, 3], dtype=np.int64))
    else:
        np.save(
            config.EMB_DIR / "image_embeddings.npy",
            np.eye(3, 5, dtype=np.float32))

    _set_provider(monkeypatch, "siglip2")
    reason = providers.resolve().fallback_reason
    assert "sample ID hash" in reason if tamper == "ids" else "dimension" in reason

def test_qwen_weights_missing_falls_back_with_named_reason(monkeypatch):
    snapshot = _snapshot(config.EMB_DIR / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    _patch_snapshots(monkeypatch, snapshot)
    monkeypatch.setattr(providers, "_weights_cached", lambda m: False)
    _set_provider(monkeypatch, "qwen3_vl")
    st = providers.resolve()
    assert st.active == "siglip2"
    assert "weights not downloaded" in st.fallback_reason
    assert "qwen3_vl" in st.fallback_reason

def test_qwen_index_not_built_falls_back(monkeypatch):
    snapshot = _snapshot(config.EMB_DIR / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    _patch_snapshots(monkeypatch, snapshot)
    monkeypatch.setattr(providers, "_weights_cached", lambda m: True)
    _set_provider(monkeypatch, "qwen3_vl")
    st = providers.resolve()
    assert st.active == "siglip2"
    assert "index not built" in st.fallback_reason
    assert "app.ingest --provider qwen3_vl" in st.fallback_reason

def test_interrupted_build_reads_as_incomplete(monkeypatch):
    snapshot = _snapshot(config.EMB_DIR / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    _patch_snapshots(monkeypatch, snapshot)
    _write_index(config.emb_dir_for("qwen3_vl"))  # arrays, but no manifest
    monkeypatch.setattr(providers, "_weights_cached", lambda m: True)
    _set_provider(monkeypatch, "qwen3_vl")
    st = providers.resolve()
    assert st.active == "siglip2"
    assert "incomplete" in st.fallback_reason

def test_model_mismatch_names_both_models(monkeypatch):
    snapshot = _snapshot(config.EMB_DIR / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    _patch_snapshots(monkeypatch, snapshot)
    qdir = config.emb_dir_for("qwen3_vl")
    _write_index(qdir)
    qsnapshot = _snapshot(
        qdir / "snapshot", model_id=config.QWEN_EMBED_MODEL,
        fingerprint="qwen-processor")
    _qwen_manifest(qdir, qsnapshot, model_id="Qwen/Some-Other-Model")
    monkeypatch.setattr(providers, "_weights_cached", lambda m: True)
    _set_provider(monkeypatch, "qwen3_vl")
    st = providers.resolve()
    assert st.active == "siglip2"
    assert "was built with Qwen/Some-Other-Model" in st.fallback_reason

def test_complete_qwen_index_activates_with_manifest_dim_and_floor(monkeypatch):
    snapshot = _snapshot(config.EMB_DIR / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    qdir = config.emb_dir_for("qwen3_vl")
    qsnapshot = _snapshot(
        qdir / "snapshot", model_id=config.QWEN_EMBED_MODEL,
        fingerprint="qwen-processor")
    _write_qwen_index(qdir, qsnapshot)
    _patch_snapshots(monkeypatch, snapshot, qsnapshot)
    monkeypatch.setattr(providers, "_weights_cached", lambda m: True)
    _set_provider(monkeypatch, "qwen3_vl")
    st = providers.resolve()
    assert st.active == "qwen3_vl" and st.fallback_reason is None
    assert st.model_id == config.QWEN_EMBED_MODEL
    assert st.dim == 4 and st.sim_floor == 0.31


def test_legacy_qwen_manifest_is_refused(monkeypatch):
    snapshot = _snapshot(config.EMB_DIR / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    _patch_snapshots(monkeypatch, snapshot)
    qdir = config.emb_dir_for("qwen3_vl")
    _write_index(qdir)
    (qdir / providers.MANIFEST_NAME).write_text(json.dumps({
        "provider": "qwen3_vl",
        "model_id": config.QWEN_EMBED_MODEL,
        "prompt_version": providers.PROMPT_VERSION,
        "status": "complete",
    }))
    monkeypatch.setattr(providers, "_weights_cached", lambda model_id: True)
    _set_provider(monkeypatch, "qwen3_vl")

    state = providers.resolve()
    assert state.active == "siglip2"
    assert "manifest schema" in state.fallback_reason
    assert "app.ingest --provider qwen3_vl" in state.fallback_reason


@pytest.mark.parametrize("tamper", ["caption_ids", "caption_dim"])
def test_qwen_tampered_caption_arrays_are_refused(
    monkeypatch, tmp_path, tamper,
):
    siglip_snapshot = _snapshot(tmp_path / "siglip")
    _write_siglip_index(config.EMB_DIR, siglip_snapshot)
    qdir = config.emb_dir_for("qwen3_vl")
    qwen_snapshot = _snapshot(
        tmp_path / "qwen",
        model_id=config.QWEN_EMBED_MODEL,
        fingerprint="qwen-processor",
    )
    _write_qwen_index(qdir, qwen_snapshot)
    _patch_snapshots(monkeypatch, siglip_snapshot, qwen_snapshot)
    monkeypatch.setattr(providers, "_weights_cached", lambda model_id: True)
    if tamper == "caption_ids":
        np.save(qdir / "caption_ids.npy", np.array([101, 999, 103]))
    else:
        np.save(qdir / "caption_embeddings.npy", np.eye(3, 6, dtype=np.float32))

    _set_provider(monkeypatch, "qwen3_vl")
    reason = providers.resolve().fallback_reason
    expected = "caption ID hash" if tamper == "caption_ids" else "dimension"
    assert expected in reason


def test_nothing_available_degrades_to_keyword_with_both_reasons(monkeypatch):
    monkeypatch.setattr(providers, "_weights_cached", lambda m: False)
    _set_provider(monkeypatch, "qwen3_vl")
    st = providers.resolve()
    assert st.active is None and st.index_ready is False
    assert "qwen3_vl" in st.fallback_reason and "siglip2" in st.fallback_reason
    assert providers.get_encoder() is None


# -- vector spaces never mix --------------------------------------------------

def test_index_loads_from_the_active_providers_dir(monkeypatch):
    snapshot = _snapshot(config.EMB_DIR / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot, n=3)
    qdir = config.emb_dir_for("qwen3_vl")
    qsnapshot = _snapshot(
        qdir / "snapshot", model_id=config.QWEN_EMBED_MODEL,
        fingerprint="qwen-processor")
    _write_qwen_index(qdir, qsnapshot, n=5, ids=[10, 11, 12, 13, 14])
    _patch_snapshots(monkeypatch, snapshot, qsnapshot)
    monkeypatch.setattr(providers, "_weights_cached", lambda m: True)

    _set_provider(monkeypatch, "siglip2")
    assert len(index_mod.get_index().ids) == 3
    _set_provider(monkeypatch, "qwen3_vl")
    assert list(index_mod.get_index().ids) == [10, 11, 12, 13, 14]

def test_fingerprint_is_provider_scoped_and_isolated(monkeypatch):
    from app.api.deps import embeddings_fingerprint

    snapshot = _snapshot(config.EMB_DIR / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    qdir = config.emb_dir_for("qwen3_vl")
    qsnapshot = _snapshot(
        qdir / "snapshot", model_id=config.QWEN_EMBED_MODEL,
        fingerprint="qwen-processor")
    _write_qwen_index(qdir, qsnapshot)
    _patch_snapshots(monkeypatch, snapshot, qsnapshot)
    monkeypatch.setattr(providers, "_weights_cached", lambda m: True)

    _set_provider(monkeypatch, "siglip2")
    fp_siglip = embeddings_fingerprint()
    _set_provider(monkeypatch, "qwen3_vl")
    fp_qwen = embeddings_fingerprint()
    assert fp_siglip != fp_qwen

    # Rebuilding the qwen index must not move the siglip fingerprint: the
    # fallback environment stays exactly as it was.
    _write_index(qdir, n=4, ids=[20, 21, 22, 23])
    _set_provider(monkeypatch, "siglip2")
    assert embeddings_fingerprint() == fp_siglip

def test_eval_cache_key_carries_the_provider(monkeypatch):
    from app.api.eval import _cache_path

    snapshot = _snapshot(config.EMB_DIR / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    _patch_snapshots(monkeypatch, snapshot)
    _set_provider(monkeypatch, "siglip2")
    assert "siglip2" in _cache_path(100).name


# -- status surface -----------------------------------------------------------

def test_stats_overview_reports_provider_truth(monkeypatch):
    snapshot = _snapshot(config.EMB_DIR / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    _patch_snapshots(monkeypatch, snapshot)
    monkeypatch.setattr(providers, "_weights_cached", lambda m: False)
    _set_provider(monkeypatch, "qwen3_vl")
    with TestClient(app) as client:
        body = client.get("/api/stats/overview").json()
    assert body["embed_preferred"] == "qwen3_vl"
    assert body["embed_provider"] == "siglip2"
    assert body["embed_model"] == config.EMBED_MODEL
    assert "weights not downloaded" in body["embed_fallback_reason"]
    assert body["vlm_model"] == config.VLM_MODEL
    assert body["chat_model"] == config.CHAT_MODEL


def test_siglip_query_encoder_loads_manifest_snapshot_offline(
    monkeypatch, tmp_path,
):
    snapshot = _snapshot(tmp_path / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    _patch_snapshots(monkeypatch, snapshot)
    loaded = {}

    class FakeEmbedder:
        def __init__(self, model_path, *, model_id, revision, local_files_only):
            loaded.update(
                path=Path(model_path),
                model_id=model_id,
                revision=revision,
                local_files_only=local_files_only,
            )

    from app.ml import embedder as embedder_mod
    monkeypatch.setattr(embedder_mod, "Embedder", FakeEmbedder)
    _set_provider(monkeypatch, "siglip2")

    assert isinstance(providers.get_encoder(), FakeEmbedder)
    assert loaded == {
        "path": snapshot.snapshot_path,
        "model_id": config.EMBED_MODEL,
        "revision": snapshot.revision,
        "local_files_only": True,
    }


def test_qwen_encoder_loads_manifest_snapshot_offline(monkeypatch, tmp_path):
    snapshot = _snapshot(
        tmp_path / "qwen-snapshot",
        model_id=config.QWEN_EMBED_MODEL,
        fingerprint="qwen-processor",
    )
    loaded = {}

    class FakeSentenceTransformer:
        def __init__(self, model_path, *, local_files_only):
            loaded["path"] = Path(model_path)
            loaded["local_files_only"] = local_files_only

    sentence_transformers = pytest.importorskip("sentence_transformers")
    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )
    encoder = providers.QwenEncoder(snapshot)

    assert encoder.revision == snapshot.revision
    assert loaded == {
        "path": snapshot.snapshot_path,
        "local_files_only": True,
    }


def test_validated_artifact_hot_path_does_not_rehash_arrays(
    monkeypatch, tmp_path,
):
    snapshot = _snapshot(tmp_path / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    _patch_snapshots(monkeypatch, snapshot)
    calls = 0
    original = providers._array_contract_problem

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(providers, "_array_contract_problem", counted)
    assert providers.siglip_manifest_problem(
        config.EMB_DIR, config.EMBED_MODEL) is None
    assert providers.siglip_manifest_problem(
        config.EMB_DIR, config.EMBED_MODEL) is None
    assert calls == 1


def test_encoder_constructor_is_singleton_under_concurrency(
    monkeypatch, tmp_path,
):
    snapshot = _snapshot(tmp_path / "snapshot")
    created = 0
    created_lock = threading.Lock()

    class FakeEmbedder:
        def __init__(self, *args, **kwargs):
            nonlocal created
            time.sleep(0.02)
            with created_lock:
                created += 1

    from app.ml import embedder as embedder_mod
    monkeypatch.setattr(embedder_mod, "Embedder", FakeEmbedder)

    with ThreadPoolExecutor(max_workers=8) as pool:
        encoders = list(pool.map(
            lambda _: providers.load_encoder_for(
                "siglip2", snapshot=snapshot),
            range(8),
        ))

    assert created == 1
    assert all(encoder is encoders[0] for encoder in encoders)


@pytest.mark.parametrize(
    ("artifact", "expected"),
    [
        ("manifest-list", "JSON object"),
        ("invalid-utf8-manifest", "manifest unreadable"),
        ("zero-byte-array", "array unreadable"),
        ("truncated-array", "array unreadable"),
    ],
)
def test_malformed_provider_artifacts_degrade_with_named_reason(
    monkeypatch, tmp_path, artifact, expected,
):
    snapshot = _snapshot(tmp_path / "snapshot")
    _write_siglip_index(config.EMB_DIR, snapshot)
    _patch_snapshots(monkeypatch, snapshot)
    if artifact == "manifest-list":
        (config.EMB_DIR / providers.MANIFEST_NAME).write_text("[]")
    elif artifact == "invalid-utf8-manifest":
        (config.EMB_DIR / providers.MANIFEST_NAME).write_bytes(b"\xff\xfe")
    elif artifact == "zero-byte-array":
        (config.EMB_DIR / "image_embeddings.npy").write_bytes(b"")
    else:
        (config.EMB_DIR / "caption_embeddings.npy").write_bytes(b"\x93NUMPY")

    problem = providers.siglip_manifest_problem(
        config.EMB_DIR, config.EMBED_MODEL)
    assert expected in problem


def test_qwen_dimension_failure_retries_whole_bundle_in_siglip_space(
    monkeypatch, tmp_path,
):
    siglip_snapshot = _snapshot(tmp_path / "siglip")
    _write_siglip_index(
        config.EMB_DIR, siglip_snapshot, ids=[1, 2, 3])
    qwen_dir = config.emb_dir_for("qwen3_vl")
    qwen_snapshot = _snapshot(
        tmp_path / "qwen",
        model_id=config.QWEN_EMBED_MODEL,
        fingerprint="qwen-processor",
    )
    _write_qwen_index(
        qwen_dir, qwen_snapshot, ids=[10, 11, 12])
    _patch_snapshots(monkeypatch, siglip_snapshot, qwen_snapshot)
    monkeypatch.setattr(providers, "_weights_cached", lambda model_id: True)

    class WrongDimQwen:
        dimension = 5

        def __init__(self, snapshot):
            pass

    class FakeSiglip:
        dimension = 4

        def __init__(self, *args, **kwargs):
            pass

        def encode_texts(self, texts):
            return np.tile(
                np.array([[1, 0, 0, 0]], dtype=np.float32),
                (len(texts), 1),
            )

    from app.ml import embedder as embedder_mod
    monkeypatch.setattr(providers, "QwenEncoder", WrongDimQwen)
    monkeypatch.setattr(embedder_mod, "Embedder", FakeSiglip)
    _set_provider(monkeypatch, "qwen3_vl")

    runtime = providers.get_retrieval_bundle()
    assert runtime.provider == "siglip2"
    assert list(runtime.image_index.ids) == [1, 2, 3]
    # This dot product is the request boundary: a retained Qwen matrix would
    # either expose IDs 10..12 or fail on the 4-D SigLIP query.
    assert runtime.image_index.search(
        runtime.encoder.encode_texts(["dog"])[0])
    assert "encoder dimension 5" in providers.resolve().fallback_reason


def test_eval_route_uses_bound_indexes_when_qwen_constructor_fails(
    monkeypatch, tmp_path,
):
    conn = db.connect()
    db.init_db(conn)
    sample_ids: list[int] = []
    caption_ids: list[int] = []
    for i in range(3):
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
            "VALUES ('eval-provider-flip', ?, 'test', 8, 8, 1)",
            (f"eval_flip_{i}.jpg",),
        )
        sample_ids.append(cur.lastrowid)
        caption = conn.execute(
            "INSERT INTO captions(sample_id, idx, text) VALUES (?, 0, ?)",
            (cur.lastrowid, f"provider flip caption {i}"),
        )
        caption_ids.append(caption.lastrowid)
        conn.execute(
            "INSERT INTO captions_fts(rowid, text) VALUES (?, ?)",
            (caption.lastrowid, f"provider flip caption {i}"),
        )
    conn.commit()

    siglip_snapshot = _snapshot(tmp_path / "siglip")
    siglip_ids = np.asarray(sample_ids, dtype=np.int64)
    siglip_caption_ids = np.asarray(caption_ids, dtype=np.int64)
    _write_index(
        config.EMB_DIR, dim=4, ids=siglip_ids, kind="image")
    _write_index(
        config.EMB_DIR, dim=4, ids=siglip_caption_ids, kind="caption")
    _siglip_manifest(
        config.EMB_DIR,
        siglip_snapshot,
        sample_ids=siglip_ids,
        caption_ids=siglip_caption_ids,
        dim=4,
    )

    qwen_dir = config.emb_dir_for("qwen3_vl")
    qwen_snapshot = _snapshot(
        tmp_path / "qwen",
        model_id=config.QWEN_EMBED_MODEL,
        fingerprint="qwen-processor",
    )
    _write_index(qwen_dir, dim=5, ids=siglip_ids, kind="image")
    _write_index(
        qwen_dir, dim=5, ids=siglip_caption_ids, kind="caption")
    _qwen_manifest(
        qwen_dir,
        qwen_snapshot,
        sample_ids=siglip_ids,
        caption_ids=siglip_caption_ids,
        dim=5,
    )
    _patch_snapshots(monkeypatch, siglip_snapshot, qwen_snapshot)
    monkeypatch.setattr(providers, "_weights_cached", lambda model_id: True)

    class FailingQwen:
        def __init__(self, snapshot):
            raise RuntimeError("synthetic Qwen constructor failure")

    monkeypatch.setattr(providers, "QwenEncoder", FailingQwen)
    _set_provider(monkeypatch, "qwen3_vl")

    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/eval/retrieval", params={"sample_size": 50})
        assert response.status_code == 200
        body = response.json()
        assert body["available"] is True
        assert body["pool_size"] == 3
        assert providers.resolve().active == "qwen3_vl"
        assert "stored caption vectors" in body["message"]
        assert "synthetic Qwen constructor failure" in body["message"]
    finally:
        marks = ",".join("?" * len(caption_ids))
        conn.execute(
            f"DELETE FROM captions_fts WHERE rowid IN ({marks})",
            caption_ids,
        )
        conn.execute(
            f"DELETE FROM captions WHERE id IN ({marks})",
            caption_ids,
        )
        marks = ",".join("?" * len(sample_ids))
        conn.execute(
            f"DELETE FROM samples WHERE id IN ({marks})",
            sample_ids,
        )
        conn.commit()
        conn.close()
