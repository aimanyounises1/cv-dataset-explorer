"""Retrieval provider resolution: preferred-to-fallback chains with named
reasons, manifest integrity as the index commit marker, provider-scoped
fingerprints so vector spaces can never mix, the deterministic mock encoder's
contract, and the additive status fields the UI renders.

The rest of the suite runs pinned to siglip2 (conftest.py); these tests flip
providers deliberately and restore state after every test.

    cd backend && pytest tests/test_providers.py
"""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app.main import app
from app.ml import index as index_mod
from app.ml import providers


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Fresh EMB_DIR per test; provider + index caches reset on both sides."""
    monkeypatch.setattr(config, "EMB_DIR", tmp_path)
    providers.invalidate_providers()
    index_mod.invalidate_index()
    yield
    providers.invalidate_providers()
    index_mod.invalidate_index()


def _write_index(d, dim=4, n=3, ids=None):
    d.mkdir(parents=True, exist_ok=True)
    ids = np.array(ids if ids is not None else range(1, n + 1), dtype=np.int64)
    vecs = np.eye(len(ids), dim, dtype=np.float32)
    np.save(d / "sample_ids.npy", ids)
    np.save(d / "image_embeddings.npy", vecs)
    return ids, vecs


def _qwen_manifest(d, **over):
    fields = dict(provider="qwen3_vl", model_id=config.QWEN_EMBED_MODEL,
                  dim=4, prompt_version=providers.PROMPT_VERSION,
                  normalized=True, corpus_count=3, status="complete",
                  sim_floor_p10=0.31)
    fields.update(over)
    (d / providers.MANIFEST_NAME).write_text(json.dumps(fields))


def _set_provider(monkeypatch, name):
    monkeypatch.setattr(config, "EMBED_PROVIDER", name)
    providers.invalidate_providers()
    index_mod.invalidate_index()


# -- mock encoder contract ----------------------------------------------------

def test_mock_encoder_is_deterministic_normalized_float32():
    enc = providers.MockEncoder()
    a = enc.encode_texts(["a dog", "a cat"])
    b = enc.encode_texts(["a dog", "a cat"])
    assert np.array_equal(a, b)
    assert a.dtype == np.float32 and a.shape == (2, providers.MockEncoder.DIM)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    # different inputs, different vectors
    assert not np.allclose(a[0], a[1])
    img = Image.new("RGB", (8, 8), (200, 30, 40))
    iv1, iv2 = enc.encode_images([img]), enc.encode_images([img])
    assert np.array_equal(iv1, iv2)
    assert np.allclose(np.linalg.norm(iv1, axis=1), 1.0, atol=1e-5)


# -- resolution matrix --------------------------------------------------------

def test_preferred_siglip2_with_flat_index_is_active_without_reasons(monkeypatch):
    _write_index(config.EMB_DIR)
    _set_provider(monkeypatch, "siglip2")
    st = providers.resolve()
    assert st.active == "siglip2" and st.fallback_reason is None
    assert st.model_id == config.EMBED_MODEL
    assert st.dim == 4  # measured from the array, not assumed

def test_qwen_weights_missing_falls_back_with_named_reason(monkeypatch):
    _write_index(config.EMB_DIR)
    monkeypatch.setattr(providers, "_weights_cached", lambda m: False)
    _set_provider(monkeypatch, "qwen3_vl")
    st = providers.resolve()
    assert st.active == "siglip2"
    assert "weights not downloaded" in st.fallback_reason
    assert "qwen3_vl" in st.fallback_reason

def test_qwen_index_not_built_falls_back(monkeypatch):
    _write_index(config.EMB_DIR)
    monkeypatch.setattr(providers, "_weights_cached", lambda m: True)
    _set_provider(monkeypatch, "qwen3_vl")
    st = providers.resolve()
    assert st.active == "siglip2"
    assert "index not built" in st.fallback_reason
    assert "app.ingest --provider qwen3_vl" in st.fallback_reason

def test_interrupted_build_reads_as_incomplete(monkeypatch):
    _write_index(config.EMB_DIR)
    _write_index(config.emb_dir_for("qwen3_vl"))  # arrays, but no manifest
    monkeypatch.setattr(providers, "_weights_cached", lambda m: True)
    _set_provider(monkeypatch, "qwen3_vl")
    st = providers.resolve()
    assert st.active == "siglip2"
    assert "incomplete" in st.fallback_reason

def test_model_mismatch_names_both_models(monkeypatch):
    _write_index(config.EMB_DIR)
    qdir = config.emb_dir_for("qwen3_vl")
    _write_index(qdir)
    _qwen_manifest(qdir, model_id="Qwen/Some-Other-Model")
    monkeypatch.setattr(providers, "_weights_cached", lambda m: True)
    _set_provider(monkeypatch, "qwen3_vl")
    st = providers.resolve()
    assert st.active == "siglip2"
    assert "was built with Qwen/Some-Other-Model" in st.fallback_reason

def test_complete_qwen_index_activates_with_manifest_dim_and_floor(monkeypatch):
    _write_index(config.EMB_DIR)
    qdir = config.emb_dir_for("qwen3_vl")
    _write_index(qdir)
    _qwen_manifest(qdir)
    monkeypatch.setattr(providers, "_weights_cached", lambda m: True)
    _set_provider(monkeypatch, "qwen3_vl")
    st = providers.resolve()
    assert st.active == "qwen3_vl" and st.fallback_reason is None
    assert st.model_id == config.QWEN_EMBED_MODEL
    assert st.dim == 4 and st.sim_floor == 0.31

def test_nothing_available_degrades_to_keyword_with_both_reasons(monkeypatch):
    monkeypatch.setattr(providers, "_weights_cached", lambda m: False)
    _set_provider(monkeypatch, "qwen3_vl")
    st = providers.resolve()
    assert st.active is None and st.index_ready is False
    assert "qwen3_vl" in st.fallback_reason and "siglip2" in st.fallback_reason
    assert providers.get_encoder() is None


# -- vector spaces never mix --------------------------------------------------

def test_index_loads_from_the_active_providers_dir(monkeypatch):
    _write_index(config.EMB_DIR, n=3)
    qdir = config.emb_dir_for("qwen3_vl")
    _write_index(qdir, n=5, ids=[10, 11, 12, 13, 14])
    _qwen_manifest(qdir, corpus_count=5)
    monkeypatch.setattr(providers, "_weights_cached", lambda m: True)

    _set_provider(monkeypatch, "siglip2")
    assert len(index_mod.get_index().ids) == 3
    _set_provider(monkeypatch, "qwen3_vl")
    assert list(index_mod.get_index().ids) == [10, 11, 12, 13, 14]

def test_fingerprint_is_provider_scoped_and_isolated(monkeypatch):
    from app.api.deps import embeddings_fingerprint

    _write_index(config.EMB_DIR)
    qdir = config.emb_dir_for("qwen3_vl")
    _write_index(qdir)
    _qwen_manifest(qdir)
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

    _write_index(config.EMB_DIR)
    _set_provider(monkeypatch, "siglip2")
    assert "siglip2" in _cache_path(100).name


# -- status surface -----------------------------------------------------------

def test_stats_overview_reports_provider_truth(monkeypatch):
    _write_index(config.EMB_DIR)
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
