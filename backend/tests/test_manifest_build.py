"""Commit-marker ordering and deterministic model configuration identity."""

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from app import analyze, config, db, ingest
from app.ml import providers


def _snapshot(path: Path) -> providers.ModelSnapshot:
    return providers.ModelSnapshot(
        model_id=config.EMBED_MODEL,
        revision="b" * 40,
        snapshot_path=path,
        processor_config_fingerprint="processor-fingerprint",
    )


def test_analysis_commits_manifest_after_every_stage(monkeypatch, tmp_path):
    """No caption rewrite or dependent analysis may run after the marker."""
    events: list[str] = []
    snapshot = _snapshot(tmp_path / "snapshot")

    monkeypatch.setattr(
        providers, "remove_manifest",
        lambda emb_dir: events.append("remove-marker"),
    )
    monkeypatch.setattr(
        analyze, "embed_captions",
        lambda conn, embedder=None: events.append("caption-arrays"),
    )
    monkeypatch.setattr(
        analyze, "compute_caption_scores",
        lambda conn: events.append("caption-scores"),
    )
    monkeypatch.setattr(
        analyze, "compute_attributes",
        lambda conn, embedder=None: events.append("attributes"),
    )
    monkeypatch.setattr(
        analyze, "compute_axes",
        lambda conn: events.append("axes"),
    )
    monkeypatch.setattr(
        providers, "finalize_siglip_manifest",
        lambda conn, bound: events.append("commit-marker"),
    )

    analyze.run_all(
        object(),
        embedder=object(),
        snapshot=snapshot,
    )

    assert events == [
        "remove-marker",
        "caption-arrays",
        "caption-scores",
        "attributes",
        "axes",
        "commit-marker",
    ]


def test_processor_config_fingerprint_is_deterministic_and_ignores_weights(
    tmp_path,
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type":"siglip"}')
    (snapshot / "preprocessor_config.json").write_text('{"size":256}')
    (snapshot / "model.safetensors").write_bytes(b"first weight payload")

    original = providers.processor_config_fingerprint(snapshot)
    assert providers.processor_config_fingerprint(snapshot) == original

    (snapshot / "model.safetensors").write_bytes(b"different weight payload")
    assert providers.processor_config_fingerprint(snapshot) == original

    (snapshot / "preprocessor_config.json").write_text('{"size":384}')
    assert providers.processor_config_fingerprint(snapshot) != original


def test_finalize_provider_manifest_binds_database_arrays_and_snapshot(tmp_path):
    snapshot_path = tmp_path / "snapshot"
    snapshot_path.mkdir()
    (snapshot_path / "config.json").write_text('{"model_type":"siglip"}')
    (snapshot_path / "preprocessor_config.json").write_text('{"size":256}')
    snapshot = providers.ModelSnapshot(
        model_id=config.EMBED_MODEL,
        revision="c" * 40,
        snapshot_path=snapshot_path,
        processor_config_fingerprint=providers.processor_config_fingerprint(
            snapshot_path),
    )
    emb_dir = tmp_path / "embeddings"
    sample_ids = np.array([1, 2], dtype=np.int64)
    caption_ids = np.array([10, 11, 12], dtype=np.int64)
    providers.atomic_save_npy(emb_dir / "sample_ids.npy", sample_ids)
    providers.atomic_save_npy(
        emb_dir / "image_embeddings.npy",
        np.eye(2, 4, dtype=np.float32),
    )
    providers.atomic_save_npy(emb_dir / "caption_ids.npy", caption_ids)
    providers.atomic_save_npy(
        emb_dir / "caption_embeddings.npy",
        np.eye(3, 4, dtype=np.float32),
    )

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE samples (id INTEGER PRIMARY KEY);"
        "CREATE TABLE captions (id INTEGER PRIMARY KEY);"
        "INSERT INTO samples VALUES (1), (2);"
        "INSERT INTO captions VALUES (10), (11), (12);"
    )
    manifest = providers.finalize_provider_manifest(
        conn,
        snapshot,
        provider="siglip2",
        emb_dir=emb_dir,
        image_encode_seconds=12.3456,
        caption_encode_seconds=4.5678,
    )

    assert providers.read_manifest(emb_dir) == manifest
    assert manifest["status"] == "complete"
    assert manifest["revision"] == snapshot.revision
    assert manifest["processor_config_fingerprint"] == (
        snapshot.processor_config_fingerprint)
    assert manifest["sample_ids_sha256"] == providers.ordered_ids_sha256(
        sample_ids)
    assert manifest["caption_ids_sha256"] == providers.ordered_ids_sha256(
        caption_ids)
    assert manifest["corpus_count"] == 2
    assert manifest["caption_count"] == 3
    assert manifest["dim"] == 4
    assert manifest["normalized"] is True
    assert manifest["image_encode_seconds"] == 12.346
    assert manifest["caption_encode_seconds"] == 4.568


def test_first_corpus_mutation_invalidates_every_provider_manifest(
    monkeypatch, tmp_path,
):
    data_dir = tmp_path / "data"
    images_dir = data_dir / "images"
    thumbs_dir = data_dir / "thumbs"
    embeddings_dir = data_dir / "embeddings"
    for directory in (images_dir, thumbs_dir, embeddings_dir):
        directory.mkdir(parents=True)
    monkeypatch.setattr(config, "IMAGES_DIR", images_dir)
    monkeypatch.setattr(config, "THUMBS_DIR", thumbs_dir)
    monkeypatch.setattr(config, "EMB_DIR", embeddings_dir)

    for provider in ("siglip2", "qwen3_vl"):
        target = config.emb_dir_for(provider)
        target.mkdir(parents=True, exist_ok=True)
        (target / providers.MANIFEST_NAME).write_text('{"status":"complete"}')

    siglip_marker = (
        config.emb_dir_for("siglip2") / providers.MANIFEST_NAME)
    qwen_marker = (
        config.emb_dir_for("qwen3_vl") / providers.MANIFEST_NAME)
    raw_image = Image.new("RGB", (16, 16), (30, 60, 90))

    class GuardedImage:
        width = raw_image.width
        height = raw_image.height

        def save(self, *args, **kwargs):
            assert not siglip_marker.exists()
            assert not qwen_marker.exists()
            return raw_image.save(*args, **kwargs)

        def copy(self):
            return raw_image.copy()

    adapter = SimpleNamespace(
        iter_samples=lambda limit=None: iter([
            SimpleNamespace(
                filename="new.jpg",
                image=GuardedImage(),
                captions=["a new sample"],
                split="train",
            )
        ])
    )
    monkeypatch.setattr(ingest, "get_adapter", lambda name: adapter)

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    assert ingest.ingest_samples(conn, "test", limit=1) == 1
    assert not siglip_marker.exists()
    assert not qwen_marker.exists()
