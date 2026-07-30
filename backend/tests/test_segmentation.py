"""SAM2 preview, accepted mask persistence, hierarchy and segment retrieval."""
import base64
import hashlib
import io
import json
import sqlite3
import sys
import threading
import time
import zipfile
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, db
from app.api import annotations as annotations_api
from app.main import app
from app.ml import segment as segment_ml
from app.ml.index import EmbeddingIndex
from app.proposal_tokens import (
    TOKEN_TTL_SECONDS,
    ProposalTokenError,
    issue_detection_proposal,
    resolve_detection_proposal,
)
from app.schemas import DetectionProposalSource, SegmentBox
from tests.fake_provider import MockEncoder


class FakeSegmenter:
    model_id = config.SEGMENT_MODEL
    revision = config.SEGMENT_REVISION

    def segment(self, image, *, points=None, labels=None, box=None):
        width, height = image.size
        mask = np.zeros((height, width), dtype=bool)
        mask[height // 4: 3 * height // 4, width // 4: 3 * width // 4] = True
        return mask, 0.93


@pytest.fixture(scope="module")
def ctx():
    conn = db.connect()
    db.init_db(conn)
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sample_ids = []
    colors = [(210, 30, 30), (30, 210, 30), (30, 30, 210), (180, 180, 30)]
    for i, color in enumerate(colors):
        filename = f"segment_{i}.jpg"
        Image.new("RGB", (80, 60), color).save(config.IMAGES_DIR / filename, "JPEG")
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
            "VALUES ('flickr8k', ?, 'train', 80, 60, 1)", (filename,))
        sample_ids.append(cur.lastrowid)
        conn.execute(
            "INSERT INTO captions(sample_id, idx, text) VALUES (?, 0, ?)",
            (cur.lastrowid, f"segment sample {i}"))
    conn.commit()
    original = segment_ml.get_segmenter
    segment_ml.get_segmenter = lambda: FakeSegmenter()
    with TestClient(app) as client:
        yield client, conn, sample_ids
    segment_ml.get_segmenter = original
    conn.execute("DELETE FROM annotations")
    for sample_id in sample_ids:
        conn.execute("DELETE FROM captions WHERE sample_id = ?", (sample_id,))
        conn.execute("DELETE FROM samples WHERE id = ?", (sample_id,))
    conn.commit()
    conn.close()
    for i in range(len(colors)):
        (config.IMAGES_DIR / f"segment_{i}.jpg").unlink(missing_ok=True)


def _box_body(**over):
    body = {
        "box": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8},
        "label_name": "dog",
        "parent_name": "animal",
    }
    body.update(over)
    return body


def _reviewed_body(client, sample_id, **over):
    """Return acceptance JSON bound to the exact preview bytes from the API."""
    body = _box_body(**over)
    preview_request = {
        "sample_id": sample_id,
        "points": body.get("points", []),
        "box": body.get("box"),
    }
    preview = client.post("/api/segment", json=preview_request)
    assert preview.status_code == 200, preview.text
    evidence = preview.json()
    return {
        **body,
        "preview_token": evidence["preview_token"],
        "mask_data_url": evidence["mask_data_url"],
    }


def _proposal_source(**over):
    source = {
        "kind": "detector",
        "model_id": config.DETECT_MODEL,
        "model_revision": config.DETECT_REVISION,
        "queries": "a dog. a person.",
        "original_label": "a dog",
        "proposed_label": "dog",
        "score": 0.87,
        "box": _box_body()["box"],
    }
    source.update(over)
    return source


def _proposal_token(sample_id, **over):
    source = DetectionProposalSource.model_validate(_proposal_source(**over))
    return issue_detection_proposal(sample_id, source)


def test_preview_is_browser_ready_and_resolves_detector_article(ctx):
    client, _, sample_ids = ctx
    response = client.post("/api/segment", json={
        "sample_id": sample_ids[0],
        "points": [{"x": 0.5, "y": 0.5, "label": 1}],
        "label_name": "a dog",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["bbox"] == {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}
    assert body["label_name"] == "dog"
    assert body["parent_name"] == "animal"
    assert body["label_path"] == ["animal", "dog"]
    assert body["model"] == config.SEGMENT_MODEL
    assert body["model_revision"] == config.SEGMENT_REVISION
    assert body["mask_width"] == 80 and body["mask_height"] == 60
    prefix, encoded = body["mask_data_url"].split(",", 1)
    assert prefix == "data:image/png;base64"
    assert Image.open(io.BytesIO(base64.b64decode(encoded))).size == (80, 60)


def test_accept_persists_the_exact_reviewed_preview_without_rerunning_sam(
    ctx,
    monkeypatch,
):
    client, _, sample_ids = ctx

    class ChangingSegmenter:
        model_id = config.SEGMENT_MODEL
        revision = config.SEGMENT_REVISION

        def __init__(self):
            self.calls = 0

        def segment(self, image, *, points=None, labels=None, box=None):
            self.calls += 1
            width, height = image.size
            mask = np.zeros((height, width), dtype=bool)
            if self.calls == 1:
                mask[5:25, 10:30] = True
            else:
                mask[30:50, 45:70] = True
            return mask, 0.91

    segmenter = ChangingSegmenter()
    monkeypatch.setattr(segment_ml, "get_segmenter", lambda: segmenter)
    prompt = {"box": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}}
    preview_response = client.post(
        "/api/segment",
        json={"sample_id": sample_ids[0], **prompt},
    )
    assert preview_response.status_code == 200
    preview = preview_response.json()
    preview_png = base64.b64decode(preview["mask_data_url"].split(",", 1)[1])

    accepted_response = client.post(
        f"/api/samples/{sample_ids[0]}/segment-annotations",
        json={
            **prompt,
            "label_name": "dog",
            "parent_name": "animal",
            "preview_token": preview["preview_token"],
            "mask_data_url": preview["mask_data_url"],
        },
    )

    assert accepted_response.status_code == 201
    accepted = accepted_response.json()
    assert segmenter.calls == 1
    persisted = client.get(accepted["mask_url"])
    assert persisted.status_code == 200
    assert persisted.content == preview_png
    client.delete(f"/api/segment-annotations/{accepted['id']}")


def test_accept_rejects_a_prompt_without_reviewed_preview_evidence(ctx):
    client, _, sample_ids = ctx

    response = client.post(
        f"/api/samples/{sample_ids[0]}/segment-annotations",
        json=_box_body(),
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Generate and review a segment preview before accepting this annotation"
    )


def test_accept_rejects_mask_bytes_that_differ_from_the_reviewed_preview(ctx):
    client, conn, sample_ids = ctx
    sample_id = sample_ids[0]
    reviewed = _reviewed_body(client, sample_id)
    prefix, encoded = reviewed["mask_data_url"].split(",", 1)
    mask_png = bytearray(base64.b64decode(encoded))
    mask_png[-1] ^= 1
    reviewed["mask_data_url"] = (
        f"{prefix},{base64.b64encode(mask_png).decode('ascii')}"
    )
    before = conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE sample_id = ?",
        (sample_id,),
    ).fetchone()[0]

    response = client.post(
        f"/api/samples/{sample_id}/segment-annotations",
        json=reviewed,
    )

    assert response.status_code == 422
    assert "mask bytes do not match" in response.json()["detail"]
    after = conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE sample_id = ?",
        (sample_id,),
    ).fetchone()[0]
    assert after == before


def test_accept_rejects_prompt_geometry_changed_after_preview(ctx):
    client, _, sample_ids = ctx
    sample_id = sample_ids[0]
    reviewed = _reviewed_body(client, sample_id)
    reviewed["box"] = {"x": 0.2, "y": 0.1, "w": 0.7, "h": 0.8}

    response = client.post(
        f"/api/samples/{sample_id}/segment-annotations",
        json=reviewed,
    )

    assert response.status_code == 422
    assert "segment preview prompt does not match" in response.json()["detail"]


def test_accept_rejects_when_source_image_changed_after_preview(ctx):
    client, conn, sample_ids = ctx
    sample_id = sample_ids[0]
    reviewed = _reviewed_body(client, sample_id)
    filename = conn.execute(
        "SELECT filename FROM samples WHERE id = ?",
        (sample_id,),
    ).fetchone()["filename"]
    image_path = config.IMAGES_DIR / filename
    original = image_path.read_bytes()
    try:
        Image.new("RGB", (80, 60), (1, 2, 3)).save(image_path, "JPEG")
        response = client.post(
            f"/api/samples/{sample_id}/segment-annotations",
            json=reviewed,
        )
    finally:
        image_path.write_bytes(original)

    assert response.status_code == 422
    assert "source image changed" in response.json()["detail"]


def test_preview_uses_real_taxonomy_paths_and_rejects_conflicts(ctx):
    client, conn, sample_ids = ctx
    root = conn.execute(
        "INSERT INTO object_labels(name, parent_id) VALUES ('living thing', NULL)"
    ).lastrowid
    conn.execute(
        "INSERT INTO object_labels(name, parent_id) VALUES ('mammal', ?)",
        (root,),
    )
    conn.commit()

    nested = client.post("/api/segment", json={
        "sample_id": sample_ids[0],
        "box": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8},
        "label_name": "wolf",
        "parent_name": "mammal",
    })
    assert nested.status_code == 200
    assert nested.json()["label_path"] == ["living thing", "mammal", "wolf"]

    conflict = client.post("/api/segment", json={
        "sample_id": sample_ids[0],
        "box": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8},
        "label_name": "dog",
        "parent_name": "vehicle",
    })
    assert conflict.status_code == 409

    conn.execute("DELETE FROM object_labels WHERE name = 'mammal'")
    conn.execute("DELETE FROM object_labels WHERE name = 'living thing'")
    conn.commit()


def test_prompt_validation_rejects_invalid_editor_state(ctx):
    client, _, sample_ids = ctx
    assert client.post("/api/segment", json={
        "sample_id": sample_ids[0],
        "points": [{"x": 0.5, "y": 0.5, "label": 0}],
    }).status_code == 422
    assert client.post("/api/segment", json={
        "sample_id": sample_ids[0],
        "points": [{"x": 0.5, "y": 0.5, "label": True}],
    }).status_code == 422
    assert client.post("/api/segment", json={
        "sample_id": sample_ids[0],
        "box": {"x": 0.9, "y": 0.1, "w": 0.2, "h": 0.5},
    }).status_code == 422
    assert client.post(
        f"/api/samples/{sample_ids[0]}/segment-annotations",
        json=_reviewed_body(
            client,
            sample_ids[0],
            proposal_token="not-a-valid-server-issued-proposal-token",
        ),
    ).status_code == 422


def test_accept_list_serve_and_delete_mask(ctx):
    client, conn, sample_ids = ctx
    source = _proposal_source()
    response = client.post(
        f"/api/samples/{sample_ids[0]}/segment-annotations",
        json=_reviewed_body(
            client,
            sample_ids[0],
            label_name="labrador",
            parent_name="animal",
            proposal_token=_proposal_token(sample_ids[0]),
        ))
    assert response.status_code == 201
    annotation = response.json()
    annotation_id = annotation["id"]
    assert annotation["kind"] == "mask"
    assert annotation["label_path"] == ["animal", "labrador"]
    assert annotation["parent_name"] == "animal"
    assert annotation["mask_data_url"] is None
    assert annotation["mask_url"] == f"/api/annotations/{annotation_id}/mask"
    assert annotation["cutout_url"] == (
        f"/api/annotations/{annotation_id}/cutout"
    )
    assert annotation["artifact_package_url"] == (
        f"/api/annotations/{annotation_id}/export"
    )
    assert annotation["model_id"] == config.SEGMENT_MODEL
    assert annotation["model_revision"] == config.SEGMENT_REVISION
    assert annotation["box"] == _box_body()["box"]
    assert annotation["proposal_source"] == source
    listed = client.get(
        f"/api/samples/{sample_ids[0]}/segment-annotations").json()
    assert [row["id"] for row in listed] == [annotation_id]
    mask_response = client.get(annotation["mask_url"])
    assert mask_response.status_code == 200
    assert mask_response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(mask_response.content)).size == (80, 60)
    stored = conn.execute(
        "SELECT model_id, model_revision, prompt_json, proposal_json "
        "FROM annotation_masks "
        "WHERE annotation_id = ?", (annotation_id,)).fetchone()
    assert stored["model_id"] == config.SEGMENT_MODEL
    assert stored["model_revision"] == config.SEGMENT_REVISION
    assert json.loads(stored["prompt_json"])["box"] == _box_body()["box"]
    assert json.loads(stored["proposal_json"]) == source
    assert client.delete(
        f"/api/segment-annotations/{annotation_id}").status_code == 200
    assert conn.execute(
        "SELECT 1 FROM annotation_masks WHERE annotation_id = ?",
        (annotation_id,)).fetchone() is None


def test_accepted_mask_exports_transparent_cutout_and_atomic_package(ctx):
    client, conn, sample_ids = ctx
    annotation = client.post(
        f"/api/samples/{sample_ids[0]}/segment-annotations",
        json=_reviewed_body(client, sample_ids[0]),
    ).json()
    annotation_id = annotation["id"]

    # Use an irregular accepted mask so the alpha contract is observable,
    # rather than merely checking that a rectangular crop has four channels.
    custom_mask = np.zeros((60, 80), dtype=np.uint8)
    custom_mask[10:50, 20:70] = 255
    custom_mask[25:35, 40:50] = 0
    encoded_mask = io.BytesIO()
    Image.fromarray(custom_mask, mode="L").save(encoded_mask, "PNG")
    mask_png = encoded_mask.getvalue()
    conn.execute(
        "UPDATE annotation_masks SET png = ?, width = 80, height = 60 "
        "WHERE annotation_id = ?",
        (mask_png, annotation_id),
    )
    conn.commit()

    cutout_response = client.get(
        f"/api/annotations/{annotation_id}/cutout",
    )
    assert cutout_response.status_code == 200
    assert cutout_response.headers["content-type"] == "image/png"
    assert cutout_response.headers["cache-control"] == "no-store"
    assert cutout_response.headers["x-content-type-options"] == "nosniff"
    assert "attachment;" in cutout_response.headers["content-disposition"]
    cutout = Image.open(io.BytesIO(cutout_response.content))
    cutout.load()
    assert cutout.mode == "RGBA"
    assert cutout.size == (50, 40)
    assert cutout.getchannel("A").getpixel((25, 20)) == 0
    assert cutout.getchannel("A").getpixel((1, 1)) == 255
    with Image.open(config.IMAGES_DIR / "segment_0.jpg") as source:
        source.load()
        assert cutout.getpixel((1, 1))[:3] == source.convert("RGB").getpixel((21, 11))

    package_response = client.get(
        f"/api/annotations/{annotation_id}/export",
    )
    assert package_response.status_code == 200
    assert package_response.headers["content-type"] == "application/zip"
    assert package_response.headers["cache-control"] == "no-store"
    assert package_response.headers["x-content-type-options"] == "nosniff"
    with zipfile.ZipFile(io.BytesIO(package_response.content)) as archive:
        names = sorted(archive.namelist())
        assert len(names) == 3
        assert names[0].endswith("-cutout.png")
        assert names[1].endswith("-manifest.json")
        assert names[2].endswith("-mask.png")
        manifest = json.loads(archive.read(names[1]))
        packaged_cutout = archive.read(names[0])
        packaged_mask = archive.read(names[2])

    assert manifest["format"] == "cvde.segment-annotation-export"
    assert manifest["version"] == 3
    assert manifest["annotation"]["id"] == annotation_id
    assert manifest["derivation"]["operations"] == [
        "Image.getbbox",
        "Image.crop",
        "Image.putalpha",
    ]
    assert manifest["derivation"]["cutout_bbox_pixels"] == {
        "left": 20,
        "upper": 10,
        "right": 70,
        "lower": 50,
    }
    assert manifest["artifacts"]["mask"]["sha256"] == hashlib.sha256(
        packaged_mask,
    ).hexdigest()
    assert manifest["artifacts"]["cutout"]["sha256"] == hashlib.sha256(
        packaged_cutout,
    ).hexdigest()
    assert packaged_mask == mask_png
    assert packaged_cutout == cutout_response.content
    client.delete(f"/api/segment-annotations/{annotation_id}")


def test_export_manifest_uses_one_sqlite_snapshot(ctx, monkeypatch):
    client, _, sample_ids = ctx
    annotation = client.post(
        f"/api/samples/{sample_ids[0]}/segment-annotations",
        json=_reviewed_body(client, sample_ids[0]),
    ).json()
    annotation_id = annotation["id"]
    reader = db.connect()
    writer = db.connect()
    originalannotation_out = annotations_api.annotation_out
    deleted = False

    def delete_between_pixel_and_manifest_reads(conn, row):
        nonlocal deleted
        if not deleted:
            writer.execute(
                "DELETE FROM annotations WHERE id = ?",
                (annotation_id,),
            )
            writer.commit()
            deleted = True
        return originalannotation_out(conn, row)

    monkeypatch.setattr(
        annotations_api,
        "annotation_out",
        delete_between_pixel_and_manifest_reads,
    )
    try:
        response = annotations_api.export_annotation_package(
            annotation_id,
            reader,
        )
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
            manifest_name = next(
                name for name in archive.namelist() if name.endswith("-manifest.json")
            )
            manifest = json.loads(archive.read(manifest_name))

        assert deleted
        assert not reader.in_transaction
        assert manifest["annotation"]["model_id"] == config.SEGMENT_MODEL
        assert manifest["annotation"]["model_revision"] == config.SEGMENT_REVISION
        assert manifest["annotation"]["label_path"] == ["animal", "dog"]
        assert writer.execute(
            "SELECT 1 FROM annotations WHERE id = ?",
            (annotation_id,),
        ).fetchone() is None
    finally:
        reader.close()
        writer.close()


def test_cutout_rejects_non_mask_and_dimension_drift(ctx):
    client, conn, sample_ids = ctx
    rect = client.post(
        f"/api/samples/{sample_ids[0]}/annotations",
        json={"kind": "rect", "geometry": _box_body()["box"]},
    ).json()
    assert client.get(
        f"/api/annotations/{rect['id']}/cutout",
    ).status_code == 404
    client.delete(f"/api/annotations/{rect['id']}")

    annotation = client.post(
        f"/api/samples/{sample_ids[0]}/segment-annotations",
        json=_reviewed_body(client, sample_ids[0]),
    ).json()
    conn.execute(
        "UPDATE annotations SET kind = 'rect' WHERE id = ?",
        (annotation["id"],),
    )
    conn.commit()
    invalid_kind = client.get(
        f"/api/annotations/{annotation['id']}/cutout",
    )
    assert invalid_kind.status_code == 503
    assert invalid_kind.json()["detail"] == (
        "Accepted mask has an invalid annotation kind"
    )
    conn.execute(
        "UPDATE annotations SET kind = 'mask' WHERE id = ?",
        (annotation["id"],),
    )
    conn.execute(
        "UPDATE annotation_masks SET width = 79 WHERE annotation_id = ?",
        (annotation["id"],),
    )
    conn.commit()
    response = client.get(
        f"/api/annotations/{annotation['id']}/cutout",
    )
    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Accepted mask dimensions do not match its source image"
    )
    client.delete(f"/api/segment-annotations/{annotation['id']}")


def test_cutout_rejects_empty_and_non_png_masks(ctx):
    client, conn, sample_ids = ctx
    annotation = client.post(
        f"/api/samples/{sample_ids[0]}/segment-annotations",
        json=_reviewed_body(client, sample_ids[0]),
    ).json()
    annotation_id = annotation["id"]

    empty_file = io.BytesIO()
    Image.new("L", (80, 60), 0).save(empty_file, "PNG")
    conn.execute(
        "UPDATE annotation_masks SET png = ? WHERE annotation_id = ?",
        (empty_file.getvalue(), annotation_id),
    )
    conn.commit()
    empty_response = client.get(
        f"/api/annotations/{annotation_id}/cutout",
    )
    assert empty_response.status_code == 503
    assert empty_response.json()["detail"] == (
        "Accepted mask contains no foreground pixels"
    )

    non_binary_file = io.BytesIO()
    Image.new("L", (80, 60), 128).save(non_binary_file, "PNG")
    conn.execute(
        "UPDATE annotation_masks SET png = ? WHERE annotation_id = ?",
        (non_binary_file.getvalue(), annotation_id),
    )
    conn.commit()
    non_binary_response = client.get(
        f"/api/annotations/{annotation_id}/cutout",
    )
    assert non_binary_response.status_code == 503
    assert non_binary_response.json()["detail"] == "Accepted mask is not binary"

    jpeg_file = io.BytesIO()
    Image.new("L", (80, 60), 255).save(jpeg_file, "JPEG")
    conn.execute(
        "UPDATE annotation_masks SET png = ? WHERE annotation_id = ?",
        (jpeg_file.getvalue(), annotation_id),
    )
    conn.commit()
    jpeg_response = client.get(
        f"/api/annotations/{annotation_id}/cutout",
    )
    assert jpeg_response.status_code == 503
    assert jpeg_response.json()["detail"] == (
        "Accepted mask is not a one-channel PNG artifact"
    )
    client.delete(f"/api/segment-annotations/{annotation_id}")


def test_detector_provenance_must_be_server_issued_and_match_prompt(ctx):
    client, _, sample_ids = ctx
    sample_id = sample_ids[0]
    token = _proposal_token(sample_id)

    payload, signature = token.split(".", 1)
    tampered = (
        f"{payload}."
        f"{'A' if signature[0] != 'A' else 'B'}{signature[1:]}"
    )
    invalid = client.post(
        f"/api/samples/{sample_id}/segment-annotations",
        json=_reviewed_body(
            client,
            sample_id,
            proposal_token=tampered,
        ),
    )
    assert invalid.status_code == 422
    assert "Run the detector again" in invalid.json()["detail"]

    another_sample = client.post(
        f"/api/samples/{sample_ids[1]}/segment-annotations",
        json=_reviewed_body(
            client,
            sample_ids[1],
            proposal_token=token,
        ),
    )
    assert another_sample.status_code == 422
    assert "another sample" in another_sample.json()["detail"]

    different_box = client.post(
        f"/api/samples/{sample_id}/segment-annotations",
        json=_reviewed_body(
            client,
            sample_id,
            box={"x": 0.2, "y": 0.1, "w": 0.7, "h": 0.8},
            proposal_token=token,
        ),
    )
    assert different_box.status_code == 422
    assert "does not match" in different_box.json()["detail"]

    client_authored = client.post(
        f"/api/samples/{sample_id}/segment-annotations",
        json=_reviewed_body(
            client,
            sample_id,
            proposal_source=_proposal_source(),
        ),
    )
    assert client_authored.status_code == 422


def test_detector_proposal_token_expires():
    source = DetectionProposalSource.model_validate(_proposal_source())
    token = issue_detection_proposal(7, source, issued_at=100)

    with pytest.raises(ProposalTokenError, match="expired"):
        resolve_detection_proposal(
            token,
            sample_id=7,
            prompt_box=SegmentBox.model_validate(_box_body()["box"]),
            now=100 + TOKEN_TTL_SECONDS + 1,
        )


def test_search_by_annotation_blends_leaf_label_and_excludes_source(ctx, monkeypatch):
    client, _, sample_ids = ctx
    annotation = client.post(
        f"/api/samples/{sample_ids[0]}/segment-annotations",
        json=_reviewed_body(client, sample_ids[0]),
    ).json()
    encoder = MockEncoder()
    images = [
        Image.open(config.IMAGES_DIR / f"segment_{i}.jpg").convert("RGB")
        for i in range(4)
    ]
    index = EmbeddingIndex(
        np.array(sample_ids, dtype=np.int64),
        encoder.encode_images(images))
    import app.api.search as search_api

    monkeypatch.setattr(
        search_api,
        "get_retrieval_bundle",
        lambda: SimpleNamespace(
            encoder=encoder,
            image_index=index,
            caption_index=None,
        ),
    )
    response = client.post(
        "/api/search/by-annotation",
        json={"annotation_id": annotation["id"], "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["mode_used"] == "segment"
    assert body["score_basis"] == "segment_composed"
    assert "leaf label 'dog'" in body["message"]
    assert sample_ids[0] not in [item["id"] for item in body["items"]]
    assert [item["score"] for item in body["items"]] == sorted(
        [item["score"] for item in body["items"]], reverse=True)
    client.delete(f"/api/segment-annotations/{annotation['id']}")


def test_corrupt_detector_provenance_is_not_reported_as_valid(ctx):
    client, conn, sample_ids = ctx
    annotation = client.post(
        f"/api/samples/{sample_ids[0]}/segment-annotations",
        json=_reviewed_body(client, sample_ids[0]),
    ).json()
    conn.execute(
        "UPDATE annotation_masks SET proposal_json = '{}' "
        "WHERE annotation_id = ?",
        (annotation["id"],),
    )
    conn.commit()

    response = client.get(
        f"/api/samples/{sample_ids[0]}/segment-annotations",
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Persisted detector proposal provenance is invalid"
    )
    client.delete(f"/api/segment-annotations/{annotation['id']}")


def test_model_load_uses_one_resolved_snapshot_offline(monkeypatch, tmp_path):
    calls = []
    snapshot = SimpleNamespace(
        model_id=config.SEGMENT_MODEL,
        revision=config.SEGMENT_REVISION,
        snapshot_path=tmp_path,
    )

    class ProcessorFactory:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("processor", model_id, kwargs))
            return object()

    class LoadedModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    class ModelFactory:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            return LoadedModel(), {}

    fake_transformers = ModuleType("transformers")
    fake_transformers.Sam2Processor = ProcessorFactory
    fake_transformers.Sam2Model = ModelFactory
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    loaded = segment_ml.Segmenter(snapshot)
    assert len(calls) == 2
    assert {call[1] for call in calls} == {str(tmp_path)}
    assert all(call[2]["local_files_only"] is True for call in calls)
    assert loaded.model_id == config.SEGMENT_MODEL
    assert loaded.revision == config.SEGMENT_REVISION


def test_segment_snapshot_resolution_is_commit_bound(monkeypatch, tmp_path):
    from app.ml import providers

    calls = []
    expected = SimpleNamespace(
        model_id=config.SEGMENT_MODEL,
        revision=config.SEGMENT_REVISION,
        snapshot_path=tmp_path,
    )

    def resolve(model_id, revision=None, local_files_only=False):
        calls.append((model_id, revision, local_files_only))
        return expected

    monkeypatch.setattr(providers, "resolve_model_snapshot", resolve)
    assert segment_ml._resolve_snapshot() is expected
    assert calls == [(
        config.SEGMENT_MODEL,
        config.SEGMENT_REVISION,
        True,
    )]


def test_segmenter_rejects_a_resolved_commit_mismatch(monkeypatch, tmp_path):
    from app.ml import providers

    mismatched = SimpleNamespace(
        model_id=config.SEGMENT_MODEL,
        revision="f" * 40,
        snapshot_path=tmp_path,
    )
    monkeypatch.setattr(
        providers,
        "resolve_model_snapshot",
        lambda *_args, **_kwargs: mismatched,
    )

    with pytest.raises(RuntimeError, match="does not match CVDE_SEGMENT_REVISION"):
        segment_ml._resolve_snapshot()


def test_loaded_segmenter_availability_does_not_probe_snapshot(monkeypatch):
    loaded = SimpleNamespace(
        model_id=config.SEGMENT_MODEL,
        revision=config.SEGMENT_REVISION,
    )
    monkeypatch.setattr(segment_ml, "_segmenter", loaded)
    monkeypatch.setattr(
        segment_ml,
        "_resolve_snapshot",
        lambda *_args, **_kwargs: pytest.fail("loaded model must not probe cache"),
    )

    state = segment_ml.segment_availability()

    assert state.ready is True
    assert state.model == config.SEGMENT_MODEL
    assert state.revision == config.SEGMENT_REVISION


def test_segmenter_rejects_a_moving_revision(monkeypatch):
    monkeypatch.setattr(segment_ml, "_segmenter", None)
    monkeypatch.setattr(segment_ml, "_failed_at", None)
    monkeypatch.setattr(segment_ml, "SEGMENT_REVISION", "main")
    ready, reason = segment_ml.segment_ready()
    assert ready is False
    assert "full 40-character Hugging Face commit" in reason


def test_segment_status_reports_load_cooldown_without_resolving(monkeypatch):
    monkeypatch.setattr(segment_ml, "_segmenter", None)
    monkeypatch.setattr(segment_ml, "_failed_at", segment_ml.time.monotonic())
    monkeypatch.setattr(segment_ml, "_failed_reason", "segmenter load failed: boom")
    monkeypatch.setattr(
        segment_ml,
        "_resolve_snapshot",
        lambda *_args, **_kwargs: pytest.fail("cooldown must not probe the snapshot"),
    )

    state = segment_ml.segment_availability()

    assert state.ready is False
    assert state.reason == "segmenter load failed: boom"
    assert state.revision == config.SEGMENT_REVISION


def test_segment_status_resolves_snapshot_once(monkeypatch, tmp_path):
    from app.api.segment import segment_status

    calls = []
    snapshot = SimpleNamespace(
        model_id=config.SEGMENT_MODEL,
        revision=config.SEGMENT_REVISION,
        snapshot_path=tmp_path,
    )

    def resolve(revision=None):
        calls.append(revision)
        return snapshot

    monkeypatch.setattr(segment_ml, "_segmenter", None)
    monkeypatch.setattr(segment_ml, "_failed_at", None)
    monkeypatch.setattr(segment_ml, "_resolve_snapshot", resolve)

    body = segment_status()

    assert body["ready"] is True
    assert body["revision"] == config.SEGMENT_REVISION
    assert "72 ms" in body["measured"]
    assert calls == [config.SEGMENT_REVISION]


def test_segment_status_does_not_reuse_measurement_for_an_override(monkeypatch):
    from app.api import segment as segment_api
    from app.ml.segment import SegmentAvailability

    monkeypatch.setattr(
        segment_api.segment_ml,
        "segment_availability",
        lambda: SegmentAvailability(
            ready=True,
            reason=None,
            model="example/custom-segmenter",
            revision="1" * 40,
        ),
    )

    assert segment_api.segment_status()["measured"] == (
        "not measured for the configured segmenter artifact"
    )


def test_legacy_mask_migration_preserves_unknown_revision():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE samples (
            id INTEGER PRIMARY KEY,
            caption_consistency REAL
        );
        CREATE TABLE captions (
            id INTEGER PRIMARY KEY,
            sample_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            agreement REAL
        );
        CREATE VIRTUAL TABLE captions_fts USING fts5(
            text, content='captions', content_rowid='id',
            tokenize='porter unicode61'
        );
        CREATE TABLE annotation_masks (
            annotation_id INTEGER PRIMARY KEY,
            png BLOB NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            model_id TEXT NOT NULL,
            prompt_json TEXT NOT NULL,
            predicted_iou REAL NOT NULL
        );
        INSERT INTO annotation_masks
            (annotation_id, png, width, height, model_id, prompt_json, predicted_iou)
        VALUES (1, X'00', 1, 1, 'legacy/model', '{}', 0.5);
    """)
    db._migrate(conn)
    row = conn.execute(
        "SELECT model_id, model_revision, proposal_json FROM annotation_masks "
        "WHERE annotation_id = 1"
    ).fetchone()
    assert row["model_id"] == "legacy/model"
    assert row["model_revision"] is None
    assert row["proposal_json"] is None
    conn.close()


def test_segment_openapi_exposes_provenance_contract():
    schema = app.openapi()
    responses = schema["paths"]["/api/segment/status"]["get"]["responses"]
    assert responses["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ModelCapabilityStatus")
    annotation = schema["components"]["schemas"]["AnnotationOut"]["properties"]
    assert annotation["prompt"]["anyOf"][0]["$ref"].endswith("/SegmentPrompt")
    assert annotation["proposal_source"]["anyOf"][0]["$ref"].endswith(
        "/DetectionProposalSource"
    )
    accept = schema["components"]["schemas"]["SegmentAcceptRequest"]["properties"]
    assert "proposal_token" in accept
    assert "preview_token" in accept
    assert "mask_data_url" in accept
    assert "proposal_source" not in accept
    preview = schema["components"]["schemas"]["SegmentPreview"]["properties"]
    assert {"preview_token", "source_sha256", "mask_sha256"} <= preview.keys()


def test_segmenter_serializes_forwards():
    torch = pytest.importorskip("torch")

    class Inputs(dict):
        def to(self, _device):
            return self

    class Processor:
        def __call__(self, **_kwargs):
            return Inputs(original_sizes=torch.tensor([[8, 8]]))

        def post_process_masks(self, _masks, _sizes):
            return [torch.ones((1, 1, 8, 8), dtype=torch.bool)]

    class Model:
        def __init__(self):
            self.guard = threading.Lock()
            self.inside = 0
            self.max_inside = 0

        def __call__(self, **_kwargs):
            with self.guard:
                self.inside += 1
                self.max_inside = max(self.max_inside, self.inside)
            time.sleep(0.02)
            with self.guard:
                self.inside -= 1
            return SimpleNamespace(
                pred_masks=torch.ones((1, 1, 1, 2, 2)),
                iou_scores=torch.tensor([[[0.9]]]))

    segmenter = object.__new__(segment_ml.Segmenter)
    segmenter.device = "cpu"
    segmenter.processor = Processor()
    segmenter.model = Model()
    segmenter._torch = torch
    segmenter._infer = threading.Lock()
    errors = []

    def run():
        try:
            segmenter.segment(Image.new("RGB", (8, 8)),
                              points=[(4, 4)], labels=[1])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not errors
    assert segmenter.model.max_inside == 1
