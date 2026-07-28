"""SAM2 preview, accepted mask persistence, hierarchy and segment retrieval."""
import base64
import io
import json
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, db
from app.main import app
from app.ml import segment as segment_ml
from app.ml.index import EmbeddingIndex
from tests.fake_provider import MockEncoder


class FakeSegmenter:
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
    assert body["mask_width"] == 80 and body["mask_height"] == 60
    prefix, encoded = body["mask_data_url"].split(",", 1)
    assert prefix == "data:image/png;base64"
    assert Image.open(io.BytesIO(base64.b64decode(encoded))).size == (80, 60)


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


def test_accept_list_serve_and_delete_mask(ctx):
    client, conn, sample_ids = ctx
    response = client.post(
        f"/api/samples/{sample_ids[0]}/segment-annotations",
        json=_box_body(label_name="labrador", parent_name="animal"))
    assert response.status_code == 201
    annotation = response.json()
    annotation_id = annotation["id"]
    assert annotation["kind"] == "mask"
    assert annotation["label_path"] == ["animal", "labrador"]
    assert annotation["parent_name"] == "animal"
    assert annotation["mask_data_url"] is None
    assert annotation["mask_url"] == f"/api/annotations/{annotation_id}/mask"
    assert annotation["box"] == _box_body()["box"]
    listed = client.get(
        f"/api/samples/{sample_ids[0]}/segment-annotations").json()
    assert [row["id"] for row in listed] == [annotation_id]
    mask_response = client.get(annotation["mask_url"])
    assert mask_response.status_code == 200
    assert mask_response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(mask_response.content)).size == (80, 60)
    stored = conn.execute(
        "SELECT model_id, prompt_json FROM annotation_masks "
        "WHERE annotation_id = ?", (annotation_id,)).fetchone()
    assert stored["model_id"] == config.SEGMENT_MODEL
    assert json.loads(stored["prompt_json"])["box"] == _box_body()["box"]
    assert client.delete(
        f"/api/segment-annotations/{annotation_id}").status_code == 200
    assert conn.execute(
        "SELECT 1 FROM annotation_masks WHERE annotation_id = ?",
        (annotation_id,)).fetchone() is None


def test_search_by_annotation_blends_leaf_label_and_excludes_source(ctx, monkeypatch):
    client, _, sample_ids = ctx
    annotation = client.post(
        f"/api/samples/{sample_ids[0]}/segment-annotations",
        json=_box_body()).json()
    encoder = MockEncoder()
    images = [
        Image.open(config.IMAGES_DIR / f"segment_{i}.jpg").convert("RGB")
        for i in range(4)
    ]
    index = EmbeddingIndex(
        np.array(sample_ids, dtype=np.int64),
        encoder.encode_images(images))
    import app.api.search as search_api

    monkeypatch.setattr(search_api, "get_index", lambda: index)
    monkeypatch.setattr(search_api, "get_embedder", lambda: encoder)
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


def test_model_load_is_offline_only(monkeypatch):
    calls = []

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
    segment_ml.Segmenter()
    assert len(calls) == 2
    assert all(call[2]["local_files_only"] is True for call in calls)


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
