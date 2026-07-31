"""Region evidence: a marked area of an existing sample as a positive or
negative query. The server crops the original file from normalized geometry —
reproducible from the request alone — and the negative role ranks by distance
and never returns its own source image.

    cd backend && pytest tests/test_region_search.py
"""
import sys
import threading
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, db
from app.main import app
from app.ml import index as index_mod
from app.ml import providers
from app.schemas import SegmentBox
from tests.fake_provider import MockEncoder


@pytest.fixture(scope="module")
def ctx():
    conn = db.connect()
    db.init_db(conn)
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sids = []
    colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]
    for i, color in enumerate(colors):
        fname = f"region_{i}.jpg"
        Image.new("RGB", (320, 240), color).save(config.IMAGES_DIR / fname, "JPEG")
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
            "VALUES ('flickr8k', ?, 'train', 320, 240, 1)", (fname,))
        sids.append(cur.lastrowid)
        conn.execute("INSERT INTO captions(sample_id, idx, text) VALUES (?, 0, ?)",
                     (sids[-1], f"region probe {i}"))
    conn.commit()

    # Index vectors = what MockEncoder produces for each FULL image, so the
    # source sample's own full-frame vector is its nearest neighbour and the
    # ordering is analytically checkable.
    enc = MockEncoder()
    vecs = np.concatenate([
        enc.encode_images([Image.open(config.IMAGES_DIR / f"region_{i}.jpg"
                                      ).convert("RGB")]) for i in range(4)])
    config.EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EMB_DIR / "sample_ids.npy", np.array(sids, dtype=np.int64))
    np.save(config.EMB_DIR / "image_embeddings.npy", vecs)
    providers.invalidate_providers()
    index_mod.invalidate_index()

    import app.api.search as search_api
    original = search_api.get_retrieval_bundle
    runtime = SimpleNamespace(
        encoder=enc,
        image_index=index_mod.EmbeddingIndex(
            np.array(sids, dtype=np.int64), vecs),
        caption_index=None,
    )
    search_api.get_retrieval_bundle = lambda: runtime
    with TestClient(app) as client:
        yield client, sids
    search_api.get_retrieval_bundle = original
    for i, sid in enumerate(sids):
        conn.execute("DELETE FROM captions WHERE sample_id = ?", (sid,))
        conn.execute("DELETE FROM samples WHERE id = ?", (sid,))
        (config.IMAGES_DIR / f"region_{i}.jpg").unlink(missing_ok=True)
    conn.commit()
    conn.close()
    for f in ("sample_ids.npy", "image_embeddings.npy"):
        (config.EMB_DIR / f).unlink(missing_ok=True)
    providers.invalidate_providers()
    index_mod.invalidate_index()


def _req(client, **over):
    body = {"sample_id": None, "x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}
    body.update(over)
    return client.post("/api/search/by-region", json=body)


def test_positive_region_ranks_and_carries_the_basis(ctx):
    client, sids = ctx
    r = _req(client, sample_id=sids[0])
    assert r.status_code == 200
    body = r.json()
    assert body["mode_used"] == "composed" and body["score_basis"] == "composed"
    assert body["items"], "a region query must rank the corpus"
    scores = [it["score"] for it in body["items"]]
    assert scores == sorted(scores, reverse=True)


def test_negative_region_excludes_its_source_and_explains_itself(ctx):
    client, sids = ctx
    body = _req(client, sample_id=sids[1], role="negative").json()
    ids = [it["id"] for it in body["items"]]
    assert ids and sids[1] not in ids, "away-from-X must not return X"
    assert "distance from the marked region" in (body["message"] or "")


def test_geometry_is_validated(ctx):
    client, sids = ctx
    assert _req(client, sample_id=sids[0], x=0.9, w=0.5).status_code == 422
    assert _req(client, sample_id=sids[0], w=0.001).status_code == 422
    # NaN can only arrive as raw wire bytes — the test client's own encoder
    # rightly refuses to produce it.
    raw = ('{"sample_id": %d, "x": NaN, "y": 0.1, "w": 0.5, "h": 0.5}'
           % sids[0]).encode()
    r = client.post("/api/search/by-region", content=raw,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert _req(client, sample_id=999_999_999).status_code == 404


def test_detect_status_names_the_enabling_command(ctx, monkeypatch):
    """The detector is an optional layer like every other: absent weights are
    a named reason with the fetch command, never a 500 or a silent download."""
    client, sids = ctx
    from app.ml import detect as detect_ml

    def missing_snapshot():
        raise RuntimeError("snapshot is incomplete")

    monkeypatch.setattr(detect_ml, "_resolve_snapshot", missing_snapshot)
    st = client.get("/api/detect/status").json()
    assert st["ready"] is False and "snapshot_download" in st["reason"]
    assert st["revision"] == config.DETECT_REVISION
    r = client.post("/api/detect", json={"sample_id": sids[0]})
    assert r.status_code == 503
    assert "snapshot_download" in r.json()["detail"]


def test_detector_rejects_a_moving_revision(monkeypatch):
    from app.ml import detect as detect_ml

    monkeypatch.setattr(detect_ml, "_detector", None)
    monkeypatch.setattr(detect_ml, "_failed_at", None)
    monkeypatch.setattr(detect_ml, "DETECT_REVISION", "main")
    state = detect_ml.detector_availability()
    assert state.ready is False
    assert state.revision is None
    assert "full 40-character Hugging Face commit" in state.reason


def test_detect_validates_input(ctx):
    client, sids = ctx
    assert client.post("/api/detect", json={"sample_id": 0}).status_code == 422
    assert client.post(
        "/api/detect",
        json={"sample_id": sids[0], "queries": "x"},
    ).status_code == 422


def test_blank_detect_query_runs_the_fixed_phrase_bank(ctx, monkeypatch):
    client, sids = ctx
    from app.api import detect as detect_api
    from app.ml import detect as detect_ml

    received = []

    class FakeDetector:
        model_id = config.DETECT_MODEL
        revision = config.DETECT_REVISION

        def detect(self, _image, queries):
            received.append(queries)
            return []

    monkeypatch.setattr(detect_ml, "get_detector", lambda: FakeDetector())
    payloads = [
        {"sample_id": sids[0]},
        {"sample_id": sids[0], "queries": ""},
        {"sample_id": sids[0], "queries": "  \n  "},
    ]
    responses = [client.post("/api/detect", json=payload) for payload in payloads]

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert received == [detect_api.AUTO_DETECT_QUERIES] * len(payloads)
    assert detect_ml.phrases_from(received[0]) == [
        "a person", "an animal", "a vehicle", "an object",
    ]
    assert all(response.json()["queries"] == detect_api.AUTO_DETECT_QUERIES
               for response in responses)
    assert all("phrase alignment" in response.json()["note"]
               for response in responses)
    assert all("confidence" not in response.json()["note"].lower()
               for response in responses)


def test_phrases_from_matches_the_processor_candidate_label_contract():
    """The processor normalizes only a LIST of dot-free phrases; the helper
    turns the period-separated wire format into exactly that list."""
    from app.ml.detect import phrases_from

    assert phrases_from("dog") == ["dog"]
    assert phrases_from("a dog. a cat.") == ["a dog", "a cat"]
    assert phrases_from("...") == []


def test_detect_rejects_queries_with_no_phrases(ctx):
    """Pydantic rejects an all-period query before the availability check —
    a 422, never a 500 from an empty candidate-label list."""
    client, sids = ctx
    r = client.post("/api/detect", json={"sample_id": sids[0], "queries": "..."})
    assert r.status_code == 422
    assert "at least one phrase" in str(r.json()["detail"])


def test_detect_returns_server_bound_proposal_tokens(ctx, monkeypatch):
    client, sids = ctx
    from app.ml import detect as detect_ml
    from app.proposal_tokens import resolve_detection_proposal
    from app.schemas import SegmentBox

    class FakeDetector:
        model_id = config.DETECT_MODEL
        revision = config.DETECT_REVISION

        def detect(self, _image, _queries):
            return [{
                "x": 0.1,
                "y": 0.2,
                "w": 0.3,
                "h": 0.4,
                "label": "a dog",
                "score": 0.91,
            }]

    monkeypatch.setattr(detect_ml, "get_detector", lambda: FakeDetector())
    response = client.post(
        "/api/detect",
        json={"sample_id": sids[0], "queries": "a dog."},
    )

    assert response.status_code == 200
    box = response.json()["boxes"][0]
    assert box["proposal_token"].count(".") == 1
    source = resolve_detection_proposal(
        box["proposal_token"],
        sample_id=sids[0],
        prompt_box=SegmentBox(
            x=box["x"],
            y=box["y"],
            w=box["w"],
            h=box["h"],
        ),
    )
    assert source.model_revision == config.DETECT_REVISION
    assert source.queries == "a dog."
    assert source.original_label == "a dog"
    assert source.proposed_label == "dog"
    assert source.score == pytest.approx(0.91)


def test_detector_model_load_uses_one_resolved_snapshot_offline(
    monkeypatch,
    tmp_path,
):
    """A request may load cached weights, but must never fetch a checkpoint."""
    from app.ml import detect as detect_ml

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
            return LoadedModel()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoProcessor = ProcessorFactory
    fake_transformers.GroundingDinoForObjectDetection = ModelFactory
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    snapshot = SimpleNamespace(
        model_id=config.DETECT_MODEL,
        revision=config.DETECT_REVISION,
        snapshot_path=tmp_path,
    )
    loaded = detect_ml._Detector(snapshot)
    assert len(calls) == 2
    assert {call[1] for call in calls} == {str(tmp_path)}
    assert all(call[2]["local_files_only"] is True for call in calls)
    assert loaded.model_id == config.DETECT_MODEL
    assert loaded.revision == config.DETECT_REVISION


def test_detector_reports_no_regions_instead_of_raising():
    """Nothing above the threshold is an answer, not a crash.

    post_process_grounded_object_detection decodes one phrase per surviving
    query, and when none survive it still returns a single [""] — so pairing
    boxes with labels strictly saw 0 against 1 and raised, which reached the
    browser as a bare 500. Measured against transformers 5.14.1 with a query
    whose best score falls under the threshold.
    """
    import threading
    from contextlib import nullcontext
    from types import SimpleNamespace

    from PIL import Image

    from app.ml import detect as detect_ml

    class Empty:
        def tolist(self):
            return []

        def __len__(self):
            return 0

    class Inputs(dict):
        def __init__(self):
            super().__init__(input_ids=[])
            self.input_ids = self["input_ids"]

        def to(self, _device):
            return self

    class Processor:
        def __call__(self, **_kwargs):
            return Inputs()

        def post_process_grounded_object_detection(self, *_args, **_kwargs):
            return [{"boxes": Empty(), "scores": Empty(), "text_labels": [""]}]

    detector = object.__new__(detect_ml._Detector)
    detector.device = "cpu"
    detector.processor = Processor()
    detector.model = lambda **_kwargs: object()
    detector._torch = SimpleNamespace(no_grad=lambda: nullcontext())
    detector._infer = threading.Lock()

    assert detector.detect(Image.new("RGB", (100, 100)), "a zebra.") == []


def test_detector_clips_boxes_and_preserves_thin_positive_geometry():
    """Transformers rescales Grounding DINO boxes but does not clip them."""
    from app.ml import detect as detect_ml

    class Values:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

        def __len__(self):
            # Real post-processing hands back tensors, which are sized; the
            # detector asks how many boxes survived before it pairs them up.
            return len(self.values)

    class Inputs(dict):
        def __init__(self):
            super().__init__(input_ids=[])
            self.input_ids = self["input_ids"]

        def to(self, _device):
            return self

    class Processor:
        def __call__(self, **_kwargs):
            return Inputs()

        def post_process_grounded_object_detection(self, *_args, **_kwargs):
            return [{
                "boxes": Values([
                    [80.0, 10.0, 120.0, 50.0],
                    [10.0, 10.0, 10.004, 50.0],
                    [-10.0, -5.0, 20.0, 30.0],
                    [120.0, 10.0, 130.0, 20.0],
                    [float("nan"), 0.0, 10.0, 10.0],
                ]),
                "scores": Values([0.9, 0.8, 0.7, 0.6, 0.5]),
                "text_labels": ["edge", "thin", "clipped", "outside", "nan"],
            }]

    detector = object.__new__(detect_ml._Detector)
    detector.device = "cpu"
    detector.processor = Processor()
    detector.model = lambda **_kwargs: object()
    detector._torch = SimpleNamespace(no_grad=lambda: nullcontext())
    detector._infer = threading.Lock()

    boxes = detector.detect(Image.new("RGB", (100, 100)), "object.")

    assert [box["label"] for box in boxes] == ["edge", "thin", "clipped"]
    assert boxes[0]["x"] == pytest.approx(0.8)
    assert boxes[0]["w"] == pytest.approx(0.2)
    assert boxes[1]["w"] == pytest.approx(0.00004)
    assert boxes[2]["x"] == 0.0
    assert boxes[2]["y"] == 0.0
    for box in boxes:
        # The model layer emits geometry; the API layer adds taxonomy and the
        # required server-issued proposal token before validating DetectBoxOut.
        SegmentBox.model_validate(box)


def test_detector_snapshot_resolution_is_commit_bound(monkeypatch, tmp_path):
    from app.ml import detect as detect_ml

    calls = []
    expected = SimpleNamespace(
        model_id=config.DETECT_MODEL,
        revision=config.DETECT_REVISION,
        snapshot_path=tmp_path,
    )

    def resolve(model_id, revision=None, local_files_only=False):
        calls.append((model_id, revision, local_files_only))
        return expected

    monkeypatch.setattr(providers, "resolve_model_snapshot", resolve)
    assert detect_ml._resolve_snapshot() is expected
    assert calls == [(
        config.DETECT_MODEL,
        config.DETECT_REVISION,
        True,
    )]


def test_detector_rejects_a_resolved_commit_mismatch(monkeypatch, tmp_path):
    from app.ml import detect as detect_ml

    mismatched = SimpleNamespace(
        model_id=config.DETECT_MODEL,
        revision="f" * 40,
        snapshot_path=tmp_path,
    )
    monkeypatch.setattr(
        providers,
        "resolve_model_snapshot",
        lambda *_args, **_kwargs: mismatched,
    )

    with pytest.raises(RuntimeError, match="does not match CVDE_DETECT_REVISION"):
        detect_ml._resolve_snapshot()


def test_loaded_detector_availability_does_not_probe_snapshot(monkeypatch):
    from app.ml import detect as detect_ml

    loaded = SimpleNamespace(
        model_id=config.DETECT_MODEL,
        revision=config.DETECT_REVISION,
    )
    monkeypatch.setattr(detect_ml, "_detector", loaded)
    monkeypatch.setattr(
        detect_ml,
        "_resolve_snapshot",
        lambda *_args, **_kwargs: pytest.fail("loaded model must not probe cache"),
    )

    state = detect_ml.detector_availability()

    assert state.ready is True
    assert state.model == config.DETECT_MODEL
    assert state.revision == config.DETECT_REVISION


def test_detect_status_reports_load_cooldown_without_resolving(monkeypatch):
    from app.ml import detect as detect_ml

    monkeypatch.setattr(detect_ml, "_detector", None)
    monkeypatch.setattr(detect_ml, "_failed_at", detect_ml.time.monotonic())
    monkeypatch.setattr(detect_ml, "_failed_reason", "detector load failed: boom")
    monkeypatch.setattr(
        detect_ml,
        "_resolve_snapshot",
        lambda *_args, **_kwargs: pytest.fail("cooldown must not probe the snapshot"),
    )

    state = detect_ml.detector_availability()

    assert state.ready is False
    assert state.reason == "detector load failed: boom"
    assert state.revision == config.DETECT_REVISION


def test_detect_status_resolves_snapshot_once(monkeypatch, tmp_path):
    from app.api.detect import detect_status
    from app.ml import detect as detect_ml

    calls = []
    snapshot = SimpleNamespace(
        model_id=config.DETECT_MODEL,
        revision=config.DETECT_REVISION,
        snapshot_path=tmp_path,
    )

    def resolve(revision=None):
        calls.append(revision)
        return snapshot

    monkeypatch.setattr(detect_ml, "_detector", None)
    monkeypatch.setattr(detect_ml, "_failed_at", None)
    monkeypatch.setattr(detect_ml, "_resolve_snapshot", resolve)

    body = detect_status()

    assert body["ready"] is True
    assert body["revision"] == config.DETECT_REVISION
    assert "330 ms" in body["measured"]
    assert calls == [config.DETECT_REVISION]


def test_detect_status_does_not_reuse_measurement_for_an_override(monkeypatch):
    from app.api import detect as detect_api
    from app.ml.detect import DetectorAvailability

    monkeypatch.setattr(
        detect_api.detect_ml,
        "detector_availability",
        lambda: DetectorAvailability(
            ready=True,
            reason=None,
            model="example/custom-detector",
            revision="1" * 40,
        ),
    )

    assert detect_api.detect_status()["measured"] == (
        "not measured for the configured detector artifact"
    )


def test_detector_openapi_exposes_provenance_contract():
    schema = app.openapi()
    responses = schema["paths"]["/api/detect"]["post"]["responses"]
    assert responses["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/DetectResponse")
    detect_box = schema["components"]["schemas"]["DetectBoxOut"]
    assert "proposal_token" in detect_box["required"]
    status_responses = schema["paths"]["/api/detect/status"]["get"]["responses"]
    assert status_responses["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ModelCapabilityStatus")
