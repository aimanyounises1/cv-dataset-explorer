"""Composed search + scenario groups, on synthetic embeddings.

Same fixture discipline as test_with_embeddings: plant real index files, swap
the embedder behind the seam, restore the world on teardown. Three orthogonal
concepts (dog / soccer / bike), eight samples each, so scenario grouping has
real cluster structure to find.

    cd backend && pytest tests/test_composed_search.py
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db
from app.api import search as search_module
from app.api.search import SCENARIO_BASIS
from app.main import app
from app.ml.index import EmbeddingIndex, invalidate_index

CONCEPTS = ["dog", "soccer", "bike"]
SCENE = {"dog": "park", "soccer": "field", "bike": "trail"}
PER_CONCEPT = 8


def _vec(weights: dict[int, float]) -> np.ndarray:
    v = np.zeros(8, dtype=np.float32)
    for idx, w in weights.items():
        v[idx] = w
    return v / (np.linalg.norm(v) or 1.0)


class FakeEmbedder:
    def encode_texts(self, texts):
        out = []
        for t in texts:
            weights = {i: 1.0 for i, c in enumerate(CONCEPTS) if c in t.lower()}
            out.append(_vec(weights or {0: 0.5, 1: 0.5, 2: 0.5}))
        return np.stack(out)

    def encode_images(self, images):
        return np.stack([_vec({0: 1.0}) for _ in images])


@pytest.fixture(scope="module")
def ctx():
    """(client, {concept: [sample ids]})."""
    conn = db.connect()
    db.init_db(conn)
    by_concept: dict[str, list[int]] = {c: [] for c in CONCEPTS}
    ids, vecs = [], []
    for ci, concept in enumerate(CONCEPTS):
        for j in range(PER_CONCEPT):
            # One dog sample lives in the test split, for the filter test.
            split = "test" if (concept == "dog" and j == PER_CONCEPT - 1) else "train"
            cur = conn.execute(
                "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
                "VALUES ('flickr8k', ?, ?, 300, 200, 1)",
                (f"cmp_{concept}_{j}.jpg", split))
            sid = cur.lastrowid
            conn.execute("INSERT INTO captions(sample_id, idx, text) VALUES (?, 0, ?)",
                         (sid, f"a {concept} photo number {j}"))
            conn.execute(
                "INSERT INTO attributes(sample_id, grp, label, confidence) "
                "VALUES (?, 'scene', ?, 0.9)", (sid, SCENE[concept]))
            by_concept[concept].append(sid)
            ids.append(sid)
            # Dominant concept plus per-sample jitter, so no two vectors tie.
            vecs.append(_vec({ci: 1.0, 3 + (j % 5): 0.05 * (j + 1)}))
    conn.commit()
    EmbeddingIndex.save(np.array(ids), np.stack(vecs), kind="image")
    invalidate_index()
    real_get_embedder = search_module.get_embedder
    search_module.get_embedder = lambda: FakeEmbedder()
    try:
        with TestClient(app) as c:
            yield c, by_concept
    finally:
        search_module.get_embedder = real_get_embedder
        from app import config
        for f in config.EMB_DIR.glob("*.npy"):
            f.unlink(missing_ok=True)
        invalidate_index()
        qmarks = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM attributes WHERE sample_id IN ({qmarks})", ids)
        conn.execute(f"DELETE FROM samples WHERE id IN ({qmarks})", ids)
        conn.commit()
        conn.close()


def _ids(resp) -> list[int]:
    return [i["id"] for i in resp.json()["items"]]


def test_text_only_ranks_the_named_concept_first(ctx):
    client, s = ctx
    r = client.post("/api/search/composed", json={"text": "dog", "top_k": 8})
    assert r.status_code == 200
    body = r.json()
    assert body["mode_used"] == "composed"
    assert body["score_basis"] == "composed"
    assert not body["degraded"]
    assert set(_ids(r)) == set(s["dog"])
    scores = [i["score"] for i in body["items"]]
    assert scores == sorted(scores, reverse=True)


def test_positive_examples_alone_are_a_query(ctx):
    client, s = ctx
    r = client.post("/api/search/composed",
                    json={"positive_ids": [s["bike"][0]], "top_k": 8})
    assert r.status_code == 200
    assert _ids(r)[0] == s["bike"][0]        # the example itself is most similar
    assert set(_ids(r)) == set(s["bike"])


def test_text_and_positives_blend_equally(ctx):
    client, s = ctx
    r = client.post("/api/search/composed",
                    json={"text": "soccer", "positive_ids": [s["bike"][0]],
                          "top_k": 16})
    assert set(_ids(r)) == set(s["soccer"]) | set(s["bike"])


def test_negative_examples_push_a_concept_down(ctx):
    client, s = ctx
    base = client.post("/api/search/composed",
                       json={"text": "dog soccer", "top_k": 8})
    assert any(i in set(s["soccer"]) for i in _ids(base))   # blended head
    r = client.post("/api/search/composed",
                    json={"text": "dog soccer",
                          "negative_ids": [s["soccer"][0]], "top_k": 8})
    assert set(_ids(r)) == set(s["dog"])     # soccer repelled below every dog


def test_requires_text_or_positive(ctx):
    client, s = ctx
    assert client.post("/api/search/composed", json={}).status_code == 422
    assert client.post("/api/search/composed", json={"text": "   "}).status_code == 422
    assert client.post("/api/search/composed",
                       json={"negative_ids": [s["dog"][0]]}).status_code == 422


def test_unknown_example_is_a_404_naming_it(ctx):
    client, _ = ctx
    r = client.post("/api/search/composed", json={"positive_ids": [987654]})
    assert r.status_code == 404
    assert "987654" in r.json()["detail"]
    r = client.post("/api/search/composed",
                    json={"text": "dog", "negative_ids": [987654]})
    assert r.status_code == 404


def test_composed_bounds(ctx):
    client, s = ctx
    ok = s["dog"][0]
    assert client.post("/api/search/composed",
                       json={"positive_ids": [ok] * 17}).status_code == 422
    assert client.post("/api/search/composed",
                       json={"text": "x" * 501}).status_code == 422
    assert client.post("/api/search/composed",
                       json={"text": "dog", "top_k": 201}).status_code == 422
    assert client.post("/api/search/composed",
                       json={"text": "dog", "offset": 5001}).status_code == 422
    assert client.post("/api/search/composed",
                       json={"text": "dog", "album": 0}).status_code == 422
    assert client.post("/api/search/composed",
                       json={"positive_ids": [2**63]}).status_code == 422


def test_filters_restrict_candidates(ctx):
    client, s = ctx
    r = client.post("/api/search/composed", json={"text": "dog", "split": "test"})
    assert r.status_code == 200
    assert _ids(r) == [s["dog"][PER_CONCEPT - 1]]    # the one test-split dog


def test_offset_pages_without_overlap(ctx):
    client, _ = ctx
    first = client.post("/api/search/composed",
                        json={"text": "dog", "top_k": 5, "offset": 0})
    second = client.post("/api/search/composed",
                         json={"text": "dog", "top_k": 5, "offset": 5})
    assert first.json()["depth_limit"] == 3 * PER_CONCEPT   # ranked everything
    assert first.json()["has_more"] is True
    assert not set(_ids(first)) & set(_ids(second))


def test_scenarios_are_deterministic_and_bounded(ctx):
    client, _ = ctx
    body = {"text": "dog soccer bike"}
    r1 = client.post("/api/search/scenarios", json=body)
    r2 = client.post("/api/search/scenarios", json=body)
    assert r1.status_code == 200
    assert r1.json() == r2.json()            # fixed seed: byte-identical reruns
    out = r1.json()
    assert out["basis"] == SCENARIO_BASIS
    groups = out["groups"]
    assert 1 <= len(groups) <= 3
    assert sum(g["count"] for g in groups) == 3 * PER_CONCEPT
    for g in groups:
        # Full membership, not a preview: saving a group must file every member.
        assert len(g["sample_ids"]) == g["count"]
        assert g["label"].endswith("images")
        assert "/" in g["evidence"]          # counted evidence, e.g. "8/8 scene:park"
    all_ids = [i for g in groups for i in g["sample_ids"]]
    assert len(all_ids) == len(set(all_ids))  # groups partition the pool
    # Labels are templated from the measured attributes, so the seeded scene
    # values are what can appear in them.
    assert any(any(v in g["label"] for v in SCENE.values()) for g in groups)


def test_scenarios_refuse_to_group_a_handful(ctx):
    client, _ = ctx
    r = client.post("/api/search/scenarios", json={"text": "dog", "split": "test"})
    assert r.status_code == 200
    assert r.json()["groups"] == []
    assert "fewer than" in r.json()["message"]
