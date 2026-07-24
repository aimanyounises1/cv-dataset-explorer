"""Contract tests for the full semantic path — search, similarity, caption QA,
and the retrieval benchmark — using a FAKE embedder and synthetic embeddings.
No model download, deterministic vectors, runs in CI in milliseconds.

The fixture plants real index files (the same on-disk format ingestion
produces), swaps the embedder behind the API's seam, and fully restores the
world on teardown so the degradation tests in test_smoke.py stay valid
regardless of execution order.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db
from app.analyze import compute_caption_scores
from app.api import search as search_module
from app.main import app
from app.ml.index import EmbeddingIndex, invalidate_index

# (env isolation happens in conftest.py, auto-loaded by pytest before imports)

# Three orthogonal concepts: index 0=dog, 1=soccer, 2=bike.
CONCEPTS = ["dog", "soccer", "bike"]


def _vec(concept_weights: dict[int, float]) -> np.ndarray:
    v = np.zeros(8, dtype=np.float32)
    for idx, w in concept_weights.items():
        v[idx] = w
    return v / (np.linalg.norm(v) or 1.0)


class FakeEmbedder:
    """Maps text to the synthetic concept space by keyword lookup."""

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            weights = {i: 1.0 for i, c in enumerate(CONCEPTS) if c in t.lower()}
            out.append(_vec(weights or {0: 0.5, 1: 0.5, 2: 0.5}))
        return np.stack(out)


@pytest.fixture(scope="module")
def client():
    conn = db.connect()
    db.init_db(conn)

    samples = [
        ("emb_dog.jpg", "train", "a happy dog chases a ball", _vec({0: 1.0})),
        ("emb_dog2.jpg", "train", "a wet dog shakes off water", _vec({0: 0.95, 1: 0.05})),
        ("emb_soccer.jpg", "test", "kids play soccer on a field", _vec({1: 1.0})),
        ("emb_bike.jpg", "train", "a rider on a mountain bike", _vec({2: 1.0})),
    ]
    sample_ids, caption_ids, img_vecs, cap_vecs = [], [], [], []
    for filename, split, caption, vec in samples:
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
            "VALUES ('flickr8k', ?, ?, 300, 200, 1)", (filename, split))
        sid = cur.lastrowid
        ccur = conn.execute(
            "INSERT INTO captions(sample_id, idx, text) VALUES (?, 0, ?)", (sid, caption))
        conn.execute("INSERT INTO captions_fts(rowid, text) VALUES (?, ?)",
                     (ccur.lastrowid, caption))
        sample_ids.append(sid)
        caption_ids.append(ccur.lastrowid)
        img_vecs.append(vec)
        cap_vecs.append(vec)  # caption embedding ≈ image embedding
    conn.commit()

    EmbeddingIndex.save(np.array(sample_ids), np.stack(img_vecs), kind="image")
    EmbeddingIndex.save(np.array(caption_ids), np.stack(cap_vecs), kind="caption")
    invalidate_index()
    compute_caption_scores(conn)

    real_get_embedder = search_module.get_embedder
    search_module.get_embedder = lambda: FakeEmbedder()  # inject behind the seam
    try:
        with TestClient(app) as c:
            yield c
    finally:
        # Restore the world so order-independent degradation tests stay valid.
        search_module.get_embedder = real_get_embedder
        from app import config
        for f in config.EMB_DIR.glob("*.npy"):
            f.unlink(missing_ok=True)
        for f in config.CACHE_DIR.glob("*.json"):
            f.unlink(missing_ok=True)
        invalidate_index()
        conn.execute("UPDATE captions SET agreement = NULL")
        conn.execute("UPDATE samples SET caption_consistency = NULL")
        qmarks = ",".join("?" * len(sample_ids))
        conn.execute(f"DELETE FROM samples WHERE id IN ({qmarks})", sample_ids)
        conn.commit()
        conn.close()


def test_semantic_search_ranks_by_similarity(client):
    r = client.get("/api/search", params={"q": "dog", "mode": "semantic"})
    body = r.json()
    assert body["degraded"] is False and body["mode_used"] == "semantic"
    captions = [it["caption"] for it in body["items"][:2]]
    assert all("dog" in c for c in captions)
    assert body["items"][0]["score"] >= body["items"][1]["score"]
    assert "dog" in body["items"][0]["match_caption"]  # explains the match


def test_semantic_search_respects_filters(client):
    r = client.get("/api/search",
                   params={"q": "soccer", "mode": "semantic", "split": "train"})
    # The soccer sample is in 'test'; filtered semantic search must not return it.
    assert all(it["split"] == "train" for it in r.json()["items"])


def test_hybrid_fuses_both_rankings(client):
    r = client.get("/api/search", params={"q": "dog", "mode": "hybrid"})
    body = r.json()
    assert body["mode_used"] == "hybrid" and body["degraded"] is False
    assert "dog" in body["items"][0]["caption"]


def test_similar_images(client):
    dog_id = client.get("/api/search", params={"q": "dog", "mode": "semantic"}
                        ).json()["items"][0]["id"]
    r = client.get(f"/api/samples/{dog_id}/similar")
    assert r.status_code == 200
    top = r.json()[0]
    assert "dog" in top["caption"] and top["id"] != dog_id


def test_caption_qa_available_and_ordered(client):
    assert client.get("/api/qa/summary").json()["available"] is True
    suspects = client.get("/api/qa/captions").json()
    scores = [s["agreement"] for s in suspects]
    assert scores == sorted(scores)  # most suspect first


def test_retrieval_benchmark(client):
    r = client.get("/api/eval/retrieval", params={"sample_size": 50})
    body = r.json()
    assert body["available"] is True
    modes = {res["mode"]: res["recall_at"] for res in body["results"]}
    assert set(modes) == {"semantic", "keyword", "hybrid"}
    # Caption embeddings equal their image embeddings ⇒ semantic R@1 is perfect.
    assert modes["semantic"]["1"] == 1.0


def test_benchmark_excludes_the_held_out_caption(client):
    """FR-EV-2: the query caption must not be in the index it searches.

    Each fixture sample has exactly one caption, so once that caption is
    excluded the lexical path has no text left pointing at the right image and
    recall must be 0. Without the exclusion every query would trivially match
    its own row and keyword recall would read 1.0 — a number measuring nothing.
    """
    body = client.get("/api/eval/retrieval", params={"sample_size": 50}).json()
    modes = {res["mode"]: res for res in body["results"]}
    assert modes["keyword"]["recall_at"]["10"] == 0.0
    assert modes["keyword"]["median_rank"] is None  # never found within depth


def test_benchmark_reports_pool_size_and_rank_metrics(client):
    """FR-EV-4: a recall number without its candidate pool is not comparable."""
    body = client.get("/api/eval/retrieval", params={"sample_size": 50}).json()
    assert body["pool_size"] == 4        # the fixture corpus, not the query count
    assert body["depth"] == 10
    semantic = next(r for r in body["results"] if r["mode"] == "semantic")
    assert semantic["mrr"] == 1.0
    assert semantic["median_rank"] == 1.0
