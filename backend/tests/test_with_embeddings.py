"""Contract tests for the full semantic path — search, similarity, caption QA,
and the retrieval benchmark — using a FAKE embedder and synthetic embeddings.
No model download, deterministic vectors, runs in CI in milliseconds.

The fixture plants real index files (the same on-disk format ingestion
produces), swaps the embedder behind the API's seam, and fully restores the
world on teardown so the degradation tests in test_smoke.py stay valid
regardless of execution order.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, db
from app.analyze import compute_caption_scores
from app.api import search as search_module
from app.api.search import normalize_query_text
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
    """Maps text to the synthetic concept space by keyword lookup.

    `EXACT` pins each caption's text to its own sample's image vector. The
    benchmark encodes its query text through this seam instead of reading a
    stored caption vector, so the fixture's premise — the caption embedding *is*
    the image embedding, which is what makes semantic R@1 perfect here — has to
    survive encoding. Keyed on the normalized form because that is what reaches
    the encoder. Any other query ("dog") falls through to concept lookup.
    """

    EXACT: dict[str, np.ndarray] = {}

    def encode_images(self, images) -> np.ndarray:
        # Every uploaded image "looks like" the dog concept — enough to assert
        # that by-image search ranks through the same index as everything else.
        return np.stack([_vec({0: 1.0}) for _ in images])

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            hit = self.EXACT.get(normalize_query_text(t))
            if hit is not None:
                out.append(hit)
                continue
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
        FakeEmbedder.EXACT[normalize_query_text(caption)] = vec
    conn.commit()

    EmbeddingIndex.save(np.array(sample_ids), np.stack(img_vecs), kind="image")
    EmbeddingIndex.save(np.array(caption_ids), np.stack(cap_vecs), kind="caption")
    invalidate_index()
    compute_caption_scores(conn)

    fake_embedder = FakeEmbedder()
    real_get_embedder = search_module.get_embedder
    real_get_runtime = search_module.get_retrieval_bundle
    real_get_indexes = search_module.get_retrieval_index_bundle
    real_bind_runtime = search_module.bind_retrieval_bundle
    indexes = SimpleNamespace(
        provider="test",
        model_id="test/fake-embedder",
        emb_dir=config.EMB_DIR,
        generation="with-embeddings-fixture",
        image_index=EmbeddingIndex.load("image"),
        caption_index=EmbeddingIndex.load("caption"),
        manifest={"dim": 8},
    )
    runtime = SimpleNamespace(**vars(indexes), encoder=fake_embedder)
    search_module.get_embedder = lambda: fake_embedder
    search_module.get_retrieval_bundle = lambda: runtime
    search_module.get_retrieval_index_bundle = lambda: indexes
    search_module.bind_retrieval_bundle = lambda bound: runtime
    try:
        with TestClient(app) as c:
            yield c
    finally:
        # Restore the world so order-independent degradation tests stay valid.
        search_module.get_embedder = real_get_embedder
        search_module.get_retrieval_bundle = real_get_runtime
        search_module.get_retrieval_index_bundle = real_get_indexes
        search_module.bind_retrieval_bundle = real_bind_runtime
        FakeEmbedder.EXACT.clear()
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


def test_search_pages_partition_the_ranking(client):
    """Paging must split one ranking, not re-rank per page.

    RRF scores depend on how deep the candidate lists go, so fusing to the
    depth of the current page would shuffle items between pages — the user
    sees duplicates and gaps. Fusing to a fixed depth makes the pages a true
    partition, which is what this asserts.
    """
    def ids(**kw):
        return [it["id"] for it in client.get("/api/search", params={
            "q": "dog", "mode": "hybrid", **kw}).json()["items"]]

    whole = ids(top_k=4)
    first, second = ids(top_k=2), ids(top_k=2, offset=2)
    assert first + second == whole
    assert not set(first) & set(second)


def test_search_reports_whether_more_results_exist(client):
    # Sized from the ranking itself: this module shares its temp database with
    # the smoke fixture, so the lexical side sees more captions than the four
    # embedded samples and a hard-coded total would be wrong.
    def page(**kw):
        return client.get("/api/search",
                          params={"q": "dog", "mode": "hybrid", **kw}).json()

    total = len(page(top_k=50)["items"])
    assert total >= 2

    first = page(top_k=1)
    assert first["offset"] == 0 and first["has_more"] is True

    final = page(top_k=1, offset=total - 1)
    assert final["offset"] == total - 1 and final["has_more"] is False


def test_export_of_a_query_matches_the_search_it_came_from(client):
    """An exported slice is only useful if it is the slice you were looking at."""
    params = {"q": "dog", "mode": "hybrid", "top_k": 3}
    searched = [it["id"] for it in client.get("/api/search", params=params).json()["items"]]
    exported = client.get("/api/export", params=params).json()
    assert [s["id"] for s in exported["samples"]] == searched      # same order
    assert exported["filters"]["q"] == "dog"
    assert exported["filters"]["embed_model"]                       # reproducible


def test_lexical_candidates_always_excludes_the_query_caption(client):
    """FR-EV-2 at the seam rather than through a metric.

    The lexical and fused paths share one helper precisely so this invariant
    has a single definition: an earlier version duplicated the SQL, and a
    mutation removing the exclusion from the fused copy left the whole suite
    green because the fixture's semantic path already ranked the target first.
    """
    from app import db
    from app.api.eval import lexical_candidates

    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT id, sample_id, text FROM captions ORDER BY id LIMIT 1").fetchone()
        with_own = conn.execute(
            "SELECT DISTINCT c.sample_id FROM captions_fts f JOIN captions c "
            "ON c.id = f.rowid WHERE captions_fts MATCH ?",
            (db.fts_escape(row["text"]),)).fetchall()
        assert row["sample_id"] in [r["sample_id"] for r in with_own]  # it is findable
        got = lexical_candidates(conn, row["text"], row["id"])
        assert row["sample_id"] not in [r["sid"] for r in got]
    finally:
        conn.close()


def test_benchmark_reports_candidate_pool_per_mode(client):
    """A recall number is only about ranking if there was something to rank."""
    body = client.get("/api/eval/retrieval", params={"sample_size": 50}).json()
    modes = {r["mode"]: r for r in body["results"]}
    # Semantic always ranks the whole corpus; the lexical path ranks only what
    # its conjunctive query matched, which for whole-caption queries is little.
    assert modes["semantic"]["mean_candidates"] == 4.0
    assert modes["semantic"]["empty_query_rate"] == 0.0
    assert modes["keyword"]["empty_query_rate"] == 1.0  # one caption each, excluded
    assert modes["keyword"]["mean_candidates"] == 0.0
    assert body["mean_query_words"] > 0


class _Recorder:
    """Wraps the fake embedder and records exactly what text reached it."""

    def __init__(self):
        self.seen: list[str] = []
        self._inner = FakeEmbedder()

    def encode_texts(self, texts):
        self.seen.extend(texts)
        return self._inner.encode_texts(texts)


def _with_recorder(client, call):
    rec = _Recorder()
    previous = search_module.get_embedder
    previous_runtime = search_module.get_retrieval_bundle
    previous_bind = search_module.bind_retrieval_bundle
    runtime = previous_runtime()
    search_module.get_embedder = lambda: rec
    recorded_runtime = SimpleNamespace(**vars(runtime))
    recorded_runtime.encoder = rec
    search_module.get_retrieval_bundle = lambda: recorded_runtime
    search_module.bind_retrieval_bundle = lambda indexes: recorded_runtime
    try:
        call()
    finally:
        search_module.get_embedder = previous
        search_module.get_retrieval_bundle = previous_runtime
        search_module.bind_retrieval_bundle = previous_bind
    return rec


def test_search_encodes_the_normalized_query(client):
    """The text handed to SigLIP is the normalized form, not what was typed."""
    rec = _with_recorder(client, lambda: client.get(
        "/api/search", params={"q": "A Happy Dog, Running!", "mode": "semantic"}))
    assert rec.seen == ["a happy dog running"]


def test_benchmark_encodes_queries_through_the_search_seam(client):
    """The benchmark's claim is that it measures what /api/search does.

    That only holds if both reach the model the same way. An earlier version
    read the stored caption vectors instead — encodings of the raw caption text,
    which no search request ever produces — so the benchmark could not have
    noticed a change to how queries are encoded. Substituting the embedder for
    search must therefore change the benchmark too.
    """
    from app import config

    for f in config.CACHE_DIR.glob("eval_*.json"):
        f.unlink(missing_ok=True)
    rec = _with_recorder(client, lambda: client.get(
        "/api/eval/retrieval", params={"sample_size": 50}))
    assert rec.seen, "the benchmark did not encode anything through the seam"
    # Every query it encoded was normalized on the way through.
    assert all(t == normalize_query_text(t) for t in rec.seen)
    for f in config.CACHE_DIR.glob("eval_*.json"):
        f.unlink(missing_ok=True)


def test_benchmark_says_so_when_it_could_not_encode_the_queries(client):
    """Degrading to the stored caption vectors is allowed; doing it silently is
    not, because those vectors are not what the shipped path produces."""
    from app import config

    for f in config.CACHE_DIR.glob("eval_*.json"):
        f.unlink(missing_ok=True)
    previous = search_module.bind_retrieval_bundle

    def fail_encoder(indexes):
        raise RuntimeError("synthetic encoder construction failure")

    search_module.bind_retrieval_bundle = fail_encoder
    try:
        body = client.get("/api/eval/retrieval", params={"sample_size": 50}).json()
    finally:
        search_module.bind_retrieval_bundle = previous
        for f in config.CACHE_DIR.glob("eval_*.json"):
            f.unlink(missing_ok=True)
    assert body["available"] is True
    assert "unavailable" in (body["message"] or "")


def test_benchmark_reports_pool_size_and_rank_metrics(client):
    """FR-EV-4: a recall number without its candidate pool is not comparable."""
    body = client.get("/api/eval/retrieval", params={"sample_size": 50}).json()
    assert body["pool_size"] == 4        # the fixture corpus, not the query count
    assert body["depth"] == 10
    semantic = next(r for r in body["results"] if r["mode"] == "semantic")
    assert semantic["mrr"] == 1.0
    assert semantic["median_rank"] == 1.0


def test_search_by_image_ranks_through_the_same_index(client):
    """An uploaded image ranks the corpus image-to-image: plain cosines,
    descending, through the exact index text search uses."""
    import io as _io

    from PIL import Image as _Image

    buf = _io.BytesIO()
    _Image.new("RGB", (4, 3), (120, 90, 40)).save(buf, format="PNG")
    r = client.post("/api/search/by-image?top_k=4", content=buf.getvalue())
    assert r.status_code == 200
    cards = r.json()
    assert len(cards) == 4
    # The fake embedder maps every image onto the dog concept, so the two dog
    # samples must lead and every card must carry its cosine, in ranked order.
    assert all("dog" in c["caption"] for c in cards[:2])
    scores = [c["score"] for c in cards]
    assert all(s is not None for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_search_by_image_rejects_non_images(client):
    r = client.post("/api/search/by-image", content=b"definitely not a decodable image")
    assert r.status_code == 400
    assert client.post("/api/search/by-image", content=b"").status_code == 400
