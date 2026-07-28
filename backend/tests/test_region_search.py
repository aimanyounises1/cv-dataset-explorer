"""Region evidence: a marked area of an existing sample as a positive or
negative query. The server crops the original file from normalized geometry —
reproducible from the request alone — and the negative role ranks by distance
and never returns its own source image.

    cd backend && pytest tests/test_region_search.py
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, db
from app.main import app
from app.ml import index as index_mod
from app.ml import providers
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
    original = search_api.get_embedder
    search_api.get_embedder = lambda: enc          # explicit injection
    with TestClient(app) as client:
        yield client, sids
    search_api.get_embedder = original
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
