"""Album intelligence: measured signals counted from stored rows (majority
attributes, tag shares, splits), provider-index coherence/outliers, honest
notes when inputs are missing, and the generated-summary boundary (explicit
request, named model, 503 with instructions when Ollama is absent, nothing
persisted).

    cd backend && pytest tests/test_album_analysis.py
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, db
from app.api.albums import _summary_prompt
from app.main import app
from app.ml import index as index_mod
from app.ml import providers


@pytest.fixture(scope="module")
def ctx():
    """(client, album_id, sample_ids): six members with a clear majority
    setting, an even day/night split, a half-share tag, and one embedding
    outlier."""
    conn = db.connect()
    db.init_db(conn)
    sids = []
    for i in range(6):
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
            "VALUES ('flickr8k', ?, 'train', 300, 200, 1)", (f"ana_{i}.jpg",))
        sids.append(cur.lastrowid)
        conn.execute("INSERT INTO captions(sample_id, idx, text) VALUES (?, 0, ?)",
                     (sids[-1], f"analysis seed caption {i}"))
    # setting: 5 of 6 outdoor (0.83 -> common); time_of_day: 3/3 (-> different)
    for i, sid in enumerate(sids):
        if i < 5:
            conn.execute("INSERT INTO attributes(sample_id, grp, label, confidence) "
                         "VALUES (?, 'setting', 'outdoor', 0.9)", (sid,))
        conn.execute("INSERT INTO attributes(sample_id, grp, label, confidence) "
                     "VALUES (?, 'time_of_day', ?, 0.9)",
                     (sid, "day" if i < 3 else "night"))
    cur = conn.execute("INSERT INTO tags(name) VALUES ('edge-case')")
    for sid in sids[:3]:  # 3 of 6 = 0.5 -> common
        conn.execute("INSERT INTO sample_tags(sample_id, tag_id) VALUES (?, ?)",
                     (sid, cur.lastrowid))
    conn.commit()

    # Flat SigLIP-layout index: five vectors on one axis, the last on another.
    config.EMB_DIR.mkdir(parents=True, exist_ok=True)
    vecs = np.zeros((6, 4), dtype=np.float32)
    vecs[:5, 0] = 1.0
    vecs[5, 1] = 1.0
    np.save(config.EMB_DIR / "sample_ids.npy", np.array(sids, dtype=np.int64))
    np.save(config.EMB_DIR / "image_embeddings.npy", vecs)
    providers.invalidate_providers()
    index_mod.invalidate_index()

    with TestClient(app) as client:
        album = client.post("/api/albums", json={"name": "analysis-fixture"}).json()
        client.post(f"/api/albums/{album['id']}/items", json={"sample_ids": sids})
        yield client, album["id"], sids

    conn.execute("DELETE FROM albums")
    conn.execute("DELETE FROM album_items")
    for sid in sids:
        conn.execute("DELETE FROM samples WHERE id = ?", (sid,))
        conn.execute("DELETE FROM captions WHERE sample_id = ?", (sid,))
        conn.execute("DELETE FROM attributes WHERE sample_id = ?", (sid,))
        conn.execute("DELETE FROM sample_tags WHERE sample_id = ?", (sid,))
    conn.commit()
    conn.close()
    for f in ("sample_ids.npy", "image_embeddings.npy"):
        (config.EMB_DIR / f).unlink(missing_ok=True)
    providers.invalidate_providers()
    index_mod.invalidate_index()


def test_measured_majority_split_and_tag_shares(ctx):
    client, album_id, sids = ctx
    m = client.get(f"/api/albums/{album_id}/analysis").json()["measured"]
    kinds = {(c["kind"], c["label"]): c["share"] for c in m["common"]}
    assert kinds[("attribute", "outdoor")] == 0.83
    assert kinds[("tag", "edge-case")] == 0.5
    split = {d["grp"]: d for d in m["different"]}
    assert "time_of_day" in split
    assert {t["label"] for t in split["time_of_day"]["top"]} == {"day", "night"}


def test_coherence_and_the_crafted_outlier(ctx):
    client, album_id, sids = ctx
    m = client.get(f"/api/albums/{album_id}/analysis").json()["measured"]
    assert isinstance(m["coherence"], float)
    assert m["outliers"], "six members with embeddings must yield outliers"
    assert m["outliers"][0]["id"] == sids[5]  # the off-axis vector ranks worst
    assert m["outliers"][0]["score"] < 0.9


def test_generated_block_names_model_and_never_contains_a_summary(ctx):
    client, album_id, _ = ctx
    g = client.get(f"/api/albums/{album_id}/analysis").json()["generated"]
    assert g["model"] == config.CHAT_MODEL
    assert g["summary"] is None  # GET never generates
    assert g["available"] in (True, False)
    if not g["available"]:
        assert "ollama" in g["message"].lower()


def test_unknown_album_404s(ctx):
    client, *_ = ctx
    assert client.get("/api/albums/999999/analysis").status_code == 404


def test_summary_without_ollama_is_a_503_with_instructions(ctx, monkeypatch):
    client, album_id, _ = ctx
    import httpx

    def refuse(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr("httpx.post", refuse)
    r = client.post(f"/api/albums/{album_id}/analysis/summary")
    assert r.status_code == 503
    assert "ollama serve" in r.json()["detail"]


def test_summary_on_empty_album_is_a_400(ctx):
    client, *_ = ctx
    empty = client.post("/api/albums", json={"name": "empty-ana"}).json()
    r = client.post(f"/api/albums/{empty['id']}/analysis/summary")
    assert r.status_code == 400
    client.delete(f"/api/albums/{empty['id']}")


def test_summary_prompt_is_grounded_and_plain():
    p = _summary_prompt(
        "Night motion",
        [{"kind": "attribute", "grp": "setting", "label": "outdoor", "share": 0.8}],
        [{"grp": "time_of_day",
          "top": [{"label": "day", "share": 0.5}, {"label": "night", "share": 0.5}]}],
        ["a dog runs across a dark field"])
    assert "Night motion" in p and "outdoor (80%)" in p
    assert "day vs night" in p
    assert "a dog runs across a dark field" in p
    assert "markdown" in p  # the no-markdown instruction survives
