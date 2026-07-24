"""Smoke tests: API works end-to-end on a seeded temp database, without the
embedding stack (exercises the graceful-degradation path).

    cd backend && pytest
"""
import os
import tempfile

import pytest

_tmpdir = tempfile.mkdtemp()
os.environ["CVDE_DATA_DIR"] = _tmpdir  # must be set before importing the app

from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    conn = db.connect()
    db.init_db(conn)
    for i, (split, caption) in enumerate([
        ("train", "a black dog runs across the wet grass"),
        ("train", "two children play soccer in a park"),
        ("test", "a man rides a red mountain bike on a trail"),
    ]):
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
            "VALUES ('flickr8k', ?, ?, 400, 300, 1000)", (f"img_{i}.jpg", split))
        sid = cur.lastrowid
        ccur = conn.execute(
            "INSERT INTO captions(sample_id, idx, text) VALUES (?, 0, ?)", (sid, caption))
        conn.execute("INSERT INTO captions_fts(rowid, text) VALUES (?, ?)",
                     (ccur.lastrowid, caption))
    conn.commit()
    conn.close()
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["samples"] == 3


def test_list_and_filter(client):
    r = client.get("/api/samples")
    assert r.status_code == 200
    assert r.json()["total"] == 3
    r = client.get("/api/samples", params={"split": "test"})
    assert r.json()["total"] == 1


def test_detail(client):
    sid = client.get("/api/samples").json()["items"][0]["id"]
    r = client.get(f"/api/samples/{sid}")
    assert r.status_code == 200
    assert len(r.json()["captions"]) == 1


def test_keyword_search(client):
    r = client.get("/api/search", params={"q": "dog grass", "mode": "keyword"})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert "dog" in items[0]["caption"]


def test_hybrid_degrades_without_embeddings(client):
    r = client.get("/api/search", params={"q": "dog", "mode": "hybrid"})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["mode_used"] == "keyword"
    assert len(body["items"]) == 1


def test_tags_roundtrip(client):
    sid = client.get("/api/samples").json()["items"][0]["id"]
    assert client.post(f"/api/samples/{sid}/tags", json={"name": "Edge-Case"}).status_code == 200
    detail = client.get(f"/api/samples/{sid}").json()
    assert "edge-case" in detail["tags"]
    r = client.get("/api/samples", params={"tag": "edge-case"})
    assert r.json()["total"] == 1
    assert client.delete(f"/api/samples/{sid}/tags/edge-case").status_code == 200


def test_stats(client):
    r = client.get("/api/stats/overview")
    assert r.status_code == 200
    assert r.json()["embeddings_available"] is False
    r = client.get("/api/stats/captions")
    assert r.status_code == 200
    assert len(r.json()["top_words"]) > 0
