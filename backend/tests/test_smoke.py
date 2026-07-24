"""Smoke tests: API works end-to-end on a seeded temp database, without the
embedding stack (exercises the graceful-degradation path).
Data-dir isolation happens in conftest.py, before any `app` import.

    cd backend && pytest
"""
import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app


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


def test_keyword_search_respects_filters_in_sql(client):
    # "a man rides..." lives in the test split; filtering to train must
    # exclude it even though it's the top keyword hit.
    r = client.get("/api/search", params={"q": "rides", "mode": "keyword"})
    assert len(r.json()["items"]) == 1
    r = client.get("/api/search",
                   params={"q": "rides", "mode": "keyword", "split": "train"})
    assert r.json()["items"] == []


def test_porter_stemming(client):
    # Caption says "runs"; the stemmed FTS index should match query "running".
    r = client.get("/api/search", params={"q": "running", "mode": "keyword"})
    items = r.json()["items"]
    assert len(items) == 1 and "runs" in items[0]["match_caption"]


def test_match_explanation_fields(client):
    r = client.get("/api/search", params={"q": "soccer", "mode": "keyword"})
    item = r.json()["items"][0]
    assert "soccer" in item["match_caption"]
    assert "soccer" in item["matched_terms"]


def test_bulk_tag(client):
    ids = [it["id"] for it in client.get("/api/samples").json()["items"][:2]]
    r = client.post("/api/tags/bulk", json={"sample_ids": ids, "name": "Batch-Tag"})
    assert r.status_code == 200 and r.json()["tag"] == "batch-tag"
    assert client.get("/api/samples", params={"tag": "batch-tag"}).json()["total"] == 2


def test_qa_and_eval_degrade_gracefully(client):
    r = client.get("/api/qa/summary")
    assert r.status_code == 200 and r.json()["available"] is False
    r = client.get("/api/eval/retrieval")
    assert r.status_code == 200 and r.json()["available"] is False
    assert client.get("/api/qa/captions").json() == []
    assert client.get("/api/attributes/coverage").json() == []


def test_admin_reload(client):
    r = client.post("/api/admin/reload")
    assert r.status_code == 200
    assert r.json()["image_index"] is False


def test_chat_unavailable_is_graceful(client):
    r = client.get("/api/chat/status")
    assert r.status_code == 200 and r.json()["available"] is False
    r = client.post("/api/chat",
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503  # clear setup instructions, not a crash
    assert "Ollama" in r.json()["detail"] or "agent stack" in r.json()["detail"]


def test_export_manifest(client):
    r = client.get("/api/export", params={"split": "test"})
    body = r.json()
    assert body["count"] == 1
    assert body["samples"][0]["captions"]
