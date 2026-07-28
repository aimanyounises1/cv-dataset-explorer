"""Composed search without embeddings: honest degradation, never a 500.

Runs with no index files planted (module name sorts before
test_composed_search, and every embedded module restores the world), so this
is the state a fresh clone is in before `python -m app.ingest`.

    cd backend && pytest tests/test_composed_degraded.py
"""
import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app
from app.ml.index import invalidate_index


@pytest.fixture(scope="module")
def client():
    conn = db.connect()
    db.init_db(conn)
    conn.close()
    invalidate_index()      # nothing cached from an earlier module
    with TestClient(app) as c:
        yield c
    invalidate_index()


def test_text_falls_back_to_keyword_and_says_so(client):
    r = client.post("/api/search/composed", json={"text": "anything at all"})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["mode_used"] == "keyword"    # honest: that is what ranked
    assert "app.ingest" in body["message"]
    assert "ignored" in body["message"]      # example images did not rank


def test_positives_alone_degrade_to_an_empty_answer(client):
    # Membership in the index cannot even be checked without the index, so
    # degradation answers before any 404 could.
    r = client.post("/api/search/composed", json={"positive_ids": [12345]})
    assert r.status_code == 200
    body = r.json()
    assert body["items"] == []
    assert body["degraded"] is True
    assert body["mode_used"] == "composed"
    assert "app.ingest" in body["message"]


def test_scenarios_degrade_empty(client):
    r = client.post("/api/search/scenarios", json={"text": "anything"})
    assert r.status_code == 200
    body = r.json()
    assert body["groups"] == []
    assert body["degraded"] is True
    assert "app.ingest" in body["message"]
    assert body["basis"]                     # the method is stated even when idle
