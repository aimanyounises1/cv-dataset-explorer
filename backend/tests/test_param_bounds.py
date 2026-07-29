"""Hostile-input bounds, pinned after a live probe found the holes.

Every limit here was measured as a real failure against the running server:
`max_agreement=Infinity` 500ed the export and `NaN` wrote an invalid-JSON
manifest; `page=2^63` reached SQL OFFSET as a bare 500; `limit=2^63` did the
same on the VLM facet list; a million-id bulk-tag body was accepted whole; a
200,000-character view name was stored; and a view named with "/" could be
created but never deleted, because starlette decodes %2F before routing.

    cd backend && pytest tests/test_param_bounds.py
"""
import urllib.parse

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app


@pytest.fixture(scope="module")
def client():
    conn = db.connect()
    db.init_db(conn)
    conn.commit()
    conn.close()
    with TestClient(app) as c:
        yield c


def test_max_agreement_rejects_what_used_to_500(client):
    for bad in ("Infinity", "-Infinity", "NaN", "2", "-0.1"):
        assert client.get("/api/samples",
                          params={"max_agreement": bad}).status_code == 422, bad
        assert client.get("/api/export",
                          params={"max_agreement": bad}).status_code == 422, bad
    assert client.get("/api/samples", params={"max_agreement": 0.1}).status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {"q": ""},
        {"q": "dog", "mode": "boosted"},
        {"q": "dog", "top_k": 0},
        {"q": "dog", "top_k": 201},
        {"q": "dog", "offset": -1},
        {"q": "dog", "offset": 5001},
        {"q": "dog", "sort": "difficulty_sideways"},
        {"q": "dog", "max_agreement": "NaN"},
        {"q": "dog", "max_agreement": 1.1},
        {"q": "dog", "axes": {"unknown": {"min": 1}}},
        {"q": "dog", "axes": {"difficulty": {"min": -1}}},
        {"q": "dog", "axes": {"difficulty": {"max": 11}}},
        {"q": "dog", "axes": {"difficulty": "high"}},
        {"q": "dog", "axes": {"difficulty": {"minimum": 8}}},
        {"q": True},
        {"q": "dog", "top_k": True},
        {"q": "dog", "offset": False},
        {"q": "dog", "max_agreement": True},
        {"q": "dog", "cluster": True},
        {"q": "dog", "album": True},
        {"q": "dog", "axes": {"difficulty": {"min": True}}},
    ],
)
def test_post_search_rejects_invalid_get_equivalents(client, body):
    assert client.post("/api/search", json=body).status_code == 422


def test_post_search_accepts_valid_axis_bounds(client):
    response = client.post(
        "/api/search",
        json={
            "q": "dog",
            "mode": "keyword",
            "axes": {"difficulty": {"min": 3, "max": 8}},
        },
    )
    assert response.status_code == 200, response.text


def test_page_has_a_ceiling(client):
    assert client.get("/api/samples", params={"page": 2**63 - 1}).status_code == 422
    assert client.get("/api/samples", params={"page": 1}).status_code == 200


def test_vlm_tag_limit_is_validated(client):
    assert client.get("/api/vlm-tags", params={"limit": 2**63}).status_code == 422
    assert client.get("/api/vlm-tags", params={"limit": -5}).status_code == 422
    assert client.get("/api/vlm-tags", params={"limit": 10}).status_code == 200


def test_bulk_tag_body_is_capped(client):
    r = client.post("/api/tags/bulk",
                    json={"sample_ids": list(range(100_001)), "name": "flood"})
    assert r.status_code == 400
    assert "100,000" in r.json()["detail"]


def test_view_name_and_query_string_are_bounded(client):
    assert client.post("/api/views",
                       json={"name": "x" * 201, "query_string": ""}).status_code == 422
    assert client.post("/api/views",
                       json={"name": "ok", "query_string": "y" * 100_001}).status_code == 422
    client.delete("/api/views/ok")


def test_view_named_with_slash_is_deletable(client):
    # The exact shape the UI sends: encodeURIComponent, so "/" arrives as %2F.
    name = "night / indoor"
    r = client.post("/api/views",
                    json={"name": name, "query_string": "attr=time_of_day:night"})
    assert r.status_code == 201
    assert client.delete(
        f"/api/views/{urllib.parse.quote(name, safe='')}").status_code == 200
    assert name not in [v["name"] for v in client.get("/api/views").json()]
