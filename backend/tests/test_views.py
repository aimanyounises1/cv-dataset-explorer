"""Saved views: create / list / restore round-trip / duplicate / delete.

Self-contained — mounts only the views router, so nothing here depends on the
seeded sample corpus the other test modules build.
Data-dir isolation happens in conftest.py, before any `app` import.

    cd backend && pytest tests/test_views.py
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import db
from app.api import views


@pytest.fixture(scope="module")
def client():
    conn = db.connect()
    db.init_db(conn)
    # The temp data dir is shared across the whole test session; start from a
    # known-empty table so these assertions do not depend on module order.
    conn.execute("DELETE FROM saved_views")
    conn.commit()
    conn.close()
    api = FastAPI()
    api.include_router(views.router, prefix="/api")
    with TestClient(api) as c:
        yield c


def test_empty_list(client):
    r = client.get("/api/views")
    assert r.status_code == 200
    assert r.json() == []


def test_create_and_list(client):
    r = client.post("/api/views", json={"name": "hard train",
                                        "query_string": "split=train&difficulty_min=8"})
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "hard train"
    assert body["created_at"]

    r = client.get("/api/views")
    assert r.status_code == 200
    names = [v["name"] for v in r.json()]
    assert "hard train" in names


def test_restore_round_trip(client):
    # Percent-encoding and '&' must survive verbatim: the client hands this
    # straight back to the router, so any normalisation would change the view.
    qs = "q=a%20black%20dog&mode=hybrid&attr=weather%3Arain&clutter_max=3"
    assert client.post("/api/views", json={"name": "rainy dogs",
                                           "query_string": qs}).status_code == 201
    saved = [v for v in client.get("/api/views").json() if v["name"] == "rainy dogs"]
    assert len(saved) == 1
    assert saved[0]["query_string"] == qs


def test_newest_first(client):
    client.post("/api/views", json={"name": "zzz older", "query_string": "split=test"})
    client.post("/api/views", json={"name": "aaa newer", "query_string": "split=val"})
    names = [v["name"] for v in client.get("/api/views").json()]
    assert names.index("aaa newer") < names.index("zzz older")


def test_name_is_trimmed(client):
    r = client.post("/api/views", json={"name": "  padded  ", "query_string": "split=train"})
    assert r.status_code == 201
    assert r.json()["name"] == "padded"
    assert "padded" in [v["name"] for v in client.get("/api/views").json()]


def test_duplicate_name_conflicts(client):
    client.post("/api/views", json={"name": "dupe", "query_string": "split=train"})
    r = client.post("/api/views", json={"name": "dupe", "query_string": "split=test"})
    assert r.status_code == 409
    # The original must be untouched, not overwritten by the rejected write.
    kept = [v for v in client.get("/api/views").json() if v["name"] == "dupe"]
    assert len(kept) == 1
    assert kept[0]["query_string"] == "split=train"


def test_empty_name_rejected(client):
    assert client.post("/api/views", json={"name": "", "query_string": "x=1"}).status_code == 400
    assert client.post("/api/views",
                       json={"name": "   ", "query_string": "x=1"}).status_code == 400
    assert "" not in [v["name"] for v in client.get("/api/views").json()]


def test_delete(client):
    client.post("/api/views", json={"name": "throwaway", "query_string": "split=train"})
    assert client.delete("/api/views/throwaway").status_code == 200
    assert "throwaway" not in [v["name"] for v in client.get("/api/views").json()]
    # Deleting again is a 404, not a silent success: the caller asked for a
    # named resource that is not there.
    assert client.delete("/api/views/throwaway").status_code == 404


def test_query_string_may_be_empty(client):
    # "no filters at all" is a legitimate view to save; only the name is required.
    r = client.post("/api/views", json={"name": "everything", "query_string": ""})
    assert r.status_code == 201
    assert r.json()["query_string"] == ""
