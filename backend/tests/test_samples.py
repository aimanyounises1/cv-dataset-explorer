"""Hostile ids on the sample routes, pinned after a live probe.

`GET /api/samples/99999999999999999999999` returned a plain-text HTTP 500:
the path param was a bare `int`, so a value past 2^63-1 passed validation and
only failed when sqlite3 tried to bind it. Its siblings in albums.py,
activity.py and annotations.py all use `PathId`, which turns the same probe
into a 422. `?cluster=` had the identical hole on both routes that accept it.

    cd backend && pytest tests/test_samples.py
"""
import pytest
from fastapi.testclient import TestClient

from app import db
from app.api.deps import MAX_SQLITE_INT
from app.main import app

TOO_BIG = 2**63          # one past what SQLite's INTEGER can hold


@pytest.fixture(scope="module")
def client():
    conn = db.connect()
    db.init_db(conn)
    conn.commit()
    conn.close()
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("path", ["/api/samples/{}", "/api/samples/{}/similar"])
def test_sample_path_ids_are_bounded_like_their_siblings(client, path):
    assert client.get(path.format(TOO_BIG)).status_code == 422
    assert client.get(path.format(0)).status_code == 422
    assert client.get(path.format(-1)).status_code == 422
    # The largest id SQLite can actually store is still a real lookup, not a
    # validation error: it 404s (or returns an empty neighbour list).
    assert client.get(path.format(MAX_SQLITE_INT)).status_code in (200, 404, 503)


def test_cluster_filter_rejects_what_used_to_500(client):
    for route in ("/api/samples", "/api/export"):
        assert client.get(route, params={"cluster": TOO_BIG}).status_code == 422, route
        # A real cluster id — including 0, which k-means labels from — answers.
        assert client.get(route, params={"cluster": 0}).status_code == 200, route
