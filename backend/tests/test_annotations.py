"""Annotations: normalized geometry in, rows out, images untouched.

    cd backend && pytest tests/test_annotations.py
"""
import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app


@pytest.fixture(scope="module")
def ctx():
    conn = db.connect()
    db.init_db(conn)
    cur = conn.execute(
        "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
        "VALUES ('flickr8k', 'ann_0.jpg', 'train', 300, 200, 1)")
    sid = cur.lastrowid
    conn.commit()
    with TestClient(app) as c:
        yield c, sid
    conn.execute("DELETE FROM annotations")
    conn.execute("DELETE FROM samples WHERE id = ?", (sid,))
    conn.commit()
    conn.close()


RECT = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}


def test_rect_lifecycle(ctx):
    client, sid = ctx
    r = client.post(f"/api/samples/{sid}/annotations",
                    json={"kind": "rect", "geometry": RECT, "label": "car"})
    assert r.status_code == 201
    ann = r.json()
    assert ann["sample_id"] == sid
    assert ann["geometry"] == RECT
    assert ann["label"] == "car"
    listed = client.get(f"/api/samples/{sid}/annotations").json()
    assert [a["id"] for a in listed] == [ann["id"]]
    assert client.delete(f"/api/annotations/{ann['id']}").status_code == 200
    assert client.delete(f"/api/annotations/{ann['id']}").status_code == 404
    assert client.get(f"/api/samples/{sid}/annotations").json() == []


def test_polygon_lifecycle_and_point_bounds(ctx):
    client, sid = ctx
    tri = {"points": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.8]]}
    r = client.post(f"/api/samples/{sid}/annotations",
                    json={"kind": "polygon", "geometry": tri})
    assert r.status_code == 201
    assert r.json()["geometry"] == tri
    assert r.json()["label"] is None
    client.delete(f"/api/annotations/{r.json()['id']}")

    two = {"points": [[0.1, 0.1], [0.9, 0.1]]}
    assert client.post(f"/api/samples/{sid}/annotations",
                       json={"kind": "polygon", "geometry": two}).status_code == 422
    many = {"points": [[0.5, 0.5]] * 101}
    assert client.post(f"/api/samples/{sid}/annotations",
                       json={"kind": "polygon", "geometry": many}).status_code == 422


def test_rect_geometry_is_validated(ctx):
    client, sid = ctx
    bad = [
        {"x": 1.5, "y": 0.2, "w": 0.3, "h": 0.4},     # out of range
        {"x": 0.1, "y": 0.2, "w": 0.0, "h": 0.4},     # zero width
        {"x": 0.9, "y": 0.2, "w": 0.3, "h": 0.4},     # extends past right edge
        {"x": 0.1, "y": 0.2, "w": 0.3},               # missing key
        {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "z": 1},  # extra key
        {"x": "0.1", "y": 0.2, "w": 0.3, "h": 0.4},   # string coordinate
    ]
    for geometry in bad:
        r = client.post(f"/api/samples/{sid}/annotations",
                        json={"kind": "rect", "geometry": geometry})
        assert r.status_code == 422, geometry
    # NaN is not legal JSON, but Python's server-side parser accepts the bare
    # literal — so the probe has to arrive as raw bytes; a compliant client
    # serializer refuses to produce it.
    raw = '{"kind": "rect", "geometry": {"x": NaN, "y": 0.2, "w": 0.3, "h": 0.4}}'
    r = client.post(f"/api/samples/{sid}/annotations", content=raw,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert client.get(f"/api/samples/{sid}/annotations").json() == []


def test_kind_and_label_bounds(ctx):
    client, sid = ctx
    assert client.post(f"/api/samples/{sid}/annotations",
                       json={"kind": "circle", "geometry": RECT}).status_code == 422
    assert client.post(
        f"/api/samples/{sid}/annotations",
        json={"kind": "rect", "geometry": RECT, "label": "x" * 201}).status_code == 422


def test_unknown_sample_is_404(ctx):
    client, _ = ctx
    assert client.get("/api/samples/999999/annotations").status_code == 404
    assert client.post("/api/samples/999999/annotations",
                       json={"kind": "rect", "geometry": RECT}).status_code == 404


def test_per_sample_cap(ctx):
    client, sid = ctx
    conn = db.connect()
    conn.executemany(
        "INSERT INTO annotations(sample_id, kind, geometry, label, created_at) "
        "VALUES (?, 'rect', ?, NULL, '2026-01-01T00:00:00+00:00')",
        [(sid, '{"x":0.1,"y":0.1,"w":0.1,"h":0.1}')] * 200)
    conn.commit()
    r = client.post(f"/api/samples/{sid}/annotations",
                    json={"kind": "rect", "geometry": RECT})
    assert r.status_code == 400
    assert "200" in r.json()["detail"]
    conn.execute("DELETE FROM annotations WHERE sample_id = ?", (sid,))
    conn.commit()
    conn.close()


def test_path_bounds(ctx):
    client, _ = ctx
    assert client.get(f"/api/samples/{2**63}/annotations").status_code == 422
    assert client.delete("/api/annotations/0").status_code == 422
