"""Activity trail: client snapshots, server-written album events, bounds.

    cd backend && pytest tests/test_activity.py
"""
import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app


@pytest.fixture(scope="module")
def ctx():
    conn = db.connect()
    db.init_db(conn)
    conn.execute("DELETE FROM activity_events")   # known-empty start
    sids = []
    for i in range(2):
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
            "VALUES ('flickr8k', ?, 'train', 300, 200, 1)", (f"act_{i}.jpg",))
        sids.append(cur.lastrowid)
    conn.commit()
    with TestClient(app) as c:
        yield c, sids
    conn.execute("DELETE FROM activity_events")
    conn.execute("DELETE FROM album_items")
    conn.execute("DELETE FROM albums")
    qmarks = ",".join("?" * len(sids))
    conn.execute(f"DELETE FROM samples WHERE id IN ({qmarks})", sids)
    conn.execute("DELETE FROM tags WHERE id NOT IN (SELECT DISTINCT tag_id FROM sample_tags)")
    conn.commit()
    conn.close()


def test_post_and_list_round_trip(ctx):
    client, _ = ctx
    r = client.post("/api/activity",
                    json={"kind": "search_snapshot", "payload": {"q": "dog", "mode": "hybrid"}})
    assert r.status_code == 201
    event = r.json()
    assert event["kind"] == "search_snapshot"
    assert event["payload"] == {"q": "dog", "mode": "hybrid"}
    listed = client.get("/api/activity").json()
    assert listed[0]["id"] == event["id"]     # newest first
    assert listed[0]["payload"] == {"q": "dog", "mode": "hybrid"}


def test_newest_first(ctx):
    client, _ = ctx
    a = client.post("/api/activity",
                    json={"kind": "image_search", "payload": {"n": 1}}).json()["id"]
    b = client.post("/api/activity",
                    json={"kind": "composed_search", "payload": {"n": 2}}).json()["id"]
    ids = [e["id"] for e in client.get("/api/activity").json()]
    assert ids.index(b) < ids.index(a)


def test_kind_allowlist(ctx):
    client, _ = ctx
    # Server-written kinds are not client-writable: accepting them would let a
    # client forge server history.
    for bad in ("album_create", "album_delete", "made_up"):
        r = client.post("/api/activity", json={"kind": bad, "payload": {}})
        assert r.status_code == 422, bad


def test_payload_cap(ctx):
    client, _ = ctx
    r = client.post("/api/activity",
                    json={"kind": "search_snapshot", "payload": {"blob": "x" * 4000}})
    assert r.status_code == 400
    assert "4,000" in r.json()["detail"]


def test_limit_bounds(ctx):
    client, _ = ctx
    assert client.get("/api/activity", params={"limit": 0}).status_code == 422
    assert client.get("/api/activity", params={"limit": 201}).status_code == 422
    assert client.get("/api/activity", params={"limit": 200}).status_code == 200


def test_delete_one_and_clear_all(ctx):
    client, _ = ctx
    eid = client.post("/api/activity",
                      json={"kind": "search_snapshot", "payload": {}}).json()["id"]
    assert client.delete(f"/api/activity/{eid}").status_code == 200
    assert client.delete(f"/api/activity/{eid}").status_code == 404
    client.post("/api/activity", json={"kind": "search_snapshot", "payload": {}})
    r = client.delete("/api/activity")
    assert r.status_code == 200
    assert r.json()["cleared"] >= 1
    assert client.get("/api/activity").json() == []


def test_album_endpoints_write_their_own_events(ctx):
    client, s = ctx
    client.delete("/api/activity")
    aid = client.post("/api/albums", json={"name": "act-album"}).json()["id"]
    client.post(f"/api/albums/{aid}/items", json={"sample_ids": [s[0], s[1]]})
    client.post(f"/api/albums/{aid}/items", json={"sample_ids": [s[0]]})  # duplicate
    client.put(f"/api/albums/{aid}/items/order", json={"sample_ids": [s[1], s[0]]})
    client.post(f"/api/samples/{s[0]}/tags", json={"name": "act-scratch"})
    client.post("/api/albums/from-tag", json={"tag": "act-scratch"})
    client.delete(f"/api/albums/{aid}")

    events = client.get("/api/activity").json()   # newest first
    kinds = [e["kind"] for e in events][::-1]     # chronological
    assert kinds == ["album_create", "album_items_add", "album_items_add",
                     "album_reorder", "album_from_tag", "album_delete"]
    by_kind = {e["kind"]: e for e in events}
    assert by_kind["album_create"]["payload"] == {"album_id": aid, "name": "act-album", "n": 0}
    adds = [e for e in events if e["kind"] == "album_items_add"]
    # This-call semantics survive into history: 2 added, then 0.
    assert sorted(a["payload"]["n"] for a in adds) == [0, 2]
    assert by_kind["album_reorder"]["payload"]["n"] == 2
    assert by_kind["album_from_tag"]["payload"]["n"] == 1
    assert by_kind["album_delete"]["payload"]["name"] == "act-album"

    # Leave the module the way it started.
    from_tag_id = by_kind["album_from_tag"]["payload"]["album_id"]
    client.delete(f"/api/albums/{from_tag_id}")
    client.delete(f"/api/samples/{s[0]}/tags/act-scratch")
    client.delete("/api/activity")


def test_tag_approval_is_a_client_kind(ctx):
    """Approving an assistant tag proposal happens in the browser, so the
    client is the honest witness: the kind must pass the allowlist."""
    client = ctx[0]
    r = client.post("/api/activity",
                    json={"kind": "tag_approval",
                          "payload": {"tag": "edge-case", "tagged": 3}})
    assert r.status_code == 201
    assert r.json()["kind"] == "tag_approval"
    client.delete(f"/api/activity/{r.json()['id']}")
