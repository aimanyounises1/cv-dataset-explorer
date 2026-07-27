"""Albums: lifecycle, ordered membership, reorder exactness, tag conversion,
cover fallback, and the input bounds.

Runs on the full app (module-scoped TestClient) over a handful of seeded
samples, all removed on teardown so module order stays irrelevant.
Data-dir isolation happens in conftest.py, before any `app` import.

    cd backend && pytest tests/test_albums.py
"""
import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app


@pytest.fixture(scope="module")
def ctx():
    """(client, sample_ids) — five seeded samples with one caption each."""
    conn = db.connect()
    db.init_db(conn)
    sids = []
    for i in range(5):
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
            "VALUES ('flickr8k', ?, 'train', 300, 200, 1)", (f"alb_{i}.jpg",))
        conn.execute("INSERT INTO captions(sample_id, idx, text) VALUES (?, 0, ?)",
                     (cur.lastrowid, f"album seed caption {i}"))
        sids.append(cur.lastrowid)
    conn.commit()
    with TestClient(app) as c:
        yield c, sids
    # Samples cascade captions and sample_tags; albums carry no FK, so their
    # rows go explicitly — the same invariant the DELETE endpoint enforces.
    conn.execute("DELETE FROM album_items")
    conn.execute("DELETE FROM albums")
    qmarks = ",".join("?" * len(sids))
    conn.execute(f"DELETE FROM samples WHERE id IN ({qmarks})", sids)
    conn.execute("DELETE FROM tags WHERE id NOT IN (SELECT DISTINCT tag_id FROM sample_tags)")
    conn.commit()
    conn.close()


def test_lifecycle(ctx):
    client, _ = ctx
    r = client.post("/api/albums", json={"name": "night set", "summary": "dark scenes",
                                         "category": "curation", "notes": "wip"})
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "night set"
    assert body["origin"] == "manual"
    assert body["item_count"] == 0
    assert body["cover"] is None
    assert body["items"] == []
    aid = body["id"]

    assert "night set" in [a["name"] for a in client.get("/api/albums").json()]

    r = client.patch(f"/api/albums/{aid}", json={"summary": "only night scenes"})
    assert r.status_code == 200
    assert r.json()["summary"] == "only night scenes"
    assert r.json()["notes"] == "wip"            # untouched fields survive PATCH

    assert client.delete(f"/api/albums/{aid}").status_code == 200
    assert client.get(f"/api/albums/{aid}").status_code == 404
    assert client.delete(f"/api/albums/{aid}").status_code == 404


def test_name_is_trimmed_and_empty_rejected(ctx):
    client, _ = ctx
    r = client.post("/api/albums", json={"name": "  padded album  "})
    assert r.status_code == 201
    assert r.json()["name"] == "padded album"
    assert client.post("/api/albums", json={"name": "   "}).status_code == 400


def test_duplicate_name_conflicts_on_create_and_rename(ctx):
    client, _ = ctx
    assert client.post("/api/albums", json={"name": "dupe-a"}).status_code == 201
    assert client.post("/api/albums", json={"name": "dupe-a"}).status_code == 409
    b = client.post("/api/albums", json={"name": "dupe-b"}).json()
    assert client.patch(f"/api/albums/{b['id']}",
                        json={"name": "dupe-a"}).status_code == 409
    # Renaming to its own name is not a conflict.
    assert client.patch(f"/api/albums/{b['id']}",
                        json={"name": "dupe-b"}).status_code == 200


def test_add_items_order_and_this_call_count(ctx):
    client, s = ctx
    aid = client.post("/api/albums", json={"name": "ordered"}).json()["id"]
    r = client.post(f"/api/albums/{aid}/items",
                    json={"sample_ids": [s[2], s[0], s[1]]})
    assert r.status_code == 200
    assert r.json()["added"] == 3
    got = [i["id"] for i in client.get(f"/api/albums/{aid}").json()["items"]]
    assert got == [s[2], s[0], s[1]]             # given order, not id order

    # Duplicates are ignored and the count is THIS call's inserts, not a total.
    r = client.post(f"/api/albums/{aid}/items", json={"sample_ids": [s[0], s[3]]})
    assert r.json()["added"] == 1
    got = [i["id"] for i in client.get(f"/api/albums/{aid}").json()["items"]]
    assert got == [s[2], s[0], s[1], s[3]]       # s[0] kept its place


def test_add_unknown_ids_are_ignored(ctx):
    client, s = ctx
    aid = client.post("/api/albums", json={"name": "unknown ids"}).json()["id"]
    r = client.post(f"/api/albums/{aid}/items",
                    json={"sample_ids": [s[4], 999_999_999]})
    assert r.status_code == 200
    assert r.json()["added"] == 1
    assert [i["id"] for i in client.get(f"/api/albums/{aid}").json()["items"]] == [s[4]]


def test_remove_item_compacts_positions(ctx):
    client, s = ctx
    aid = client.post("/api/albums", json={"name": "compaction"}).json()["id"]
    client.post(f"/api/albums/{aid}/items", json={"sample_ids": [s[0], s[1], s[2]]})
    assert client.delete(f"/api/albums/{aid}/items/{s[1]}").status_code == 200
    got = [i["id"] for i in client.get(f"/api/albums/{aid}").json()["items"]]
    assert got == [s[0], s[2]]
    conn = db.connect()
    positions = [r["position"] for r in conn.execute(
        "SELECT position FROM album_items WHERE album_id = ? ORDER BY position", (aid,))]
    conn.close()
    assert positions == [0, 1]                   # dense again, order preserved
    # Not a member (already removed) and unknown album are both 404s.
    assert client.delete(f"/api/albums/{aid}/items/{s[1]}").status_code == 404
    assert client.delete(f"/api/albums/999999/items/{s[1]}").status_code == 404


def test_reorder_items_round_trip_and_exactness(ctx):
    client, s = ctx
    aid = client.post("/api/albums", json={"name": "reorder"}).json()["id"]
    client.post(f"/api/albums/{aid}/items", json={"sample_ids": [s[0], s[1], s[2]]})
    r = client.put(f"/api/albums/{aid}/items/order",
                   json={"sample_ids": [s[2], s[0], s[1]]})
    assert r.status_code == 200
    got = [i["id"] for i in client.get(f"/api/albums/{aid}").json()["items"]]
    assert got == [s[2], s[0], s[1]]

    # Wrong membership is refused, naming the difference.
    r = client.put(f"/api/albums/{aid}/items/order", json={"sample_ids": [s[2], s[0]]})
    assert r.status_code == 400
    assert str(s[1]) in r.json()["detail"]
    r = client.put(f"/api/albums/{aid}/items/order",
                   json={"sample_ids": [s[2], s[0], s[1], s[4]]})
    assert r.status_code == 400
    assert str(s[4]) in r.json()["detail"]
    assert client.put(f"/api/albums/{aid}/items/order",
                      json={"sample_ids": [s[2], s[2], s[0]]}).status_code == 400
    # The failed reorders changed nothing.
    got = [i["id"] for i in client.get(f"/api/albums/{aid}").json()["items"]]
    assert got == [s[2], s[0], s[1]]


def test_reorder_albums(ctx):
    client, _ = ctx
    ids = [a["id"] for a in client.get("/api/albums").json()]
    assert len(ids) >= 2
    r = client.put("/api/albums/order", json={"album_ids": list(reversed(ids))})
    assert r.status_code == 200
    assert [a["id"] for a in client.get("/api/albums").json()] == list(reversed(ids))
    # Exactness holds here too: a subset is refused.
    assert client.put("/api/albums/order",
                      json={"album_ids": ids[:1]}).status_code == 400


def test_from_tag_conversion(ctx):
    client, s = ctx
    for sid in (s[3], s[1]):
        assert client.post(f"/api/samples/{sid}/tags",
                           json={"name": "alb-scratch"}).status_code == 200
    r = client.post("/api/albums/from-tag", json={"tag": "alb-scratch"})
    assert r.status_code == 201
    body = r.json()
    assert body["origin"] == "tag"
    assert body["name"] == "alb-scratch"
    # Members ordered by sample id, regardless of tagging order.
    assert [i["id"] for i in body["items"]] == sorted([s[1], s[3]])
    # The tag survives conversion — tags stay labels.
    tags = {t["name"]: t["count"] for t in client.get("/api/tags").json()}
    assert tags.get("alb-scratch") == 2

    assert client.post("/api/albums/from-tag",
                       json={"tag": "alb-scratch"}).status_code == 409
    assert client.post("/api/albums/from-tag",
                       json={"tag": "no-such-tag"}).status_code == 404


def test_cover_fallback_and_explicit_cover(ctx):
    client, s = ctx
    aid = client.post("/api/albums", json={"name": "covers"}).json()["id"]
    assert client.get(f"/api/albums/{aid}").json()["cover"] is None
    client.post(f"/api/albums/{aid}/items", json={"sample_ids": [s[1], s[0]]})
    first_thumb = client.get(f"/api/albums/{aid}").json()["items"][0]["thumb_url"]
    assert client.get(f"/api/albums/{aid}").json()["cover"] == first_thumb

    r = client.patch(f"/api/albums/{aid}", json={"cover_sample_id": s[0]})
    assert r.status_code == 200
    assert r.json()["cover"] != first_thumb      # explicit cover honoured
    assert r.json()["cover_sample_id"] == s[0]
    # A non-member cannot front the album.
    assert client.patch(f"/api/albums/{aid}",
                        json={"cover_sample_id": s[4]}).status_code == 400
    # Explicit null clears back to the first-item fallback.
    r = client.patch(f"/api/albums/{aid}", json={"cover_sample_id": None})
    assert r.json()["cover"] == first_thumb
    # Removing the chosen cover clears it rather than leaving a ghost.
    client.patch(f"/api/albums/{aid}", json={"cover_sample_id": s[0]})
    client.delete(f"/api/albums/{aid}/items/{s[0]}")
    body = client.get(f"/api/albums/{aid}").json()
    assert body["cover_sample_id"] is None
    assert body["cover"] == first_thumb          # s[1] is first again

    # The list endpoint reports the same cover as the detail.
    listed = [a for a in client.get("/api/albums").json() if a["id"] == aid][0]
    assert listed["cover"] == first_thumb
    assert listed["item_count"] == 1


def test_bounds(ctx):
    client, s = ctx
    assert client.post("/api/albums", json={"name": "x" * 201}).status_code == 422
    assert client.post("/api/albums",
                       json={"name": "ok", "summary": "y" * 2001}).status_code == 422
    aid = client.post("/api/albums", json={"name": "bounds"}).json()["id"]
    r = client.post(f"/api/albums/{aid}/items",
                    json={"sample_ids": list(range(1, 100_002))})
    assert r.status_code == 400
    assert "100,000" in r.json()["detail"]
    # Past SQLite's signed 64-bit range: a 422, never an OverflowError 500.
    assert client.get(f"/api/albums/{2**63}").status_code == 422
    assert client.post(f"/api/albums/{aid}/items",
                       json={"sample_ids": [2**63]}).status_code == 422
    assert client.patch(f"/api/albums/{aid}",
                        json={"cover_sample_id": 2**63}).status_code == 422
    assert client.delete(f"/api/albums/{aid}/items/{2**63}").status_code == 422
