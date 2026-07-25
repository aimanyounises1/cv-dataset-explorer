"""ID-list ingress: bringing a set computed elsewhere back into the tool.

The round-trip is the point — a slice found here must be able to leave as a list
of ids and come back as the identical set — so the last test in this file is the
one that matters most.

Own fixture, own filenames, full teardown: conftest points the whole session at
one temp data dir, so a module that left rows behind would change another
module's counts.
"""
import pytest
from fastapi.testclient import TestClient

from app import db
from app.api.deps import MAX_ID_LIST, parse_id_list
from app.main import app

SAMPLES = [
    ("idf_a.jpg", "train", "a red kayak on a calm lake", 2),
    ("idf_b.jpg", "train", "a cyclist climbing a steep hill", 5),
    ("idf_c.jpg", "validation", "a wheelchair user crossing a street", 9),
]


@pytest.fixture(scope="module")
def client():
    conn = db.connect()
    db.init_db(conn)
    ids = []
    for filename, split, caption, difficulty in SAMPLES:
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize, "
            "difficulty) VALUES ('flickr8k', ?, ?, 320, 240, 900, ?)",
            (filename, split, difficulty))
        ids.append(cur.lastrowid)
        ccur = conn.execute(
            "INSERT INTO captions(sample_id, idx, text) VALUES (?, 0, ?)",
            (cur.lastrowid, caption))
        conn.execute("INSERT INTO captions_fts(rowid, text) VALUES (?, ?)",
                     (ccur.lastrowid, caption))
    conn.commit()
    try:
        with TestClient(app) as c:
            c.sample_ids = ids  # type: ignore[attr-defined]
            yield c
    finally:
        for r in conn.execute(
            f"SELECT id, text FROM captions WHERE sample_id IN ({','.join('?' * len(ids))})",
                ids).fetchall():
            conn.execute("INSERT INTO captions_fts(captions_fts, rowid, text) "
                         "VALUES('delete', ?, ?)", (r["id"], r["text"]))
        conn.execute(f"DELETE FROM captions WHERE sample_id IN ({','.join('?' * len(ids))})", ids)
        conn.execute(f"DELETE FROM samples WHERE id IN ({','.join('?' * len(ids))})", ids)
        conn.commit()
        conn.close()


# -- parsing ------------------------------------------------------------------

def test_parse_accepts_commas_newlines_and_both():
    assert parse_id_list("1,2,3") == ["1", "2", "3"]
    assert parse_id_list("1\n2\n3") == ["1", "2", "3"]
    assert parse_id_list("1, 2\n3,\n\n4 ") == ["1", "2", "3", "4"]


def test_parse_drops_repeats_but_keeps_order():
    # A duplicated line must not let one sample weight a slice twice.
    assert parse_id_list("7,3,7,1,3") == ["7", "3", "1"]


def test_parse_of_nothing_is_empty_not_an_error():
    assert parse_id_list(None) == []
    assert parse_id_list("") == []
    assert parse_id_list("  ,\n , ") == []


# -- filtering ----------------------------------------------------------------

def test_ids_filter_selects_exactly_the_listed_samples(client):
    a, b, _c = client.sample_ids
    body = client.get("/api/samples", params={"ids": f"{a},{b}"}).json()
    assert body["total"] == 2
    assert {i["id"] for i in body["items"]} == {a, b}


def test_ids_and_filenames_may_be_mixed(client):
    # ids come from this tool's export; filenames come from anything that
    # touched the images on disk. Both are things a researcher already has.
    a = client.sample_ids[0]
    body = client.get("/api/samples", params={"ids": f"{a}\nidf_c.jpg"}).json()
    assert {i["filename"] for i in body["items"]} == {"idf_a.jpg", "idf_c.jpg"}


def test_unknown_entries_are_ignored_not_rejected(client):
    a = client.sample_ids[0]
    body = client.get("/api/samples", params={
        "ids": f"{a},999999,not_a_real_file.jpg"}).json()
    assert body["total"] == 1 and body["items"][0]["id"] == a


def test_ids_compose_with_other_filters_rather_than_replacing_them(client):
    a, b, c = client.sample_ids
    all_three = f"{a},{b},{c}"
    assert client.get("/api/samples", params={"ids": all_three}).json()["total"] == 3
    # Same list, narrowed by split, then by an axis range.
    assert client.get("/api/samples", params={
        "ids": all_three, "split": "train"}).json()["total"] == 2
    assert client.get("/api/samples", params={
        "ids": all_three, "split": "train", "difficulty_min": 4}).json()["total"] == 1


def test_id_filter_is_applied_inside_the_ranking_not_after_it(client):
    """The same shape as test_keyword_search_respects_filters_in_sql.

    "kayak" matches exactly one sample. Restricting the id list to a *different*
    sample must yield nothing: if the predicate were applied after the ranking's
    LIMIT, the match would enter the result set and be trimmed afterwards, and a
    reported total would be computed over the wrong set.
    """
    a, b, _c = client.sample_ids
    assert len(client.get("/api/search", params={
        "q": "kayak", "mode": "keyword"}).json()["items"]) == 1
    assert client.get("/api/search", params={
        "q": "kayak", "mode": "keyword", "ids": str(b)}).json()["items"] == []
    assert len(client.get("/api/search", params={
        "q": "kayak", "mode": "keyword", "ids": str(a)}).json()["items"]) == 1


def test_search_reports_how_many_entries_resolved(client):
    a = client.sample_ids[0]
    body = client.get("/api/search", params={
        "q": "kayak", "mode": "keyword", "ids": f"{a},999999,ghost.jpg"}).json()
    # Reported, not enforced: 1 of the 3 pasted entries exists in this dataset.
    assert body["ids_resolved"] == 1


def test_over_the_cap_is_a_clear_error(client):
    too_many = ",".join(str(i) for i in range(MAX_ID_LIST + 2))
    r = client.post("/api/search", json={"q": "kayak", "ids": too_many})
    assert r.status_code == 400
    assert str(MAX_ID_LIST) in r.json()["detail"]


def test_post_search_matches_the_get_for_the_same_query(client):
    """The POST exists only to carry a list a URL cannot; it must not become a
    second ranking implementation."""
    a, b, c = client.sample_ids
    ids = f"{a},{b},{c}"
    got = client.get("/api/search", params={"q": "a", "mode": "keyword", "ids": ids}).json()
    posted = client.post("/api/search", json={
        "q": "a", "mode": "keyword", "ids": ids}).json()
    assert [i["id"] for i in posted["items"]] == [i["id"] for i in got["items"]]


# -- the acceptance criterion -------------------------------------------------

def test_export_then_reimport_reproduces_the_identical_set(client):
    """Item 2's round trip: narrow a set, export it, paste it back, get it back.

    Exercised through the axis filter as well as the query, because that is the
    workflow the feature exists for — find the hard ones, take the list away,
    bring it back later.
    """
    narrowed = client.get("/api/samples", params={
        "split": "train", "difficulty_min": 4}).json()
    original = [i["id"] for i in narrowed["items"]]
    assert original, "fixture should leave something to export"

    exported = client.get("/api/export", params={
        "split": "train", "difficulty_min": 4}).json()
    assert [s["id"] for s in exported["samples"]] == original

    # The manifest's own ids, pasted back with no other filter.
    pasted = "\n".join(str(s["id"]) for s in exported["samples"])
    round_tripped = client.get("/api/samples", params={"ids": pasted}).json()
    assert [i["id"] for i in round_tripped["items"]] == original

    # And by filename, which is the other thing an export gives you.
    by_name = client.get("/api/samples", params={
        "ids": ",".join(s["filename"] for s in exported["samples"])}).json()
    assert [i["id"] for i in by_name["items"]] == original
