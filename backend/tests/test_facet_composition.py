"""Repeated attribute facets must intersect, and must do so order-independently.

These pin a bug that produced no error and no warning: `attr` was a bare `str`,
so FastAPI bound only the last occurrence of a repeated parameter. A URL asking
for night AND indoors returned every indoor image, and reversing the two terms
returned a different set again. The address bar and the result count disagreed,
and nothing anywhere said so.

The describe panel inherited the same fault from the other end. Its rows are
counted *inside* the current selection, so a row reading "223 images" could only
be honoured by adding its facet to what was already applied — which a
single-valued parameter could not express, so the drill-through replaced the
selection and opened a set 6.6x larger than the row advertised.

The fixture is deliberately large enough (300 samples) for the describe endpoint
to have something to say. Its facets are hypergeometric-tested with a
finite-population correction, so a handful of rows in a handful-sized corpus
correctly reports nothing significant and would leave the drill assertion
vacuous.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import config, db
from app.api.deps import build_filters
from app.main import app

G_TIME, G_PLACE = "fc_time", "fc_place"

N_SAMPLES = 300
N_NIGHT = 60          # of which...
N_NIGHT_INDOOR = 50   # ...these are also indoor
N_DAY_INDOOR = 30


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A database of this module's own, not the one the rest of the suite shares.

    300 rows in the shared temp database is not a neighbourly thing to do:
    `test_smoke` asserts the corpus is exactly its own three samples, and it was
    right to. `db.connect()` reads `config.DB_PATH` on every call rather than
    caching it, so repointing it here covers the app's request dependency too,
    and restoring it afterwards leaves the following modules untouched.
    """
    original = config.DB_PATH
    config.DB_PATH = tmp_path_factory.mktemp("facets") / "facets.db"

    conn = db.connect()
    db.init_db(conn)
    for i in range(N_SAMPLES):
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
            "VALUES ('flickr8k', ?, 'train', 400, 300, 1000)", (f"fc_{i}.jpg",))
        sid = cur.lastrowid
        # Every sample carries the same caption, so a keyword query selects the
        # whole corpus and any narrowing in a search result is the facets' doing.
        ccur = conn.execute(
            "INSERT INTO captions(sample_id, idx, text) VALUES (?, 0, ?)",
            (sid, "a dog on a street"))
        conn.execute("INSERT INTO captions_fts(rowid, text) VALUES (?, ?)",
                     (ccur.lastrowid, "a dog on a street"))
        night = i < N_NIGHT
        indoor = i < N_NIGHT_INDOOR or N_NIGHT <= i < N_NIGHT + N_DAY_INDOOR
        conn.executemany(
            "INSERT INTO attributes(sample_id, grp, label, confidence) "
            "VALUES (?, ?, ?, 0.9)",
            [(sid, G_TIME, "night" if night else "day"),
             (sid, G_PLACE, "indoor" if indoor else "outdoor")])
    conn.commit()
    conn.close()
    with TestClient(app) as c:
        yield c
    config.DB_PATH = original


def _n(client, **params) -> int:
    """Size of a filtered set, straight off the API."""
    return client.get("/api/samples", params=params).json()["total"]


# -- the composer itself ------------------------------------------------------

def test_repeated_attr_produces_one_clause_each():
    where, params = build_filters(None, None, None, attr=["a:b", "c:d"])
    assert where.count("FROM attributes") == 2
    assert params == ["a", "b", "c", "d"]


def test_single_attr_string_still_works():
    """The composer is shared with callers that still pass a bare string."""
    where, params = build_filters(None, None, None, attr="a:b")
    assert where.count("FROM attributes") == 1
    assert params == ["a", "b"]


@pytest.mark.parametrize("attr", [None, [], "", ["", "no-colon"]])
def test_empty_and_malformed_attrs_add_nothing(attr):
    where, params = build_filters(None, None, None, attr=attr)
    assert "FROM attributes" not in where
    assert params == []


# -- through the API ----------------------------------------------------------

def test_two_facets_intersect_and_ignore_order(client):
    night, indoor = f"{G_TIME}:night", f"{G_PLACE}:indoor"
    assert _n(client, attr=[night]) == N_NIGHT
    assert _n(client, attr=[indoor]) == N_NIGHT_INDOOR + N_DAY_INDOOR

    forward = _n(client, attr=[night, indoor])
    reverse = _n(client, attr=[indoor, night])

    assert forward == N_NIGHT_INDOOR, "not an intersection"
    assert reverse == N_NIGHT_INDOOR, "the result depends on parameter order"


def test_contradictory_facets_return_nothing_rather_than_one_of_them(client):
    """Two labels from the same group cannot both hold: one sample, one label.

    Last-wins turned this into a silent 60 or 240 images depending on which term
    came second. An impossible filter should say so by being empty.
    """
    assert _n(client, attr=[f"{G_TIME}:night", f"{G_TIME}:day"]) == 0


def test_describe_row_count_matches_where_its_drill_lands(client):
    """A row advertising N images must open exactly those N.

    This is the assertion the old drill failed: the panel said 223 and the link
    went to 1,483, because it dropped the selection the row was measured inside.
    """
    night = f"{G_TIME}:night"
    body = client.get("/api/describe", params={"attr": [night]}).json()
    assert body["set_size"] == N_NIGHT

    rows = [f for f in body["over"] + body["under"] if f["group"] == G_PLACE]
    assert rows, "the fixture is built so indoor is over-represented among night"

    for row in rows:
        facet = row["drill"].split("=", 1)[1]
        merged = _n(client, attr=[night, facet])
        assert merged == row["count"], (
            f"row '{facet}' advertises {row['count']} images, "
            f"its drill lands on {merged}")


def test_export_of_an_intersection_is_the_intersection(client):
    """The download must be the set on screen, not a wider one.

    The export link is built from the same parameters, so a dropped repeat here
    would hand the user a CSV of 80 rows for a gallery showing 50.
    """
    night, indoor = f"{G_TIME}:night", f"{G_PLACE}:indoor"
    r = client.get("/api/export",
                   params={"attr": [night, indoor], "format": "jsonl"})
    assert r.status_code == 200
    lines = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]

    # The first line is the provenance manifest, which is the other half of the
    # contract: an export that quietly dropped a facet would also mis-record
    # which query produced it, and the file would misdescribe itself forever.
    manifest = lines[0]["_manifest"]
    assert manifest["attr"] == [night, indoor]
    assert len(lines) - 1 == N_NIGHT_INDOOR


def test_search_narrows_by_every_facet_not_just_the_last(client):
    """Filtering to an intersection and then typing a query must stay narrow.

    `/api/search` composes the same filters through a different path, so it had
    the same last-wins fault: a gallery showing 50 images would widen back to 80
    the moment the user searched inside it.
    """
    night, indoor = f"{G_TIME}:night", f"{G_PLACE}:indoor"

    def hits(**kw) -> set[int]:
        body = client.get("/api/search",
                          params={"q": "dog", "mode": "keyword",
                                  "top_k": 200, **kw}).json()
        return {i["id"] for i in body["items"]}

    only_night, only_indoor = hits(attr=[night]), hits(attr=[indoor])
    both = hits(attr=[night, indoor])

    assert both == only_night & only_indoor
    assert both == hits(attr=[indoor, night]), "depends on parameter order"


def test_search_post_body_still_accepts_a_single_facet(client):
    """The JSON body kept its old shape: a bare string means one facet."""
    night = f"{G_TIME}:night"
    one = client.post("/api/search", json={"q": "dog", "mode": "keyword",
                                           "top_k": 200, "attr": night}).json()
    listed = client.post("/api/search", json={"q": "dog", "mode": "keyword",
                                              "top_k": 200, "attr": [night]}).json()
    assert {i["id"] for i in one["items"]} == {i["id"] for i in listed["items"]}
