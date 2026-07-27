"""The detail payload must carry the axes *and* the components behind them.

The sample page renders each difficulty axis expanded into what produced it, so
one request has to hand over both the four 0-10 buckets and the `axis_detail`
blob (blur, luma, word_rarity, agreement, and the templated `why` phrase). The
endpoint gets that for free from `SELECT *`, which is exactly the kind of
property that survives by accident until someone lists columns explicitly for a
good reason and leaves out the newest one. Cheap to pin, silent when it breaks:
the expanded axis rows would simply render empty, and nothing would 500.

The unscored case is the other half of the contract, and it is the one worth
being pedantic about. Zero is the *easy* end of every axis, so a sample whose
scores have not been computed must arrive as null rather than as four zeros —
otherwise a corpus that has never had `analyze.compute_axes` run over it reads
as a uniformly easy one, and the sparkline draws four flat bars it has no
evidence for. A real 0 and an absent score must stay distinguishable.

This module takes a database of its own rather than the suite's shared temp one.
`test_smoke` asserts the corpus is exactly its three samples and was right to,
so inserting here would break it depending on collection order. `db.connect()`
re-reads `config.DB_PATH` on every call rather than caching it, so repointing it
in the fixture covers the app's request dependency too, and restoring it
afterwards leaves the following modules untouched.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import config, db
from app.db import AXES
from app.main import app

# The shape `analyze.compute_axes` writes: per-axis raw components, rounded, plus
# a `why` phrase on the axes whose components landed in their own hard tail.
HARD_DETAIL = {
    "legibility": {"blur": 0.0042, "luma": 0.1131, "why": "blurred, dark"},
    "rarity": {"word_rarity": 7.4012, "isolation": 0.6183},
    "difficulty": {"agreement": 0.0912, "consistency": 0.4407,
                   "why": "weak caption match"},
    "clutter": {"vocab": 19.0, "length_sd": 2.1},
}
HARD_SCORES = {"legibility": 9, "rarity": 6, "difficulty": 8, "clutter": 4}

# One axis measured at the easy end, the other three never measured. This is the
# row that tells 0 apart from "no score".
EDGE_DETAIL = {"legibility": {"blur": 0.9911, "luma": 0.8804}}


@pytest.fixture(scope="module")
def ids(tmp_path_factory):
    """filename -> sample id, in a database this module owns."""
    original = config.DB_PATH
    config.DB_PATH = tmp_path_factory.mktemp("detail") / "detail.db"

    conn = db.connect()
    db.init_db(conn)
    out = {}
    for name in ("sd_hard.jpg", "sd_unscored.jpg", "sd_edge.jpg"):
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
            "VALUES ('flickr8k', ?, 'train', 400, 300, 1000)", (name,))
        out[name] = cur.lastrowid
        conn.execute("INSERT INTO captions(sample_id, idx, text) VALUES (?, 0, ?)",
                     (out[name], "a dog on a street"))
    conn.execute(
        "UPDATE samples SET legibility=?, rarity=?, difficulty=?, clutter=?, "
        "axis_detail=? WHERE id=?",
        (*(HARD_SCORES[a] for a in AXES), json.dumps(HARD_DETAIL), out["sd_hard.jpg"]))
    conn.execute("UPDATE samples SET legibility=0, axis_detail=? WHERE id=?",
                 (json.dumps(EDGE_DETAIL), out["sd_edge.jpg"]))
    conn.commit()
    conn.close()
    yield out
    config.DB_PATH = original


@pytest.fixture(scope="module")
def client(ids):
    with TestClient(app) as c:
        yield c


def _axes(client, sample_id):
    r = client.get(f"/api/samples/{sample_id}")
    assert r.status_code == 200
    return r.json()["axes"]


def test_detail_carries_every_axis_score(client, ids):
    axes = _axes(client, ids["sd_hard.jpg"])
    assert {a: axes[a] for a in AXES} == HARD_SCORES


def test_detail_carries_the_components_behind_each_axis(client, ids):
    """The numbers the expanded rows are made of, already parsed.

    Asserted as an object rather than a string: `axis_detail` is stored as JSON
    text, and handing that text through unparsed would leave the page doing the
    decoding — which is how the same blob ends up being decoded two different
    ways in two different views.
    """
    detail = _axes(client, ids["sd_hard.jpg"])["detail"]
    assert isinstance(detail, dict)
    assert set(detail) == set(AXES), "an axis cannot be expanded without its components"
    assert detail == HARD_DETAIL

    # Named explicitly because these four are what the page labels; a rename in
    # the analysis pass has to be a visible break here, not an empty row there.
    assert detail["legibility"]["blur"] == pytest.approx(0.0042)
    assert detail["legibility"]["luma"] == pytest.approx(0.1131)
    assert detail["rarity"]["word_rarity"] == pytest.approx(7.4012)
    assert detail["difficulty"]["agreement"] == pytest.approx(0.0912)


def test_the_why_phrase_survives_the_round_trip(client, ids):
    """The gutter annotation is templated in the analysis pass, not the UI, so it
    has to reach the client with the numbers that justify it."""
    detail = _axes(client, ids["sd_hard.jpg"])["detail"]
    assert detail["legibility"]["why"] == "blurred, dark"
    assert "why" not in detail["clutter"], "no component was in its own hard tail"


def test_an_unscored_sample_is_null_not_zero(client, ids):
    """The distinction the UI cannot recover on its own.

    Four zeros describe the easiest image in the corpus. A sample the analysis
    pass has never reached describes nothing, and the payload has to say so.
    """
    r = client.get(f"/api/samples/{ids['sd_unscored.jpg']}")
    assert r.status_code == 200
    assert r.json()["axes"] is None


def test_a_real_zero_is_kept_and_the_unmeasured_axes_stay_null(client, ids):
    """Measured-as-easy and not-measured coexist on one sample.

    A row with a genuine 0 must not be mistaken for an unscored one, and the
    axes beside it that were never scored must not be filled in with 0 to make
    the object look complete.
    """
    axes = _axes(client, ids["sd_edge.jpg"])
    assert axes["legibility"] == 0
    assert [axes[a] for a in AXES if a != "legibility"] == [None, None, None]
    assert axes["detail"] == EDGE_DETAIL
