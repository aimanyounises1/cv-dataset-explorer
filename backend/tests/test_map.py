"""The map endpoint, pinned. It was the one router with no test of its own,
and its failure mode is the quiet kind this suite exists to prevent: the
query's JOIN + json_extract could regress into returning [] for every point
and the scatter page would render an empty map that reads as "no projection"
rather than as a defect.

Pinned behaviours, read off the query in app/api/map.py:
- only samples with BOTH umap coordinates appear;
- a NULL cluster is coerced to 0 (the UI colours by cluster id);
- `isolation` is extracted from stored axis_detail JSON, never recomputed;
- `agreement` is the per-sample mean of non-NULL caption agreements.

Same isolation pattern as test_sample_detail.py: this module owns its DB by
repointing `config.DB_PATH`, because the shared test env dir is populated by
other modules and inserting there would couple to collection order.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import config, db
from app.db import AXES
from app.main import app

DETAIL = {"rarity": {"word_rarity": 7.4012, "isolation": 0.6183}}
SCORES = {"legibility": 9, "rarity": 6, "difficulty": 8, "clutter": 4}


@pytest.fixture(scope="module")
def ids(tmp_path_factory):
    """filename -> sample id, in a database this module owns."""
    original = config.DB_PATH
    config.DB_PATH = tmp_path_factory.mktemp("map") / "map.db"

    conn = db.connect()
    db.init_db(conn)
    out = {}
    for name in ("map_on.jpg", "map_nocluster.jpg", "map_off.jpg"):
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
            "VALUES ('flickr8k', ?, 'train', 400, 300, 1000)", (name,))
        out[name] = cur.lastrowid
    # Projected, clustered, measured: the fully-populated point.
    conn.execute(
        "UPDATE samples SET umap_x=1.5, umap_y=-2.25, cluster=3, "
        "legibility=?, rarity=?, difficulty=?, clutter=?, axis_detail=? "
        "WHERE id=?",
        (*(SCORES[a] for a in AXES), json.dumps(DETAIL), out["map_on.jpg"]))
    for idx, agreement in ((0, 0.1), (1, 0.3), (2, None)):
        conn.execute(
            "INSERT INTO captions(sample_id, idx, text, agreement) "
            "VALUES (?, ?, 'a dog on a street', ?)",
            (out["map_on.jpg"], idx, agreement))
    # Projected but never clustered or measured: the coercion case.
    conn.execute("UPDATE samples SET umap_x=0.25, umap_y=4.0 WHERE id=?",
                 (out["map_nocluster.jpg"],))
    # No projection at all: must not appear.
    conn.commit()
    conn.close()
    yield out
    config.DB_PATH = original


@pytest.fixture(scope="module")
def client(ids):
    with TestClient(app) as c:
        yield c


def test_map_serves_only_projected_samples(client, ids):
    points = {p["id"]: p for p in client.get("/api/map").json()}
    assert set(points) == {ids["map_on.jpg"], ids["map_nocluster.jpg"]}
    assert points[ids["map_on.jpg"]]["x"] == 1.5
    assert points[ids["map_on.jpg"]]["y"] == -2.25


def test_map_coerces_a_null_cluster_to_zero(client, ids):
    points = {p["id"]: p for p in client.get("/api/map").json()}
    assert points[ids["map_on.jpg"]]["cluster"] == 3
    assert points[ids["map_nocluster.jpg"]]["cluster"] == 0


def test_map_reads_isolation_out_of_stored_axis_detail(client, ids):
    points = {p["id"]: p for p in client.get("/api/map").json()}
    assert points[ids["map_on.jpg"]]["isolation"] == pytest.approx(0.6183)
    assert points[ids["map_nocluster.jpg"]]["isolation"] is None
    # The colour dimensions ride along from the stored axis buckets.
    assert points[ids["map_on.jpg"]]["rarity"] == SCORES["rarity"]
    assert points[ids["map_nocluster.jpg"]]["rarity"] is None


def test_map_averages_only_scored_captions(client, ids):
    points = {p["id"]: p for p in client.get("/api/map").json()}
    # (0.1 + 0.3) / 2 — the NULL-agreement caption must not drag the mean.
    assert points[ids["map_on.jpg"]]["agreement"] == pytest.approx(0.2)
    assert points[ids["map_nocluster.jpg"]]["agreement"] is None
