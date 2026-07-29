"""Scenario labels say what makes a group DIFFERENT, on a fixture where the
answer is known by construction.

Thirty samples, three scenes of ten. Every sample is a dog outdoors — the trait
a count-based label puts on all three groups — and each scene carries one trait
of its own, plus one trait held by too few members to title anything and one
that is barely more common here than everywhere. The labelling helpers are
exercised directly (memberships chosen, so the expected label is arithmetic,
not a guess) and then the endpoint is checked end to end.

    cd backend && pytest tests/test_scenarios.py
"""
import re
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db
from app.api import search as search_module
from app.api.search import (
    SCENARIO_MIN_LIFT,
    SCENARIO_MIN_SHARE,
    _distinctive_candidates,
    _label_parts,
    _pool_terms,
    _stem,
)
from app.main import app
from app.ml.index import EmbeddingIndex, invalidate_index

# scene (an `environment` attribute) -> the caption word only that scene carries
SCENES = {"field": "alone", "street": "people", "beach": "water"}
PER_SCENE = 10
RARE = "leash"          # given to 3 of the 10 field samples: max lift, below the floor
RARE_HOLDERS = 3
NEAR = "sunny"          # 7/10 here, 20/30 overall: more common, but not by enough to name
NEAR_HOLDERS = {"field": 7, "street": 7, "beach": 6}

EVIDENCE_RE = re.compile(r"^(\d+)/(\d+) (\S+) — (\d+)% here vs (\d+)% across the results$")


def _vec(weights: dict[int, float]) -> np.ndarray:
    v = np.zeros(8, dtype=np.float32)
    for idx, w in weights.items():
        v[idx] = w
    return v / (np.linalg.norm(v) or 1.0)


class FakeEmbedder:
    """Every text sits equidistant from the three scenes, so the whole fixture
    ranks in and the grouping — not the ranking — decides the outcome."""

    def encode_texts(self, texts):
        return np.stack([_vec({0: 1.0, 1: 1.0, 2: 1.0}) for _ in texts])

    def encode_images(self, images):
        return np.stack([_vec({0: 1.0}) for _ in images])


@pytest.fixture(scope="module")
def ctx():
    """(client, conn, {scene: [sample ids]})."""
    conn = db.connect()
    db.init_db(conn)
    by_scene: dict[str, list[int]] = {s: [] for s in SCENES}
    ids, vecs = [], []
    idx = 0
    for si, (scene, word) in enumerate(SCENES.items()):
        for j in range(PER_SCENE):
            cur = conn.execute(
                "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
                "VALUES ('flickr8k', ?, 'train', 300, 200, 1)",
                (f"scn_{scene}_{j}.jpg",))
            sid = cur.lastrowid
            # Shared by all thirty: "dog". Distinguishing: the scene's own word.
            text = f"a dog {word} photo number {j}"
            if scene == "field" and j < RARE_HOLDERS:
                text += f" {RARE}"
            # NEAR sits on the tail of every scene, so the mixed subset the
            # tests build from the first three of each scene never carries it.
            if j >= PER_SCENE - NEAR_HOLDERS[scene]:
                text += f" {NEAR}"
            conn.execute("INSERT INTO captions(sample_id, idx, text) VALUES (?, 0, ?)",
                         (sid, text))
            conn.execute(
                "INSERT INTO attributes(sample_id, grp, label, confidence) "
                "VALUES (?, 'setting', 'outdoor', 0.9)", (sid,))
            conn.execute(
                "INSERT INTO attributes(sample_id, grp, label, confidence) "
                "VALUES (?, 'environment', ?, 0.9)", (sid, scene))
            by_scene[scene].append(sid)
            ids.append(sid)
            vecs.append(_vec({si: 1.0, 3 + (idx % 5): 0.01 * (idx + 1)}))
            idx += 1
    conn.commit()
    EmbeddingIndex.save(np.array(ids), np.stack(vecs), kind="image")
    invalidate_index()
    image_index = EmbeddingIndex.load("image")
    real_get_runtime = search_module.get_retrieval_bundle
    search_module.get_retrieval_bundle = lambda: SimpleNamespace(
        encoder=FakeEmbedder(),
        image_index=image_index,
        caption_index=None,
    )
    try:
        with TestClient(app) as c:
            yield c, conn, by_scene
    finally:
        search_module.get_retrieval_bundle = real_get_runtime
        from app import config
        for f in config.EMB_DIR.glob("*.npy"):
            f.unlink(missing_ok=True)
        invalidate_index()
        qmarks = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM attributes WHERE sample_id IN ({qmarks})", ids)
        conn.execute(f"DELETE FROM samples WHERE id IN ({qmarks})", ids)
        conn.commit()
        conn.close()


def _pool(by_scene) -> list[int]:
    return [sid for scene in SCENES for sid in by_scene[scene]]


# -- the labelling itself, on memberships we choose -------------------------


def test_the_trait_everything_shares_never_titles_a_group(ctx):
    """"dog" and "outdoor" are on all thirty: 100% here, 100% across the
    results, so they distinguish nothing and cannot reach a label."""
    _client, conn, by_scene = ctx
    pool = _pool(by_scene)
    terms = _pool_terms(conn, pool)
    assert len(terms["caption:dog"][3]) == len(pool)          # measured on every sample
    for scene in SCENES:
        cands = _distinctive_candidates(terms, by_scene[scene], len(pool))
        names = [c[6] for c in cands]
        assert "caption:dog" not in names
        assert "setting:outdoor" not in names
        parts, _ev = _label_parts(cands, PER_SCENE, set())
        assert "dog" not in parts and "outdoor" not in parts


def test_the_distinguishing_traits_do_title_their_group(ctx):
    _client, conn, by_scene = ctx
    pool = _pool(by_scene)
    terms = _pool_terms(conn, pool)
    for scene, word in SCENES.items():
        cands = _distinctive_candidates(terms, by_scene[scene], len(pool))
        parts, evidence = _label_parts(cands, PER_SCENE, set())
        # Both traits are on 10/10 here and 10/30 overall — lift 3.0, the most a
        # third of the pool can reach; attributes outrank caption words on ties.
        assert parts == [scene, word], f"{scene}: {parts}"
        assert cands[0][0] == pytest.approx(3.0)
        assert evidence.startswith(f"{PER_SCENE}/{PER_SCENE} environment:{scene} — "
                                   f"100% here vs 33% across the results")


def test_a_trait_on_three_of_ten_never_titles_the_group(ctx):
    """`RARE` is exclusive to the field group, so its lift is the maximum 3.0 —
    but it describes 30% of the group, under the floor, and stays out."""
    _client, conn, by_scene = ctx
    pool = _pool(by_scene)
    terms = _pool_terms(conn, pool)
    assert len(terms[f"caption:{RARE}"][3]) == RARE_HOLDERS
    assert RARE_HOLDERS / PER_SCENE < SCENARIO_MIN_SHARE
    cands = _distinctive_candidates(terms, by_scene["field"], len(pool))
    assert f"caption:{RARE}" not in [c[6] for c in cands]
    parts, evidence = _label_parts(cands, PER_SCENE, set())
    assert RARE not in parts and RARE not in evidence


def test_a_trait_barely_more_common_here_is_not_named(ctx):
    """`NEAR` is on 7 of the 10 field images and 20 of the 30 results: more
    common here, but "70% here vs 67% across the results" names nothing."""
    _client, conn, by_scene = ctx
    pool = _pool(by_scene)
    terms = _pool_terms(conn, pool)
    holders = terms[f"caption:{NEAR}"][3]
    assert len(holders) == sum(NEAR_HOLDERS.values())
    n_in = len(set(by_scene["field"]) & holders)
    lift = (n_in / PER_SCENE) / (len(holders) / len(pool))
    assert 1.0 < lift < SCENARIO_MIN_LIFT
    cands = _distinctive_candidates(terms, by_scene["field"], len(pool))
    assert f"caption:{NEAR}" not in [c[6] for c in cands]


def test_a_claimed_lead_makes_the_label_descend(ctx):
    _client, conn, by_scene = ctx
    pool = _pool(by_scene)
    terms = _pool_terms(conn, pool)
    cands = _distinctive_candidates(terms, by_scene["field"], len(pool))
    parts, _ev = _label_parts(cands, PER_SCENE, {_stem("field")})
    assert parts == ["alone"]          # descended past the trait already claimed
    parts, evidence = _label_parts(cands, PER_SCENE, {_stem("field"), _stem("alone")})
    assert parts == []                 # nothing distinctive left to be titled with
    assert evidence.startswith("every trait over-represented here already titles another group")
    assert "10/10 environment:field — 100% here vs 33% across the results" in evidence


def test_a_group_with_nothing_distinctive_says_so(ctx):
    """Three from each scene: no trait reaches the floor, and the honest answer
    is that nothing separates this group — not an invented difference."""
    _client, conn, by_scene = ctx
    pool = _pool(by_scene)
    terms = _pool_terms(conn, pool)
    mixed = [sid for scene in SCENES for sid in by_scene[scene][:3]]
    cands = _distinctive_candidates(terms, mixed, len(pool))
    assert cands == []
    parts, evidence = _label_parts(cands, len(mixed), set())
    assert parts == []
    assert evidence == (f"no trait describes {SCENARIO_MIN_SHARE:.0%} of these {len(mixed)} "
                        f"images and is {SCENARIO_MIN_LIFT}x as common here as across "
                        "the results")


# -- the endpoint -----------------------------------------------------------


def test_endpoint_labels_are_distinct_and_evidenced(ctx):
    client, _conn, by_scene = ctx
    r = client.post("/api/search/scenarios", json={"text": "dog", "top_k": 120})
    assert r.status_code == 200
    groups = r.json()["groups"]
    assert len(groups) == 3
    scene_of = {sid: scene for scene, sids in by_scene.items() for sid in sids}
    labels = [g["label"] for g in groups]
    assert len(set(labels)) == 3, labels
    seen_ids: list[int] = []
    for g in groups:
        assert g["count"] == len(g["sample_ids"])          # a group is saved whole
        seen_ids += g["sample_ids"]
        head = g["label"].rsplit(" — ", 1)[0]
        assert g["label"].endswith(f"— {g['count']} images")
        assert "dog" not in head and "outdoor" not in head and RARE not in head
        # Every group here is one whole scene, so the label names that scene.
        scenes = {scene_of[sid] for sid in g["sample_ids"]}
        assert len(scenes) == 1, scenes
        scene = scenes.pop()
        assert head == f"{scene} · {SCENES[scene]}", head
        for claim in g["evidence"].split("; "):
            m = EVIDENCE_RE.match(claim)
            assert m, claim
            n_in, count, _name, pct_in, pct_bg = m.groups()
            assert int(count) == g["count"]                # both numbers, checkable
            assert int(n_in) <= int(count)
            assert int(pct_in) > int(pct_bg)               # over-represented, or no claim
    assert sorted(seen_ids) == sorted(_pool(by_scene))     # the groups partition the pool


def test_two_identical_calls_return_identical_output(ctx):
    client, _conn, _by_scene = ctx
    body = {"text": "dog", "top_k": 120}
    first = client.post("/api/search/scenarios", json=body)
    second = client.post("/api/search/scenarios", json=body)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


def test_a_label_part_must_hold_for_most_of_its_group():
    """A title is read as a claim about the images under it.

    This is the rule a live grouping broke: 80 results were titled
    "water · black" on 45 black members and 35 that were not, because the floors
    admitted a part describing a large minority at a lift of 1.10. A reader sees
    that as a misclassification, not a summary — the label says "these are the
    black ones" and a plurality of them are brown.

    So a part must describe a MAJORITY of its group and be meaningfully more
    common inside it than across the pool. Both floors are asserted here rather
    than only in a comment, because loosening either one silently reintroduces
    exactly that label.
    """
    assert SCENARIO_MIN_SHARE > 0.5 or SCENARIO_MIN_SHARE == 0.5, \
        "a label part that holds for a minority mislabels the rest of the group"
    assert SCENARIO_MIN_SHARE >= 0.5
    assert SCENARIO_MIN_LIFT >= 1.25, \
        "56% here vs 51% across the results (lift 1.10) names nothing a reader can use"
    # The concrete case, in the units the label is built from.
    black_share, black_lift = 45 / 80, (45 / 80) / 0.51
    assert not (black_share >= SCENARIO_MIN_SHARE and black_lift >= SCENARIO_MIN_LIFT), \
        "the 'water · black' label would still be produced"
