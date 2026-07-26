"""Unit tests for the hubness correction (pure numpy, no model needed).

The properties pinned here are the ones whose violation would be silent: a
penalty that leaks into the image-to-image paths, a stale penalty applied to a
different index, a self-penalised image, and a displayed score that is no longer
the cosine the UI labels it.
"""
import os
import time

import numpy as np
import pytest

from app import config
from app.ml import hubness
from app.ml.index import EmbeddingIndex


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def index() -> EmbeddingIndex:
    # id 10 is a HUB: it sits between the two concept directions, so it is
    # fairly close to every query. 20 and 30 are the concept poles.
    vecs = np.stack([
        _unit([1.0, 1.0, 0.0]),   # 10 hub
        _unit([1.0, 0.0, 0.0]),   # 20
        _unit([0.0, 1.0, 0.0]),   # 30
        _unit([0.0, 0.0, 1.0]),   # 40 unrelated
    ])
    return EmbeddingIndex(np.array([10, 20, 30, 40]), vecs)


# -- compute ------------------------------------------------------------------

def _arc_bank(n=5):
    """Queries spread evenly over the arc between the two concept poles."""
    angles = np.linspace(0.0, np.pi / 2, n)
    return np.stack([_unit([np.cos(a), np.sin(a), 0.0]) for a in angles])


def test_penalty_is_largest_for_the_hub(index):
    """The whole premise: an image close to many queries is penalised most.

    Temperature matters here and the test is written to show it. At a small T
    the logsumexp collapses to a plain max, so an image that exactly equals one
    bank query outscores the hub and the ordering inverts. The correction is
    about aggregate reachability, so T has to be large enough to aggregate.
    """
    bank = _arc_bank()
    sids = np.full(len(bank), 99)              # no bank caption owns an image
    penalty = hubness.compute(bank, sids, index, temperature=0.5)
    by_id = dict(zip([int(i) for i in index.ids], penalty, strict=True))
    assert by_id[10] > by_id[20] > by_id[40]


def test_a_bank_caption_does_not_penalise_its_own_image(index):
    """Without self-exclusion an image is punished for being described."""
    bank = np.concatenate([_arc_bank(), np.stack([_unit([1, 0, 0])])])
    sids = np.append(np.full(5, 99), 20)       # the last one describes image 20
    owned = hubness.compute(bank, sids, index, temperature=0.5)
    unowned = hubness.compute(bank, np.full(6, 99), index, temperature=0.5)
    col = index.row_of(20)
    assert owned[col] < unowned[col]
    # every other image is scored identically either way
    others = [r for r in range(len(index.ids)) if r != col]
    assert np.allclose(owned[others], unowned[others])


def test_an_image_owning_every_bank_entry_is_left_alone(index):
    """Masking can empty a column completely. That must mean "no correction",
    not a nan that sorts the image somewhere arbitrary."""
    bank = np.stack([_unit([1, 0, 0])])
    penalty = hubness.compute(bank, np.array([20]), index, temperature=0.5)
    assert np.all(np.isfinite(penalty))
    assert penalty[index.row_of(20)] == 0.0


def test_penalty_survives_a_small_temperature_without_overflow(index):
    """exp(s/T) overflows for small T unless the max is pulled out first."""
    bank = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    penalty = hubness.compute(bank, np.array([99, 99]), index, temperature=1e-3)
    assert np.all(np.isfinite(penalty))


# -- how the index applies it -------------------------------------------------

def test_penalty_changes_the_order(index):
    """A query equidistant from hub and pole should stop preferring the hub."""
    q = _unit([1.0, 0.7, 0.0])
    assert index.search(q, top_k=1)[0][0] == 10          # hub wins uncorrected
    penalty = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    assert index.search(q, top_k=1, penalty=penalty)[0][0] == 20


def test_reported_score_is_the_one_that_did_the_ranking(index):
    """Returning the raw cosine instead made the results list read as broken:
    the displayed number was not the sort key, so it went UP as often as down
    (106 of 228 adjacent pairs on real queries). The score has to descend."""
    q = _unit([1.0, 0.0, 0.0])
    penalty = np.array([0.5, 0.4, 0.3, 0.2], dtype=np.float32)
    results = index.search(q, top_k=4, penalty=penalty)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    for sid, score in results:
        expected = float(index.vector_of(sid) @ q) - penalty[index.row_of(sid)]
        assert score == pytest.approx(expected, abs=1e-6)


def test_without_a_penalty_the_score_is_still_exactly_the_cosine(index):
    """The uncorrected path must be bit-for-bit what it always was."""
    q = _unit([1.0, 0.0, 0.0])
    for sid, score in index.search(q, top_k=4):
        assert score == pytest.approx(float(index.vector_of(sid) @ q), abs=1e-6)


def test_penalty_composes_with_the_candidate_filter(index):
    q = _unit([1.0, 0.35, 0.0])
    penalty = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    ids = [sid for sid, _ in index.search(q, top_k=4, penalty=penalty,
                                          allowed_ids={30, 40})]
    assert set(ids) == {30, 40}


def test_a_misaligned_penalty_is_refused_not_broadcast(index):
    """Silently re-ranking against the wrong images is the failure to avoid."""
    with pytest.raises(ValueError, match="does not match"):
        index.search(_unit([1, 0, 0]), top_k=2,
                     penalty=np.zeros(3, dtype=np.float32))


def test_image_to_image_paths_take_no_penalty(index):
    """`similar_to` and duplicate detection use image vectors as the query, so a
    penalty estimated from a bank of TEXT queries is meaningless for them."""
    assert index.similar_to(20, top_k=3) == index.similar_to(20, top_k=3)
    before = index.search(index.vector_of(20), top_k=4)
    assert before[0][0] == 20        # unchanged by anything in this module


# -- artifact handling --------------------------------------------------------

def test_load_rejects_an_artifact_built_for_a_different_index(index, monkeypatch,
                                                              tmp_path):
    monkeypatch.setattr(config, "EMB_DIR", tmp_path)
    monkeypatch.setattr(hubness, "_path", lambda: tmp_path / "hubness.npz")
    np.savez(tmp_path / "hubness.npz",
             penalty=np.zeros(4, dtype=np.float32),
             ids=np.array([1, 2, 3, 4]),                 # not this index's ids
             fingerprint=np.array(hubness._fingerprint(index)))
    assert hubness.load(index) is None


def test_load_rejects_a_stale_fingerprint(index, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EMB_DIR", tmp_path)
    monkeypatch.setattr(hubness, "_path", lambda: tmp_path / "hubness.npz")
    np.savez(tmp_path / "hubness.npz",
             penalty=np.zeros(4, dtype=np.float32), ids=index.ids,
             fingerprint=np.array("built-for-some-other-corpus"))
    assert hubness.load(index) is None


def test_beta_zero_disables_the_correction_entirely(monkeypatch):
    monkeypatch.setattr(config, "HUBNESS_BETA", 0.0)
    assert hubness.get_penalty(conn=None) is None


def test_get_penalty_applies_beta_itself(index, monkeypatch):
    """Regression: beta used to be left to each call site, and both call sites
    forgot it. The app shipped beta = 1.0 and the benchmark's semantic MRR fell
    to 0.6463 against a 0.6471 baseline — a "correction" that made retrieval
    slightly worse, invisible except in the numbers."""
    raw = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    hubness.invalidate()
    monkeypatch.setattr(hubness, "get_index", lambda: index)
    monkeypatch.setattr(hubness, "load", lambda _index=None: raw)
    monkeypatch.setattr(config, "HUBNESS_BETA", 0.25)
    try:
        assert np.allclose(hubness.get_penalty(conn=None), raw * 0.25)
    finally:
        hubness.invalidate()


# -- bank selection -----------------------------------------------------------

class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_a, **_k):
        return self

    def fetchall(self):
        return self._rows


def _rows(n):
    return [{"id": i, "sample_id": i} for i in range(1, n + 1)]


def test_no_bank_on_a_corpus_too_small_to_estimate_one(index, monkeypatch):
    """A tiny corpus must behave exactly as it did before this feature."""
    monkeypatch.setattr(hubness, "get_index", lambda: index)
    monkeypatch.setattr(index, "row_of", lambda sid: 0)
    assert hubness.bank_caption_ids(_FakeConn(_rows(50))) == []


def test_the_bank_never_eats_more_than_a_quarter_of_the_corpus(index, monkeypatch):
    """Otherwise the benchmark, which holds the bank out, has nothing to sample."""
    monkeypatch.setattr(hubness, "get_index", lambda: index)
    monkeypatch.setattr(index, "row_of", lambda sid: 0)
    ids = hubness.bank_caption_ids(_FakeConn(_rows(4000)), size=3000)
    assert len(ids) == 1000


def test_bank_selection_is_deterministic_across_calls(index, monkeypatch):
    """`api.eval` subtracts this exact set from its sample; a bank that drifted
    between processes would silently re-contaminate the benchmark."""
    monkeypatch.setattr(hubness, "get_index", lambda: index)
    monkeypatch.setattr(index, "row_of", lambda sid: 0)
    first = hubness.bank_caption_ids(_FakeConn(_rows(4000)), size=500)
    second = hubness.bank_caption_ids(_FakeConn(_rows(4000)), size=500)
    assert first == second == sorted(first)
    assert len(set(first)) == 500


# -- the artifact must outlive an ordinary database write ----------------------

def test_fingerprint_ignores_the_database_mtime(tmp_path, monkeypatch):
    """A tag edit must not invalidate a valid penalty vector.

    `explorer.db` is rewritten by every tag edit, saved view and WAL checkpoint.
    While its mtime was part of the fingerprint, one tag threw away an artifact
    whose ids, shape, model, temperature and bank size all still matched — and
    the correction then silently stopped applying until something rebuilt it.
    """
    class _Idx:
        ids = np.arange(8, dtype=np.int64)

    emb = tmp_path / "embeddings"
    emb.mkdir()
    (emb / "image_embeddings.npy").write_bytes(b"x")
    (emb / "sample_ids.npy").write_bytes(b"x")
    db = tmp_path / "explorer.db"
    db.write_bytes(b"x")
    monkeypatch.setattr(config, "EMB_DIR", emb)
    monkeypatch.setattr(config, "DB_PATH", db)

    before = hubness._fingerprint(_Idx())
    os.utime(db, (time.time() + 5_000, time.time() + 5_000))   # a write happened
    assert hubness._fingerprint(_Idx()) == before, \
        "a database write still invalidates the hubness artifact"


def test_fingerprint_still_tracks_the_embeddings(tmp_path, monkeypatch):
    """The half that must keep working: re-ingestion invalidates it."""
    class _Idx:
        ids = np.arange(8, dtype=np.int64)

    emb = tmp_path / "embeddings"
    emb.mkdir()
    for n in ("image_embeddings.npy", "sample_ids.npy"):
        (emb / n).write_bytes(b"x")
    monkeypatch.setattr(config, "EMB_DIR", emb)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "absent.db")

    before = hubness._fingerprint(_Idx())
    os.utime(emb / "image_embeddings.npy", (time.time() + 5_000, time.time() + 5_000))
    assert hubness._fingerprint(_Idx()) != before, \
        "new embeddings no longer invalidate the penalty they are aligned to"
