"""Unit tests for the exact embedding index (pure numpy, no model needed)."""
import numpy as np
import pytest

from app.ml.index import EmbeddingIndex


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def index() -> EmbeddingIndex:
    # Four samples on distinct directions; ids intentionally non-contiguous.
    vecs = np.stack([
        _unit([1, 0, 0]),
        _unit([0.9, 0.1, 0]),   # near-duplicate of the first
        _unit([0, 1, 0]),
        _unit([0, 0, 1]),
    ])
    return EmbeddingIndex(np.array([10, 20, 30, 40]), vecs)


def test_search_orders_by_similarity(index):
    results = index.search(_unit([1, 0, 0]), top_k=4)
    assert [sid for sid, _ in results][:2] == [10, 20]
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_search_respects_allowed_ids(index):
    results = index.search(_unit([1, 0, 0]), top_k=4, allowed_ids={30, 40})
    ids = [sid for sid, _ in results]
    assert set(ids) == {30, 40}          # filtered candidates only
    assert 10 not in ids and 20 not in ids


def test_search_allowed_ids_never_empties_results(index):
    # Even though the best match is excluded, filtered results still appear.
    results = index.search(_unit([1, 0, 0]), top_k=1, allowed_ids={30})
    assert results == [(30, pytest.approx(0.0, abs=1e-5))]


def test_similar_to_excludes_self(index):
    results = index.similar_to(10, top_k=2)
    assert [sid for sid, _ in results][0] == 20
    assert all(sid != 10 for sid, _ in results)


def test_duplicate_pairs(index):
    pairs = index.duplicate_pairs(threshold=0.95)
    assert pairs and pairs[0][:2] == (10, 20)
    assert all(a < b or (a, b) == (10, 20) for a, b, _ in pairs)


def test_duplicate_pairs_no_false_positives(index):
    assert index.duplicate_pairs(threshold=0.999) == []
