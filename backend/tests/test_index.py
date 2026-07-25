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


# -- difficulty axes: percentile bucketing (pure numpy, no model needed) ------

def test_percentile_buckets_span_the_scale_and_preserve_order():
    from app.analyze import _percentile_buckets

    raw = {i: float(i) for i in range(11)}          # evenly spaced inputs
    got = _percentile_buckets(raw)
    assert got[0] == 0 and got[10] == 10
    assert [got[i] for i in range(11)] == sorted(got[i] for i in range(11))


def test_percentile_buckets_are_scale_invariant():
    """The point of ranking: the bucket depends on order, not on magnitude, so
    a Laplacian variance and a cosine distance become comparable."""
    from app.analyze import _percentile_buckets

    small = _percentile_buckets({1: 0.001, 2: 0.002, 3: 0.003})
    huge = _percentile_buckets({1: 10.0, 2: 5000.0, 3: 900000.0})
    assert small == huge


def test_percentile_buckets_invert_flips_the_scale():
    from app.analyze import _percentile_buckets

    values = {1: 1.0, 2: 2.0, 3: 3.0}
    assert _percentile_buckets(values)[3] == 10
    assert _percentile_buckets(values, invert=True)[3] == 0


def test_percentile_buckets_are_deterministic_under_ties():
    """Ties share the lowest bucket, which is what makes the pass idempotent."""
    from app.analyze import _percentile_buckets

    values = {1: 5.0, 2: 5.0, 3: 5.0, 4: 9.0}
    first = _percentile_buckets(values)
    assert first == _percentile_buckets(values)
    assert first[1] == first[2] == first[3]
    assert first[4] > first[1]


def test_percentile_buckets_handle_degenerate_input():
    from app.analyze import _percentile_buckets

    assert _percentile_buckets({}) == {}
    assert _percentile_buckets({7: 1.0}) == {7: 0}   # one sample has no spread
