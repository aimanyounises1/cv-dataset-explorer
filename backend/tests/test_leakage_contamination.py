"""Focused contract tests for ``GET /stats/leakage/contaminated``."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import leakage
from app.main import app


class _Conn:
    def __init__(self, splits: dict[int, str]):
        self.rows = [
            {"id": sample_id, "split": split}
            for sample_id, split in splits.items()
        ]

    def execute(self, query):
        assert query == "SELECT id, split FROM samples"
        return self.rows


class _Index:
    def __init__(self, pairs: list[tuple[int, int, float]]):
        self.pairs = pairs
        self.ids = np.asarray(
            sorted({sample_id for pair in pairs for sample_id in pair[:2]}),
            dtype=np.int64,
        )
        self.calls = 0

    def all_pairs_above(self, threshold):
        assert threshold == leakage.FLOOR
        self.calls += 1
        return self.pairs


@pytest.fixture(autouse=True)
def _clear_pair_cache():
    leakage.clear_cache()
    yield
    leakage.clear_cache()


def _request(
    monkeypatch,
    *,
    pairs: list[tuple[int, int, float]],
    splits: dict[int, str],
    threshold: float = 0.90,
    split: str | None = None,
):
    index = _Index(pairs)
    monkeypatch.setattr(leakage, "get_index", lambda: index)
    response = leakage.contaminated_ids(
        threshold=threshold,
        split=split,
        conn=_Conn(splits),
    )
    return response, index


def test_only_held_out_ids_with_a_train_neighbour_are_returned(monkeypatch):
    response, index = _request(
        monkeypatch,
        pairs=[
            (1, 2, 0.99),  # train -> test
            (3, 6, 0.98),  # validation -> train (opposite orientation)
            (4, 5, 0.97),  # validation -> test: no training side
            (7, 8, 0.96),  # train -> train: no held-out side
        ],
        splits={
            1: "train",
            2: "test",
            3: "validation",
            4: "validation",
            5: "test",
            6: "train",
            7: "train",
            8: "train",
        },
    )

    assert response.ids == [2, 3]
    assert response.total == 2
    assert response.held_out_split is None
    assert response.truncated is False
    assert index.calls == 1


def test_threshold_is_strictly_greater_than_not_greater_than_or_equal(
    monkeypatch,
):
    response, _ = _request(
        monkeypatch,
        pairs=[
            (1, 2, 0.9001),
            (1, 3, 0.9000),
            (1, 4, 0.8999),
        ],
        splits={1: "train", 2: "test", 3: "test", 4: "test"},
        threshold=0.90,
    )

    assert response.threshold == 0.90
    assert response.ids == [2]
    assert response.total == 1


def test_split_restricts_only_the_held_out_side(monkeypatch):
    pairs = [
        (1, 20, 0.99),
        (30, 1, 0.98),
    ]
    splits = {1: "train", 20: "test", 30: "validation"}

    combined, _ = _request(
        monkeypatch, pairs=pairs, splits=splits)
    leakage.clear_cache()
    test_only, _ = _request(
        monkeypatch, pairs=pairs, splits=splits, split="test")
    leakage.clear_cache()
    validation_only, _ = _request(
        monkeypatch, pairs=pairs, splits=splits, split="validation")

    assert combined.ids == [20, 30]
    assert test_only.ids == [20]
    assert test_only.held_out_split == "test"
    assert validation_only.ids == [30]
    assert validation_only.held_out_split == "validation"


def test_ids_are_sorted_then_capped_and_truncation_is_explicit(
    monkeypatch,
):
    monkeypatch.setattr(leakage, "MAX_ID_LIST", 3)
    response, _ = _request(
        monkeypatch,
        pairs=[
            (1, 50, 0.99),
            (40, 1, 0.98),
            (1, 30, 0.97),
            (20, 1, 0.96),
            (1, 10, 0.95),
        ],
        splits={
            1: "train",
            10: "test",
            20: "test",
            30: "test",
            40: "test",
            50: "test",
        },
    )

    assert response.total == 5
    assert response.ids == [10, 20, 30]
    assert response.truncated is True


def test_endpoint_returns_503_when_the_image_index_is_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(leakage, "get_index", lambda: None)

    with TestClient(app) as client:
        response = client.get("/api/stats/leakage/contaminated")

    assert response.status_code == 503
    assert "needs image embeddings" in response.json()["detail"]


@pytest.mark.parametrize(
    "path",
    ["/api/stats/leakage", "/api/stats/leakage/contaminated"],
)
@pytest.mark.parametrize("split", ["train", "typo"])
def test_endpoints_reject_non_held_out_splits(path, split):
    with TestClient(app) as client:
        response = client.get(
            path,
            params={"split": split},
        )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "split"]
