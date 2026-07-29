"""Guards for the two code-level claims in the truth-debt pass (item G).

Both are cases where a comment asserted a property the code did not have.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import leakage, qa_run
from app.main import app
from app.qa import runner


class _FakeIndex:
    """Just enough index to exercise the cache key."""

    def __init__(self, ids, pairs):
        self.ids = np.asarray(ids, dtype=np.int64)
        self._pairs = pairs
        self.calls = 0

    def all_pairs_above(self, _threshold):
        self.calls += 1
        return self._pairs


@pytest.fixture(autouse=True)
def _clean_cache():
    leakage.clear_cache()
    yield
    leakage.clear_cache()


# -- G7: the leakage pair cache ----------------------------------------------

def test_pairs_are_cached_for_the_same_corpus():
    index = _FakeIndex([1, 2, 3], [(1, 2, 0.99)])
    assert leakage._pairs(index) == [(1, 2, 0.99)]
    assert leakage._pairs(index) == [(1, 2, 0.99)]
    assert index.calls == 1, "the expensive scan ran twice for one corpus"


def test_equal_ids_from_different_indexes_do_not_share_pairs():
    """Provider generations share IDs but never an embedding space."""
    qwen = _FakeIndex([1, 2, 3], [(1, 2, 0.99)])
    siglip = _FakeIndex([1, 2, 3], [(2, 3, 0.98)])

    assert leakage._pairs(qwen) == [(1, 2, 0.99)]
    assert leakage._pairs(siglip) == [(2, 3, 0.98)]
    assert qwen.calls == 1
    assert siglip.calls == 1


def test_concurrent_requests_scan_one_index_once():
    """The report and ID hand-off cold-load in parallel in the UI."""
    started = Event()
    release = Event()
    second_scan = Event()

    class _BlockingIndex(_FakeIndex):
        def all_pairs_above(self, threshold):
            self.calls += 1
            if self.calls == 1:
                started.set()
                assert release.wait(timeout=1)
            else:
                second_scan.set()
            return self._pairs

    index = _BlockingIndex([1, 2, 3], [(1, 2, 0.99)])
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(leakage._pairs, index)
        assert started.wait(timeout=1)
        second = pool.submit(leakage._pairs, index)
        assert not second_scan.wait(timeout=0.05)
        release.set()
        assert first.result(timeout=1) == [(1, 2, 0.99)]
        assert second.result(timeout=1) == [(1, 2, 0.99)]

    assert index.calls == 1


def test_admin_reload_clears_the_pair_cache():
    index = _FakeIndex([1, 2, 3], [(1, 2, 0.99)])
    leakage._pairs(index)
    assert leakage._cached_index is index
    assert leakage._cached_pairs == [(1, 2, 0.99)]

    with TestClient(app) as client:
        assert client.post("/api/admin/reload").status_code == 200

    assert leakage._cached_index is None
    assert leakage._cached_pairs is None


# -- G8: run_id is a path segment --------------------------------------------
#
# Exercised at the handler, not over HTTP. Starlette normalizes `/qa/run/../x`
# before routing, so a request-level test gets its 404 from the router and would
# pass with the guard deleted -- it proves the client is well behaved, not that
# the code is. These call the functions with the value they would receive if any
# layer in front ever forwarded a segment unnormalized, which is what the guard
# is for.

TRAVERSALS = ["../../etc", "..", "a/b", "/etc", "./x", "", ".", "x/", "a\\b",
              "run\x00id", "évil"]


@pytest.mark.parametrize("run_id", TRAVERSALS)
def test_get_qa_run_rejects_a_run_id_that_is_not_a_plain_name(run_id):
    with pytest.raises(HTTPException) as exc:
        qa_run.get_qa_run(run_id)
    assert exc.value.status_code == 404


@pytest.mark.parametrize("run_id", TRAVERSALS)
def test_qa_artifact_rejects_a_run_id_that_is_not_a_plain_name(run_id):
    with pytest.raises(HTTPException) as exc:
        qa_run.qa_artifact(run_id, "report.json")
    assert exc.value.status_code == 404


def test_dotdot_is_rejected_though_it_equals_its_own_basename():
    """The specific case a `Path(run_id).name` comparison would let through."""
    assert Path("..").name == "..", "pathlib changed; revisit the guard's rationale"
    with pytest.raises(HTTPException):
        qa_run.qa_artifact("..", "report.json")


def test_a_traversal_cannot_serve_a_file_outside_the_qa_directory(tmp_path,
                                                                  monkeypatch):
    """The exploit itself, not just its status code.

    Every traversal happens to 404 anyway once `load_report` rejects it and the
    probed path does not exist -- so asserting `404` passes with the guard
    deleted and proves nothing. This plants a real, readable file exactly where
    `run_id=".."` would reach it: QA_DIR/../secret.json. Without the guard the
    handler returns it with a 200.
    """
    qa_dir = tmp_path / "qa"
    (qa_dir / "20260101-000000-aaaa").mkdir(parents=True)
    secret = tmp_path / "secret.json"
    secret.write_text('{"not": "an artifact"}')
    monkeypatch.setattr(qa_run.config, "QA_DIR", qa_dir)
    monkeypatch.setattr(runner.config, "QA_DIR", qa_dir)

    # The traversal resolves to a file that genuinely exists and is readable.
    assert (qa_dir / ".." / "secret.json").resolve() == secret.resolve()
    assert secret.is_file()

    with pytest.raises(HTTPException) as exc:
        qa_run.qa_artifact("..", "secret.json")
    assert exc.value.status_code == 404


def test_a_well_formed_but_absent_run_id_is_still_404():
    """The guard must not be the only reason a request 404s."""
    with pytest.raises(HTTPException) as exc:
        qa_run.get_qa_run("20260101-000000-abcd")
    assert exc.value.status_code == 404


def test_the_two_layers_agree_on_what_a_run_id_is():
    """One definition, enforced in both places."""
    for bad in TRAVERSALS:
        assert runner.load_report(bad) is None
    assert runner.RUN_ID_RE.fullmatch("20260726-204715-1a2b")
