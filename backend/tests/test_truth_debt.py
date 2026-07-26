"""Guards for the two code-level claims in the truth-debt pass (item G).

Both are cases where a comment asserted a property the code did not have.
"""
from pathlib import Path

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


def test_the_cache_is_keyed_on_contents_not_object_identity():
    """The property `id(index)` did not have.

    Asserted on the key itself rather than on a returned value: two live indexes
    never share an `id()`, so a "call _pairs twice and compare results" test
    passes under both implementations and proves nothing. The collision the old
    key allowed needs one index to be freed and the next to land at the same
    address, which is real but not something a test can schedule.

    What *is* deterministic is which key the cache used.
    """
    index = _FakeIndex([1, 2, 3], [(1, 2, 0.99)])
    leakage._pairs(index)

    assert list(leakage._cache) == [leakage._cache_key(index)]
    assert id(index) not in leakage._cache, "cache is keyed on the object address"


def test_two_corpora_with_equal_ids_share_the_cached_scan():
    """The flip side: content keying must not re-scan an identical corpus."""
    a = _FakeIndex([1, 2, 3], [(1, 2, 0.99)])
    b = _FakeIndex([1, 2, 3], [(1, 2, 0.99)])
    leakage._pairs(a)
    leakage._pairs(b)
    assert b.calls == 0, "rescanned a corpus whose ids were already cached"


def test_cache_key_ignores_object_identity_and_follows_contents():
    a = _FakeIndex([1, 2, 3], [])
    b = _FakeIndex([1, 2, 3], [])
    assert leakage._cache_key(a) == leakage._cache_key(b)
    assert leakage._cache_key(a) != leakage._cache_key(_FakeIndex([1, 2, 4], []))


def test_admin_reload_clears_the_pair_cache():
    index = _FakeIndex([1, 2, 3], [(1, 2, 0.99)])
    leakage._pairs(index)
    assert leakage._cache, "nothing was cached, so the test proves nothing"

    with TestClient(app) as client:
        assert client.post("/api/admin/reload").status_code == 200

    assert not leakage._cache, "/api/admin/reload left the superseded pairs resident"


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
