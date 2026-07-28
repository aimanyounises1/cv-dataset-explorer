"""QA run selection: a failed or empty run must never shadow a successful
report. Reproduces the exact pair an independent audit caught in the wild —
20260728-084318-63d0 (status failed, 0/0, no flows) sorting after
20260728-080320-0947 (done, 94/94, 16 flows) and being served as "the latest
report".

    cd backend && pytest tests/test_qa_selection.py
"""
import json
from pathlib import Path

import pytest

from app import config
from app.qa import runner

GOOD = "20260728-080320-0947"
BAD = "20260728-084318-63d0"


@pytest.fixture()
def two_runs():
    qa = Path(config.QA_DIR)
    (qa / GOOD).mkdir(parents=True, exist_ok=True)
    (qa / BAD).mkdir(parents=True, exist_ok=True)
    (qa / GOOD / "report.json").write_text(json.dumps({
        "run_id": GOOD, "status": "done", "passed": 94, "total": 94,
        "flows": [{"name": f"f{i}", "status": "pass"} for i in range(16)]}))
    (qa / BAD / "report.json").write_text(json.dumps({
        "run_id": BAD, "status": "failed", "passed": 0, "total": 0,
        "flows": []}))
    yield
    for rid in (GOOD, BAD):
        (qa / rid / "report.json").unlink(missing_ok=True)
        (qa / rid).rmdir()


def test_failed_empty_run_cannot_shadow_a_successful_report(two_runs):
    rep = runner.latest_report()
    assert rep is not None
    assert rep["run_id"] == GOOD, (
        "the newest COMPLETE run is the latest report, not the newest attempt")
    assert rep["passed"] == 94 and rep["total"] == 94


def test_failed_run_stays_inspectable_and_clearly_failed(two_runs):
    bad = runner.load_report(BAD)
    assert bad is not None
    assert bad["status"] == "failed" and bad["total"] == 0


def test_with_only_a_failed_run_it_is_returned_honestly(two_runs):
    qa = Path(config.QA_DIR)
    (qa / GOOD / "report.json").unlink()
    rep = runner.latest_report()
    assert rep is not None and rep["run_id"] == BAD
    assert rep["status"] == "failed"  # served as what it is, never dressed up
