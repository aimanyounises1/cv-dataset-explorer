"""Tests for the QA sweep that do not need a browser.

The sweep itself can only be verified by running it, but everything around it —
the registry's shape, the report rendering, the optional-dependency paths, the
single-run lock — is ordinary code and is tested as such. That matters more than
usual here: this machinery is what *reports* whether the app works, so a silent
failure in it would be a silent failure to notice failures.
"""
import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config
from app.agent import blocks
from app.api import qa_run
from app.qa import deck, flows, runner


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A private artifact dir and a clean run manager per test.

    `MANAGER` is a module-level singleton (there is only ever one browser), so a
    test that leaves a run behind would otherwise decide the next test's answer.
    """
    monkeypatch.setattr(config, "QA_DIR", tmp_path / "qa")
    runner.MANAGER._state = None
    runner.MANAGER._thread = None
    yield
    runner.MANAGER._state = None
    runner.MANAGER._thread = None


def sample_report(**over) -> dict:
    report = runner.blank_report("20260725-120000", ["Gallery", "Quality"])
    report.update(
        status="failed", finished_at="2026-07-25T12:02:00+00:00", duration_s=121.5,
        passed=2, total=3, flows_passed=1, flows_total=2, pending=[],
        console_errors=["[Quality] [pageerror] TypeError: x is not a function"],
        network_failures=["[Quality] 500 GET /api/qa/selection"],
        flows=[
            {"name": "Gallery", "status": "pass", "duration_s": 18.2,
             "screenshot": "/media/qa/20260725-120000/gallery.png", "detail": None,
             "checks": [{"name": "browse renders 60 cards", "ok": True, "detail": "60"},
                        {"name": "export links present", "ok": True, "detail": "3"}]},
            {"name": "Quality", "status": "fail", "duration_s": 9.0,
             "screenshot": None, "detail": "TimeoutError: waiting for .dist-bars",
             "checks": [{"name": "histogram rendered", "ok": False, "detail": "0 bars"}]},
        ])
    report.update(over)
    return report


# ----------------------------------------------------------------- the registry

def test_registry_is_populated_and_names_are_unique():
    names = [f.name for f in flows.FLOWS]
    assert len(names) >= 8
    assert len(names) == len(set(names))


def test_registry_covers_every_view_of_the_app():
    # Losing a workflow from the sweep is losing coverage silently, so the groups
    # the app has are asserted by name rather than left to whoever edits flows.py.
    joined = " ".join(f.name.lower() for f in flows.FLOWS)
    for view in ("route", "gallery", "map", "statistic", "quality", "benchmark",
                 "sample", "assistant"):
        assert view in joined, f"no flow covers {view}"


def test_every_flow_takes_a_page_and_a_recorder():
    import inspect

    for f in flows.FLOWS:
        params = list(inspect.signature(f.fn).parameters)
        assert params == ["pg", "ok"], f"{f.name} has signature {params}"
        assert f.budget_s > 0


def test_flow_filter_matches_by_substring_case_insensitively():
    assert [f.name for f in flows.get_flows(["gallery"])] == ["Gallery"]
    assert len(flows.get_flows(["MAP", "quality"])) == 2
    assert flows.get_flows(["nothing-by-this-name"]) == []
    assert len(flows.get_flows()) == len(flows.FLOWS)


def test_duplicate_flow_name_is_rejected_at_import():
    with pytest.raises(ValueError, match="duplicate"):
        flows.flow(flows.FLOWS[0].name)(lambda pg, ok: None)
    # The failed registration must not have been recorded.
    assert len(flows.FLOWS) == len({f.name for f in flows.FLOWS})


def test_flow_urls_follow_config(monkeypatch):
    monkeypatch.setattr(config, "QA_BASE_URL", "http://example.test:99/")
    assert flows.url("/map") == "http://example.test:99/map"


# -------------------------------------------------------------------- rendering

def test_markdown_reports_every_flow_check_and_problem():
    md = deck.render_markdown(sample_report())
    assert "2 / 3 checks passed" in md
    for fragment in ("## Gallery — PASS", "## Quality — FAIL",
                     "browse renders 60 cards", "histogram rendered",
                     "TimeoutError: waiting for .dist-bars",
                     "TypeError: x is not a function",
                     "500 GET /api/qa/selection"):
        assert fragment in md, fragment
    # Screenshots are linked relatively, so the document reads on disk too.
    assert "![Gallery](gallery.png)" in md


def test_markdown_of_a_run_in_progress_says_so():
    md = deck.render_markdown(sample_report(status="running", pending=["Assistant"]))
    assert "Running —" in md
    assert "| Assistant | pending |" in md


def test_healthy_requires_checks_flows_and_a_clean_console():
    clean = sample_report(status="done", passed=3, total=3, console_errors=[],
                          flows=[{"name": "Gallery", "status": "pass", "detail": None,
                                  "duration_s": 1.0, "screenshot": None,
                                  "checks": [{"name": "a", "ok": True, "detail": ""}]}])
    assert runner.healthy(clean) is True
    # A console error is a failure even when every assertion passed: the UI threw.
    assert runner.healthy({**clean, "console_errors": ["boom"]}) is False
    assert runner.healthy({**clean, "passed": 2}) is False
    # A sweep that recorded nothing has not shown that anything works.
    assert runner.healthy({**clean, "total": 0, "passed": 0}) is False


def test_block_payload_is_exactly_a_qablock():
    payload = runner.block_payload(sample_report())
    block = blocks.QABlock(title="Application status",
                           source="Playwright sweep of the running app", **payload)
    assert block.kind == "qa"
    assert block.status == "failed" and block.passed == 2 and block.total == 3
    assert [f.name for f in block.flows] == ["Gallery", "Quality"]
    assert block.flows[0].checks[0]["name"] == "browse renders 60 cards"
    assert block.flows[1].status == "fail"
    # Nothing beyond QABlock's own fields, so **payload can never be a surprise.
    assert set(payload) <= set(blocks.QABlock.model_fields)


# ------------------------------------------------------- optional dependencies

def test_deck_degrades_to_markdown_without_pptx(tmp_path, monkeypatch):
    monkeypatch.setattr(deck, "_pptx_api", lambda: None)
    out = deck.write(sample_report(), tmp_path)
    assert out["deck_file"] is None
    assert "requirements-qa.txt" in out["deck_note"]
    assert (tmp_path / "report.md").exists()
    assert not (tmp_path / "report.pptx").exists()
    # The surviving document has to admit what is missing from it.
    assert "requirements-qa.txt" in (tmp_path / "report.md").read_text()


def test_no_deck_still_writes_the_markdown(tmp_path):
    out = deck.write(sample_report(), tmp_path, make_deck=False)
    assert out["deck_file"] is None and "skipped" in out["deck_note"]
    assert (tmp_path / "report.md").exists()


def test_a_failing_deck_render_does_not_lose_the_report(tmp_path, monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("no such layout")

    monkeypatch.setattr(deck, "render_deck", explode)
    out = deck.write(sample_report(), tmp_path)
    assert out["deck_file"] is None
    assert "Deck render failed" in out["deck_note"] and "no such layout" in out["deck_note"]
    assert (tmp_path / "report.md").exists()


def test_deck_renders_when_pptx_is_installed(tmp_path):
    pytest.importorskip("pptx", reason="python-pptx is an optional QA dependency")
    report = sample_report()
    out = deck.write(report, tmp_path)
    assert out["deck_file"] == "report.pptx" and out["deck_note"] is None
    # Title slide, one per flow, one for the console errors.
    from pptx import Presentation
    assert len(Presentation(str(tmp_path / "report.pptx")).slides) == 4


# ------------------------------------------------------------------- endpoints

@pytest.fixture
def client():
    """The router alone: it is registered in main.py by hand, and these tests
    must not depend on the rest of the app being importable-and-seeded."""
    api = FastAPI()
    api.include_router(qa_run.router, prefix="/api")
    return TestClient(api)


def test_run_is_503_with_setup_instructions_without_playwright(client, monkeypatch):
    monkeypatch.setattr(runner, "playwright_available", lambda: False)
    r = client.post("/api/qa/run")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "requirements-qa.txt" in detail and "channel=" in detail


def test_run_is_503_when_the_frontend_is_not_running(client, monkeypatch):
    monkeypatch.setattr(runner, "playwright_available", lambda: True)
    monkeypatch.setattr(runner, "frontend_reachable", lambda *a, **k: False)
    r = client.post("/api/qa/run")
    assert r.status_code == 503 and "npm run dev" in r.json()["detail"]


def test_no_run_yet_is_a_404_not_an_empty_report(client):
    r = client.get("/api/qa/run")
    assert r.status_code == 404 and "POST /api/qa/run" in r.json()["detail"]
    assert client.get("/api/qa/run/20200101-000000").status_code == 404


@pytest.fixture
def fake_sweep(monkeypatch):
    """A run that blocks until released, so concurrency is testable without a
    browser."""
    entered, release = threading.Event(), threading.Event()

    def fake_run(only=None, *, make_deck=True, state=None, run_id=None, **kw):
        state["heartbeat"] = time.time()
        entered.set()
        release.wait(10)
        state.update(status="done", passed=1, total=1, finished_at="later")
        return state

    monkeypatch.setattr(runner, "playwright_available", lambda: True)
    monkeypatch.setattr(runner, "frontend_reachable", lambda *a, **k: True)
    monkeypatch.setattr(runner, "run", fake_run)
    yield entered, release
    release.set()


def test_a_second_request_gets_the_inflight_run_not_a_second_browser(client, fake_sweep):
    entered, release = fake_sweep
    first = client.post("/api/qa/run")
    assert first.status_code == 202 and first.json()["started"] is True
    assert entered.wait(5)

    second = client.post("/api/qa/run")
    assert second.status_code == 200
    assert second.json()["started"] is False
    assert second.json()["run_id"] == first.json()["run_id"]
    assert "already in flight" in second.json()["note"]

    live = client.get("/api/qa/run").json()
    assert live["run_id"] == first.json()["run_id"] and live["status"] == "running"
    assert "heartbeat" not in live

    release.set()
    for _ in range(100):                          # the thread has to notice
        if client.get("/api/qa/run").json()["status"] == "done":
            break
        time.sleep(0.05)
    assert client.get("/api/qa/run").json()["status"] == "done"
    # With the lock free, a later request starts a new run.
    assert client.post("/api/qa/run").json()["started"] is True


def test_a_stalled_run_is_abandoned_so_the_endpoint_recovers(client, fake_sweep):
    entered, _ = fake_sweep
    first = client.post("/api/qa/run").json()
    assert entered.wait(5)
    # Simulate a browser that hung: the thread is alive but nothing progresses.
    runner.MANAGER._state["heartbeat"] = time.time() - runner.STALL_S - 1
    second = client.post("/api/qa/run").json()
    assert second["started"] is True and second["run_id"] != first["run_id"]


def test_a_run_thread_that_dies_does_not_hold_the_lock(client, monkeypatch):
    monkeypatch.setattr(runner, "playwright_available", lambda: True)
    monkeypatch.setattr(runner, "frontend_reachable", lambda *a, **k: True)

    def boom(*a, **k):
        raise RuntimeError("chrome not found")

    monkeypatch.setattr(runner, "run", boom)
    started = client.post("/api/qa/run").json()
    for _ in range(100):
        if client.get(f"/api/qa/run/{started['run_id']}").json()["status"] != "running":
            break
        time.sleep(0.05)
    failed = client.get(f"/api/qa/run/{started['run_id']}").json()
    assert failed["status"] == "failed" and "chrome not found" in failed["error"]
    assert client.post("/api/qa/run").json()["started"] is True


def test_a_finished_run_is_replayed_from_disk(client, tmp_path):
    out = config.QA_DIR / "20260725-093000"
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(sample_report(run_id="20260725-093000")))
    (out / "gallery.png").write_bytes(b"\x89PNG\r\n\x1a\nnot-really")

    body = client.get("/api/qa/run/20260725-093000").json()
    assert body["run_id"] == "20260725-093000" and body["total"] == 3
    # The latest endpoint falls back to disk when nothing is in memory.
    assert client.get("/api/qa/run").json()["run_id"] == "20260725-093000"

    shot = client.get("/api/qa/artifact/20260725-093000/gallery.png")
    assert shot.status_code == 200 and shot.headers["content-type"] == "image/png"


def test_the_artifact_route_serves_only_a_runs_own_artifacts(client):
    out = config.QA_DIR / "20260725-093000"
    out.mkdir(parents=True)
    (out / "notes.txt").write_text("not an artifact")
    (out / "report.md").write_text("# ok")

    assert client.get("/api/qa/artifact/20260725-093000/report.md").status_code == 200
    # Wrong type, absent file, and a run id that is really a path fragment.
    assert client.get("/api/qa/artifact/20260725-093000/notes.txt").status_code == 404
    assert client.get("/api/qa/artifact/20260725-093000/report.pptx").status_code == 404
    assert client.get("/api/qa/artifact/..%2F..%2Fetc/passwd").status_code == 404
    assert runner.load_report("../../etc") is None


def test_the_flow_listing_describes_the_sweep_without_running_it(client):
    body = client.get("/api/qa/flows").json()
    assert len(body["flows"]) == len(flows.FLOWS)
    assert body["flows"][0]["budget_s"] > 0
    assert body["frontend"] == config.QA_BASE_URL


# ------------------------------------------------------- the sweep, browserless

class _FakePage:
    """Just enough of a Playwright page for the runner's own bookkeeping. The
    flows are faked alongside it: what is under test here is how the runner
    assembles a report, not the assertions inside any particular flow."""

    def __init__(self):
        self.closed = False

    def screenshot(self, path):
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n")

    def close(self):
        self.closed = True


class _FakeContext:
    def __init__(self):
        self.pages = []

    def new_page(self):
        self.pages.append(_FakePage())
        return self.pages[-1]

    def set_default_timeout(self, ms):
        pass

    def set_default_navigation_timeout(self, ms):
        pass

    def on(self, event, callback):
        pass


class _FakeBrowser:
    def __init__(self):
        self.closed = False
        self.context = _FakeContext()

    def new_context(self, **kw):
        return self.context

    def close(self):
        self.closed = True


@pytest.fixture
def fake_browser(monkeypatch):
    browser = _FakeBrowser()

    class _Playwright:
        chromium = types.SimpleNamespace(launch=lambda **kw: browser)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    sync_api = types.ModuleType("playwright.sync_api")
    sync_api.sync_playwright = lambda: _Playwright()
    root = types.ModuleType("playwright")
    root.sync_api = sync_api
    monkeypatch.setitem(sys.modules, "playwright", root)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    return browser


def test_run_assembles_a_report_and_isolates_a_throwing_flow(fake_browser, monkeypatch):
    def alpha(pg, ok):
        ok("first", True, "detail")
        ok("second", True)

    def beta(pg, ok):
        raise RuntimeError("selector never appeared")

    monkeypatch.setattr(runner, "get_flows", lambda only=None: [
        flows.Flow("Alpha", alpha, 30.0), flows.Flow("Beta", beta, 30.0)])

    report = runner.run(make_deck=False, run_id="unit-run")

    # Beta threw, so the sweep reports a failure — and still finished, with a
    # screenshot and a report, which is the whole point of isolating flows.
    assert report["status"] == "failed"
    assert (report["passed"], report["total"]) == (2, 2)
    assert report["flows_passed"] == 1 and report["pending"] == []
    alpha_r, beta_r = report["flows"]
    assert alpha_r["status"] == "pass" and len(alpha_r["checks"]) == 2
    assert beta_r["status"] == "fail"
    assert "selector never appeared" in beta_r["detail"]
    assert alpha_r["screenshot"].endswith("/unit-run/alpha.png")
    assert (config.QA_DIR / "unit-run" / "beta.png").exists()
    on_disk = json.loads((config.QA_DIR / "unit-run" / "report.json").read_text())
    assert on_disk["run_id"] == "unit-run" and on_disk["status"] == "failed"
    # No page and no browser is left open, whatever the flows did.
    assert fake_browser.closed
    assert all(p.closed for p in fake_browser.context.pages)


def test_a_flow_that_outruns_its_budget_is_stopped_at_the_next_check(fake_browser,
                                                                    monkeypatch):
    reached = []

    def slow(pg, ok):
        ok("first", True)
        time.sleep(0.6)
        ok("second", True)          # recorded, then the budget cuts the flow off
        reached.append("third")
        ok("third", True)

    monkeypatch.setattr(runner, "get_flows",
                        lambda only=None: [flows.Flow("Slow", slow, 0.5)])
    report = runner.run(make_deck=False, run_id="budget-run")

    result = report["flows"][0]
    assert result["status"] == "fail" and "budget" in result["detail"]
    # The check that ran long is still reported; the one after it never ran.
    assert [c["name"] for c in result["checks"]] == ["first", "second"]
    assert reached == []


def test_status_is_published_only_after_every_artifact_exists(fake_browser, monkeypatch):
    """Regression: the terminal status was set before the deck and Markdown were
    written, so a consumer polling until status != "running" — which is the only
    completion signal there is — got "done" with deck_url and markdown_url still
    None, and reported a report that did not exist yet."""
    monkeypatch.setattr(runner, "get_flows", lambda only=None: [
        flows.Flow("Alpha", lambda pg, ok: ok("a", True), 30.0)])
    state = runner.blank_report("race-run", ["Alpha"])
    seen = {}
    real_write = deck.write

    def spy(report, out_dir, **kw):
        seen["live"] = state["status"]          # must still be "running"
        seen["document"] = report["status"]     # must already know the verdict
        return real_write(report, out_dir, **kw)

    monkeypatch.setattr(runner.deck_mod, "write", spy)
    report = runner.run(make_deck=False, state=state, run_id="race-run")

    assert seen == {"live": "running", "document": "done"}
    assert report is state, "the caller's dict is the live report"
    assert state["status"] == "done"
    assert state["markdown_url"].endswith("/race-run/report.md")
    assert (config.QA_DIR / "race-run" / "report.md").exists()
    assert (config.QA_DIR / "race-run" / "report.json").exists()


def test_a_filter_that_matches_nothing_does_not_claim_success(fake_browser, monkeypatch):
    report = runner.run(["no-such-flow"], make_deck=False, run_id="empty-run")
    assert report["flows"] == [] and report["total"] == 0
    assert report["status"] == "failed"


# ------------------------------------------------------------------------- CLI

def test_cli_lists_flows_without_a_browser(capsys):
    assert runner.main(["--list"]) == 0
    assert "Gallery" in capsys.readouterr().out


def test_cli_refuses_clearly_when_playwright_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(runner, "playwright_available", lambda: False)
    assert runner.main([]) == 2
    assert "requirements-qa.txt" in capsys.readouterr().err


def test_cli_refuses_clearly_when_the_frontend_is_down(monkeypatch, capsys):
    monkeypatch.setattr(runner, "playwright_available", lambda: True)
    monkeypatch.setattr(runner, "frontend_reachable", lambda *a, **k: False)
    assert runner.main([]) == 2
    assert "npm run dev" in capsys.readouterr().err
