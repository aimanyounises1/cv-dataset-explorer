"""Drives the flow registry through real Chrome and writes a report.

    # both servers must already be running
    cd backend && .venv/bin/uvicorn app.main:app --port 8000 &
    cd frontend && npm run dev &

    cd backend && uv run --with playwright --with python-pptx \
        --python .venv/bin/python python -m app.qa.runner

Playwright and python-pptx are deliberately absent from requirements.txt: they
are developer tools, not runtime dependencies, and `uv run --with` keeps them out
of the app's environment (see requirements-qa.txt). Chrome is driven through
`channel="chrome"`, so there is no browser download either.

Three properties matter more than the checks themselves:

* **Isolation.** Each flow gets its own page and its own try/except, so a flow
  that throws costs one slide, not the sweep. A run always produces a report.
* **Boundedness.** Every Playwright wait is bounded by a default timeout, and
  each flow carries a wall-clock budget that stops it at the next check. A sweep
  is minutes long; it must not become unbounded because one selector never
  appears.
* **A live report.** `run()` mutates the report dict as it goes, so the HTTP
  layer can serve progress for a run that is still going without a second
  representation of the same state.

`status` describes the *application*, not the job: "done" means the sweep
finished and everything passed, "failed" means it finished with failures (or the
sweep itself broke), "running" means in progress. The flow list carries the
detail either way.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse
from uuid import uuid4

from .. import config
from . import deck as deck_mod
from .flows import Flow, get_flows

# Bounds every Playwright wait that does not name its own timeout. Generous
# because a cold Vite dev server compiles a route on first visit.
DEFAULT_TIMEOUT_MS = 25_000
VIEWPORT = {"width": 1600, "height": 1000}

# A run with no progress for this long is presumed wedged (a hung browser, a
# system dialog, a dead dev server). The manager then abandons it rather than
# letting one stuck run block every later request forever.
STALL_S = 300.0

SETUP_HINT = (
    "Playwright is not installed. It is a developer tool, so it is not in "
    "requirements.txt. Install it with `pip install -r backend/requirements-qa.txt`, "
    "or run the sweep without installing anything: `cd backend && uv run "
    "--with playwright --with python-pptx --python .venv/bin/python "
    "python -m app.qa.runner`. Chrome is driven via channel=\"chrome\", so no "
    "browser download is needed."
)


class FlowBudgetExceeded(Exception):
    """Raised at a check boundary once a flow has outrun its budget."""


def playwright_available() -> bool:
    return importlib.util.find_spec("playwright") is not None


def frontend_reachable(url: Optional[str] = None, timeout: float = 2.0) -> bool:
    """Cheap TCP probe of the dev server.

    Worth doing before launching a browser: without it, "the frontend is not
    running" arrives as seven identical navigation timeouts three minutes later.
    """
    parsed = urlparse(url or config.QA_BASE_URL)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname or "localhost", port), timeout):
            return True
    except OSError:
        return False


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "flow"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    """A sortable id with a random tail.

    The timestamp is what makes `latest_report` a lexicographic sort, but second
    resolution alone is not unique: two runs a moment apart would share an id and
    then overwrite each other's screenshots in the same directory.
    """
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:4]}"


def artifact_url(run_id: str, filename: str) -> str:
    return f"{config.QA_URL_PREFIX.rstrip('/')}/{run_id}/{filename}"


def blank_report(run_id: str, flow_names: list[str]) -> dict:
    """A report that is already answerable before the browser has started."""
    return {
        "run_id": run_id,
        "status": "running",
        "started_at": _now(),
        "finished_at": None,
        "duration_s": None,
        "flows": [],
        "pending": flow_names,
        "current_flow": None,
        "passed": 0,
        "total": 0,
        "flows_passed": 0,
        "flows_total": len(flow_names),
        "console_errors": [],
        "network_failures": [],
        # Errors a degradation flow caused on purpose. Kept for the record and
        # out of the health verdict — see `induces_errors` on Flow.
        "induced_errors": [],
        "deck_url": None,
        "markdown_url": None,
        "report_url": None,
        "deck_note": None,
        "base_url": config.QA_BASE_URL,
        "api_url": config.QA_API_URL,
        "media_dir": None,
        "error": None,
        "heartbeat": time.time(),
    }


# QABlock's own field names, so the assistant can hand a report straight to the
# render-block contract: blocks.QABlock(title=..., source=..., **block_payload(r)).
BLOCK_FIELDS = ("status", "run_id", "flows", "passed", "total", "console_errors",
                "deck_url", "markdown_url", "started_at", "finished_at")
FLOW_FIELDS = ("name", "status", "checks", "screenshot", "detail", "duration_s")


def block_payload(report: dict) -> dict:
    """The subset of a report that maps exactly onto `agent.blocks.QABlock`.

    `passed`/`total` count *checks*, not flows — "42/44 checks passed" is the
    headline a status report needs; per-flow rollups are in `flows`.
    """
    out = {k: report.get(k) for k in BLOCK_FIELDS}
    out["flows"] = [{k: f.get(k) for k in FLOW_FIELDS} for f in report.get("flows", [])]
    return out


# ------------------------------------------------------------------ one flow

def _run_flow(ctx, f: Flow, out_dir: Path, run_id: str,
              on_check: Optional[Callable[[str, dict], None]],
              beat: Callable[[], None]) -> dict:
    """Run one flow on its own page. Never raises: a flow's failure is data."""
    checks: list[dict] = []
    deadline = time.monotonic() + f.budget_s
    started = time.monotonic()

    def ok(name: str, cond, detail: str = "") -> bool:
        result = bool(cond)
        checks.append({"name": name, "ok": result, "detail": str(detail)})
        if on_check:
            on_check(f.name, checks[-1])
        beat()
        # Checked *after* recording, so the check that ran long is still reported.
        if time.monotonic() > deadline:
            raise FlowBudgetExceeded(
                f"exceeded its {f.budget_s:.0f}s budget after {len(checks)} checks")
        return result

    detail: Optional[str] = None
    page = ctx.new_page()
    try:
        f.fn(page, ok)
        if not checks:
            status, detail = "skip", "flow recorded no checks"
        else:
            status = "pass" if all(c["ok"] for c in checks) else "fail"
    except FlowBudgetExceeded as e:
        status, detail = "fail", str(e)
    except Exception as e:                            # noqa: BLE001 — report anything
        status = "fail"
        detail = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"

    # Screenshot last: on a failure the final state is the evidence.
    shot: Optional[str] = None
    try:
        name = f"{slug(f.name)}.png"
        page.screenshot(path=str(out_dir / name))
        shot = artifact_url(run_id, name)
    except Exception as e:                            # noqa: BLE001
        detail = f"{detail + '; ' if detail else ''}screenshot failed: {e}"[:400]
    try:
        page.close()
    except Exception:                                 # noqa: BLE001 — already gone
        pass

    return {"name": f.name, "status": status, "checks": checks, "screenshot": shot,
            "detail": detail, "duration_s": round(time.monotonic() - started, 2)}


# ------------------------------------------------------------------ the sweep

def run(only: Optional[list[str]] = None, *, make_deck: bool = True,
        on_check: Optional[Callable[[str, dict], None]] = None,
        on_flow: Optional[Callable[[dict], None]] = None,
        state: Optional[dict] = None, run_id: Optional[str] = None,
        headed: bool = False, slow_mo: int = 0) -> dict:
    """Execute the registry and return the report.

    `state`, if given, is used as the report dict itself, so a caller holding a
    reference watches progress arrive rather than polling a copy.

    `headed` opens a real Chrome window instead of running invisibly, and
    `slow_mo` pauses that many milliseconds between actions. Neither changes what
    is asserted — they exist because a suite you cannot watch is a suite you have
    to take on trust, and watching it is how you notice the thing no assertion
    covers. The endpoint always runs headless: a server must not try to open a
    window.
    """
    from playwright.sync_api import sync_playwright  # optional dep, imported late

    flows = get_flows(only)
    run_id = run_id or new_run_id()
    out_dir = Path(config.QA_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report = state if state is not None else blank_report(run_id, [f.name for f in flows])
    report.update(run_id=run_id, status="running", flows_total=len(flows),
                  pending=[f.name for f in flows], media_dir=str(out_dir),
                  base_url=config.QA_BASE_URL, api_url=config.QA_API_URL)
    report.setdefault("started_at", _now())
    t0 = time.monotonic()

    def beat() -> None:
        report["heartbeat"] = time.time()

    # `induced` is set while a flow that deliberately breaks something is running.
    # Its errors are the point of the test, so they are filed under
    # `induced_errors` for the record rather than counted against the app: a
    # degradation test proving the UI handles a 500 must not itself be the reason
    # the sweep declares the application unhealthy.
    current = {"flow": "startup", "induced": False}
    report.setdefault("induced_errors", [])

    def _file(key: str, line: str) -> None:
        report["induced_errors" if current["induced"] else key].append(line)

    def note_console(m) -> None:
        if m.type == "error":
            _file("console_errors", f"[{current['flow']}] {m.text[:200]}")

    def note_response(r) -> None:
        if r.status >= 400:
            where = r.url.replace(config.QA_BASE_URL, "")[:110]
            _file("network_failures",
                  f"[{current['flow']}] {r.status} {r.request.method} {where}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome", headless=not headed,
            slow_mo=slow_mo if headed else 0,
            args=["--window-position=30,30",
                  f"--window-size={VIEWPORT['width']},{VIEWPORT['height'] + 120}"]
            if headed else [])
        try:
            ctx = browser.new_context(viewport=VIEWPORT)
            ctx.set_default_timeout(DEFAULT_TIMEOUT_MS)
            ctx.set_default_navigation_timeout(45_000)
            ctx.on("console", note_console)
            ctx.on("response", note_response)
            ctx.on("weberror", lambda e:
                   _file("console_errors",
                         f"[{current['flow']}] [pageerror] {str(e.error)[:200]}"))

            for f in flows:
                current["flow"] = f.name
                current["induced"] = f.induces_errors
                report["current_flow"] = f.name
                beat()
                result = _run_flow(ctx, f, out_dir, run_id, on_check, beat)
                report["flows"].append(result)
                report["pending"] = [n for n in report["pending"] if n != f.name]
                report["passed"] = sum(1 for fl in report["flows"]
                                       for c in fl["checks"] if c["ok"])
                report["total"] = sum(len(fl["checks"]) for fl in report["flows"])
                report["flows_passed"] = sum(1 for fl in report["flows"]
                                             if fl["status"] == "pass")
                beat()
                if on_flow:
                    on_flow(result)
        finally:
            current["flow"] = "teardown"
            current["induced"] = False
            browser.close()

    report["current_flow"] = None
    report["finished_at"] = _now()
    report["duration_s"] = round(time.monotonic() - t0, 2)
    # De-duplicated in order: one broken image logs the same error per card.
    report["console_errors"] = list(dict.fromkeys(report["console_errors"]))
    report["network_failures"] = list(dict.fromkeys(report["network_failures"]))
    report["induced_errors"] = list(dict.fromkeys(report["induced_errors"]))

    # The terminal status is computed here but published *last*, once every
    # artifact the report points at exists. A consumer polls until the status
    # stops being "running" — that is the only signal it has — so flipping it
    # before the .pptx finishes rendering hands it "done" with `deck_url: None`
    # and it reports a deck that does not exist yet. The documents still need the
    # final verdict while they are being written, hence the override rather than
    # an early assignment.
    final_status = "done" if healthy(report) else "failed"
    finished = {**report, "status": final_status}

    files = deck_mod.write(finished, out_dir, make_deck=make_deck)
    report["deck_note"] = files["deck_note"]
    report["markdown_url"] = artifact_url(run_id, files["markdown_file"])
    report["deck_url"] = (artifact_url(run_id, files["deck_file"])
                          if files["deck_file"] else None)
    report["report_url"] = artifact_url(run_id, "report.json")
    (out_dir / "report.json").write_text(
        json.dumps({**report, "status": final_status}, indent=2))
    report["status"] = final_status
    beat()
    return report


def healthy(report: dict) -> bool:
    """Whether the application passed: every check, every flow, no console error."""
    return (report["total"] > 0
            and report["passed"] == report["total"]
            and not report["console_errors"]
            and all(f["status"] != "fail" for f in report["flows"]))


# --------------------------------------------------------------- run manager

class RunManager:
    """One sweep at a time, in a background thread.

    A second request must not start a competing browser: two Chromes driving the
    same dev server interleave their navigations and both report nonsense. So a
    request that arrives during a run gets the in-flight run instead — which is
    also what the caller wanted ("show me the status"), one run later.

    The exception is a wedged run. A browser can hang in a way no timeout inside
    the sweep can see, and a lock with no way out would then break the endpoint
    until the server restarts; a run with a stale heartbeat is abandoned so the
    next request can start a real one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None

    def latest(self) -> Optional[dict]:
        with self._lock:
            self._reap()
            return self._state

    def start(self, only: Optional[list[str]] = None, *,
              make_deck: bool = True) -> tuple[dict, bool]:
        """Return (state, started). `started` is False when an in-flight run was
        handed back instead of a new one."""
        with self._lock:
            self._reap()
            if self._state is not None and self._state["status"] == "running":
                return self._state, False
            flows = get_flows(only)
            state = blank_report(new_run_id(), [f.name for f in flows])
            self._state = state
            self._thread = threading.Thread(
                target=self._work, args=(state, only, make_deck),
                name=f"qa-run-{state['run_id']}", daemon=True)
            self._thread.start()
            return state, True

    def _work(self, state: dict, only, make_deck: bool) -> None:
        try:
            run(only, make_deck=make_deck, state=state, run_id=state["run_id"])
        except Exception as e:                        # noqa: BLE001 — the report says why
            state["status"] = "failed"
            state["error"] = f"{type(e).__name__}: {e}"
            state["finished_at"] = _now()
            state["heartbeat"] = time.time()

    def _reap(self) -> None:
        """Abandon a run whose thread died silently or stopped making progress."""
        st = self._state
        if not st or st["status"] != "running":
            return
        alive = self._thread is not None and self._thread.is_alive()
        stalled = time.time() - st.get("heartbeat", 0) > STALL_S
        if not alive or stalled:
            st["status"] = "failed"
            st["finished_at"] = _now()
            st["error"] = ("abandoned: no progress for "
                           f"{int(time.time() - st.get('heartbeat', 0))}s "
                           "(hung browser?)" if stalled else
                           "abandoned: the run thread died without reporting")


MANAGER = RunManager()


def load_report(run_id: str) -> Optional[dict]:
    """A finished run's report from disk, so history outlives the process."""
    if not re.fullmatch(r"[0-9A-Za-z_-]{1,64}", run_id):    # never a path fragment
        return None
    path = Path(config.QA_DIR) / run_id / "report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def latest_report() -> Optional[dict]:
    """The in-memory run if there is one, else the newest report on disk."""
    live = MANAGER.latest()
    if live is not None:
        return live
    runs = sorted(p.parent.name for p in Path(config.QA_DIR).glob("*/report.json"))
    return load_report(runs[-1]) if runs else None


# ---------------------------------------------------------------------- CLI

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Drive every application workflow through real Chrome.")
    ap.add_argument("--flow", action="append", metavar="NAME",
                    help="only flows whose name contains NAME (repeatable)")
    ap.add_argument("--no-deck", action="store_true",
                    help="skip the .pptx/.md render, just run the checks")
    ap.add_argument("--list", action="store_true", help="list the flows and exit")
    ap.add_argument("--headed", action="store_true",
                    help="drive a VISIBLE Chrome window instead of headless")
    ap.add_argument("--slow-mo", type=int, default=350, metavar="MS",
                    help="ms to pause between actions when --headed (default 350)")
    args = ap.parse_args(argv)

    if args.list:
        for f in get_flows(args.flow):
            print(f"  {f.name}  (budget {f.budget_s:.0f}s)")
        return 0
    if not playwright_available():
        print(SETUP_HINT, file=sys.stderr)
        return 2
    if not frontend_reachable():
        print(f"Frontend not reachable at {config.QA_BASE_URL} — start it with "
              f"`cd frontend && npm run dev`.", file=sys.stderr)
        return 2

    seen: set[str] = set()

    def on_check(flow_name: str, check: dict) -> None:
        if flow_name not in seen:
            seen.add(flow_name)
            print(f"\n== {flow_name.lower()} ==")
        detail = f" — {check['detail']}" if check["detail"] else ""
        print(f"  {'PASS' if check['ok'] else 'FAIL'}  {check['name']}{detail}")

    def on_flow(result: dict) -> None:
        if result["detail"]:
            print(f"  !!    {result['name']}: {result['detail']}")

    report = run(args.flow, make_deck=not args.no_deck,
                 on_check=on_check, on_flow=on_flow,
                 headed=args.headed, slow_mo=args.slow_mo)

    for heading, lines in (("console errors", report["console_errors"]),
                           ("failed network requests", report["network_failures"])):
        print(f"\n== {heading} ==")
        for line in lines or ["none"]:
            print("  " + line)

    # Printed apart from the two lists above and after them, so a reader never
    # mistakes a deliberately-injected 500 for a real one. These do not affect
    # the exit code.
    if report.get("induced_errors"):
        print("\n== induced on purpose (degradation tests; not failures) ==")
        for line in report["induced_errors"]:
            print("  " + line)

    print(f"\n{report['passed']}/{report['total']} checks passed "
          f"in {report['duration_s']:.0f}s")
    failed = [c["name"] for fl in report["flows"] for c in fl["checks"] if not c["ok"]]
    if failed:
        print("FAILED: " + ", ".join(failed))
    broken = [f["name"] for f in report["flows"] if f["status"] == "fail" and f["detail"]]
    if broken:
        print("FLOWS THAT DID NOT FINISH: " + ", ".join(broken))
    if report["deck_note"]:
        print(report["deck_note"])
    print(f"artifacts: {report['media_dir']}")
    return 0 if report["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
