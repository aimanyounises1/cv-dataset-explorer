"""Tools that let the assistant report on the application itself.

"Show me the status of the application" is a different kind of question from
everything else in this platform: the answer is not in the dataset, it is in
whether the software still works. So it is a specialist with its own tools, and
those tools drive a real browser rather than consulting anything.

The awkward part is time. A sweep takes minutes; a chat turn cannot. Rather than
blocking a lane until it is killed by the lane timeout, `run_app_qa` waits up to a
budget that is deliberately shorter than that timeout and then returns the run
*in progress* — a real answer with a run id the user can ask about again. A tool
that returns nothing after four minutes and a tool that returns a partial report
after three look identical to the model; only one of them is useful.
"""
import json
import logging
import time

from langchain_core.tools import tool

from .. import config
from . import blocks

logger = logging.getLogger(__name__)

# Stop waiting well before the lane is killed, so the lane returns a report
# rather than being cut off with nothing to show.
WAIT_MARGIN_S = 45.0
POLL_S = 2.0

SETUP = ("The QA sweep needs Playwright and the system Chrome: "
         "`pip install -r requirements-qa.txt` (or "
         "`uv run --with playwright --with python-pptx ...`), then retry.")


def _wait_budget() -> float:
    return max(30.0, config.AGENT_LANE_TIMEOUT - WAIT_MARGIN_S)


def _as_block(report: dict, title: str) -> blocks.QABlock:
    from ..qa import runner

    payload = runner.block_payload(report)
    status = payload.get("status") or "running"
    if status not in ("running", "done", "failed"):
        status = "failed"
    payload["status"] = status
    note = report.get("error") or report.get("deck_note")
    if status == "running":
        done = len(report.get("flows") or [])
        total = report.get("flows_total") or 0
        current = report.get("current_flow")
        note = (f"Still running: {done} of {total} workflows finished"
                + (f", currently “{current}”" if current else "")
                + f". Ask again for run {report.get('run_id')}.")
    return blocks.QABlock(
        title=title,
        source=f"Real Chrome driven over {report.get('base_url', 'the app')}; "
               f"pass/fail per workflow with a screenshot of each",
        note=note, **payload)


def _unavailable(reason: str) -> str:
    return json.dumps({"error": reason})


@tool
def run_app_qa(only: str = "") -> str:
    """Drive a real browser over every workflow of THIS application, screenshot
    each one, and return a pass/fail status report with a downloadable deck.

    Use for "is the app working", "show me the status of the application", "run
    the tests". Takes minutes. Pass `only` to limit the sweep to workflows whose
    name contains that text (e.g. 'gallery'); leave it empty for everything. If a
    sweep is already running this attaches to it rather than starting a second."""
    try:
        from ..qa import runner
    except ImportError as exc:                            # pragma: no cover
        return _unavailable(f"{SETUP} (import error: {exc})")

    if not runner.playwright_available():
        return _unavailable(SETUP)
    if not runner.frontend_reachable():
        return _unavailable(
            f"The frontend is not reachable at {config.QA_BASE_URL}, so there is "
            f"nothing to drive. Start it with `cd frontend && npm run dev` and retry.")

    flows = [f.strip() for f in only.split(",") if f.strip()] or None
    state, started = runner.MANAGER.start(flows)
    run_id = state["run_id"]
    logger.info("QA sweep %s (%s)", run_id, "started" if started else "already running")

    deadline = time.monotonic() + _wait_budget()
    while state.get("status") == "running" and time.monotonic() < deadline:
        time.sleep(POLL_S)
        state = runner.MANAGER.latest() or state

    title = ("Application status" if state.get("status") == "done"
             else "Application status (in progress)")
    block = _as_block(state, title)
    summary = (f"{block.passed}/{block.total} checks passed across "
               f"{len(block.flows)} workflows")
    if state.get("status") == "running":
        summary = (f"Sweep {run_id} is still running — {len(block.flows)} of "
                   f"{state.get('flows_total', 0)} workflows done so far.")
    elif state.get("status") == "failed":
        summary = f"Sweep {run_id} did not finish: {state.get('error')}"
    failing = [f.name for f in block.flows if f.status == "fail"]
    return json.dumps({
        "blocks": [block.model_dump(mode="json")],
        "summary": summary,
        "status": state.get("status"),
        "run_id": run_id,
        "failing_workflows": failing,
        "console_errors": block.console_errors[:5],
    })


@tool
def app_qa_status() -> str:
    """Read the most recent application QA report without starting a new sweep.
    Prefer this when the user asks about status and a run already happened — it is
    instant, where a fresh sweep takes minutes."""
    try:
        from ..qa import runner
    except ImportError as exc:                            # pragma: no cover
        return _unavailable(f"{SETUP} (import error: {exc})")

    report = runner.latest_report()
    if report is None:
        return json.dumps({
            "error": "No QA sweep has been run yet on this machine.",
            "next": "Call run_app_qa to perform one (it takes a few minutes)."})
    block = _as_block(report, "Application status (last sweep)")
    return json.dumps({
        "blocks": [block.model_dump(mode="json")],
        "summary": f"Last sweep {block.run_id} ({block.status}): "
                   f"{block.passed}/{block.total} checks passed.",
        "finished_at": report.get("finished_at"),
        "healthy": runner.healthy(report) if report.get("status") == "done" else None,
    })


QA_TOOLS = [app_qa_status, run_app_qa]
