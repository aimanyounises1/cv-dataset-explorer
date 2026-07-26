"""Endpoints for the autonomous UI sweep.

    POST /api/qa/run            start a run, return immediately with its id
    GET  /api/qa/run            the current or most recent run
    GET  /api/qa/run/{run_id}   one run, live from memory or replayed from disk
    GET  /api/qa/artifact/...   screenshots, deck and report of a run

A sweep drives a real browser for minutes, so the POST cannot block a request
thread for its duration and cannot be allowed to run twice at once: two Chromes
against one dev server interleave their navigations and both report nonsense. The
POST therefore starts a background thread and a second POST gets the in-flight
run handed back — the caller asked for the application's status, and a run that
is already producing it answers that.

Playwright is an optional developer dependency, so its absence is a 503 with the
exact command to fix it, the same way the assistant reports a missing Ollama and
the benchmark reports missing embeddings.
"""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse

from .. import config
from ..qa import runner

router = APIRouter()

# Served read-only, so the set is closed rather than "whatever is in the folder".
ARTIFACT_TYPES = {".png": "image/png", ".pptx": "application/vnd.openxmlformats-"
                  "officedocument.presentationml.presentation",
                  ".md": "text/markdown; charset=utf-8",
                  ".json": "application/json"}


def _public(state: dict) -> dict:
    """The report minus bookkeeping the caller has no use for."""
    return {k: v for k, v in state.items() if k != "heartbeat"}


@router.post("/qa/run")
def start_qa_run(
    response: Response,
    flow: Optional[str] = Query(None, description="Only flows whose name contains this"),
    deck: bool = Query(True, description="Also render the .pptx deck"),
):
    """Start a sweep. Returns at once: poll GET /api/qa/run/{run_id} for progress."""
    if not runner.playwright_available():
        raise HTTPException(503, runner.SETUP_HINT)
    if not runner.frontend_reachable():
        raise HTTPException(
            503, f"The frontend is not reachable at {config.QA_BASE_URL}. Start it "
                 f"with `cd frontend && npm run dev` (or point CVDE_QA_BASE_URL at "
                 f"a running instance) — a UI sweep has nothing to drive without it.")
    state, started = runner.MANAGER.start([flow] if flow else None, make_deck=deck)
    body = _public(state)
    # 202 for "I started one", 200 for "here is the one already running".
    response.status_code = 202 if started else 200
    body["started"] = started
    if not started:
        body["note"] = ("A run was already in flight; returning it rather than "
                        "starting a competing browser.")
    return body


@router.get("/qa/run")
def latest_qa_run():
    """The live run, or the most recent finished one from disk."""
    report = runner.latest_report()
    if report is None:
        raise HTTPException(404, "No QA run yet — POST /api/qa/run to start one.")
    return _public(report)


@router.get("/qa/run/{run_id}")
def get_qa_run(run_id: str):
    live = runner.MANAGER.latest()
    if live is not None and live["run_id"] == run_id:
        return _public(live)
    report = runner.load_report(run_id)
    if report is None:
        raise HTTPException(404, f"No QA run {run_id!r}.")
    return _public(report)


@router.get("/qa/artifact/{run_id}/{name}")
def qa_artifact(run_id: str, name: str):
    """A run's screenshot, deck, Markdown or JSON.

    The reports address artifacts by URL prefix (CVDE_QA_URL_PREFIX, default
    /media/qa), which needs a StaticFiles mount; this route serves the same files
    from the router itself, so artifacts are reachable either way.
    """
    if runner.load_report(run_id) is None and not (Path(config.QA_DIR) / run_id).is_dir():
        raise HTTPException(404, f"No QA run {run_id!r}.")
    # Reject anything that is not a plain filename: this path is user input.
    if name != Path(name).name or Path(name).suffix not in ARTIFACT_TYPES:
        raise HTTPException(404, "Not a QA artifact.")
    path = Path(config.QA_DIR) / run_id / name
    if not path.is_file():
        raise HTTPException(404, f"No artifact {name!r} in run {run_id!r}.")
    return FileResponse(path, media_type=ARTIFACT_TYPES[Path(name).suffix])


@router.get("/qa/flows")
def list_qa_flows():
    """What a sweep would cover, without running one."""
    from ..qa.flows import FLOWS

    return {"available": runner.playwright_available(),
            "frontend": config.QA_BASE_URL,
            "flows": [{"name": f.name, "budget_s": f.budget_s,
                       "doc": (f.fn.__doc__ or "").strip().split("\n\n")[0]}
                      for f in FLOWS]}
