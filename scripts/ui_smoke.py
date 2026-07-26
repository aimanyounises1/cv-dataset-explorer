#!/usr/bin/env python3
"""End-to-end UI smoke test: drives real Chrome through every view.

A thin CLI over `app.qa.runner`, which is also what `POST /api/qa/run` and the
assistant's status report use. The flows themselves live in `backend/app/qa/
flows.py` — one definition, so the suite a developer runs and the suite the app
runs can never drift apart.

    # both servers must already be running
    cd backend && .venv/bin/uvicorn app.main:app --port 8000 &
    cd frontend && npm run dev &

    cd backend && uv run --with playwright --with python-pptx \
        --python .venv/bin/python python ../scripts/ui_smoke.py

    ... --no-deck              just the checks, no .pptx/.md render
    ... --flow gallery --flow map     only the named workflows (substring match)
    ... --list                what would run

Playwright is intentionally NOT in requirements.txt: it is a developer tool, not
a runtime dependency, and `uv run --with` keeps it out of the app's environment
(see backend/requirements-qa.txt). It drives the system Chrome via
channel="chrome", so no browser download either.

Exits non-zero on any failed check, any flow that could not finish, or any
console error.
"""
import sys
from pathlib import Path

# Importable from anywhere: running a script puts *its own* directory on the
# path, not the caller's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.qa.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
