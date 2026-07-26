"""Autonomous UI QA: one definition of every workflow, three consumers.

`flows.py` says what "working" means, `runner.py` executes it against real
Chrome, `deck.py` renders the result. The CLI (`scripts/ui_smoke.py`), the HTTP
endpoint (`app/api/qa_run.py`) and the assistant's status report all go through
these — none of them owns a second copy of the assertions::

    from app.qa import runner
    report = runner.run(["gallery"])            # or runner.MANAGER.start()
    block = QABlock(title=..., source=..., **runner.block_payload(report))

Nothing is imported here on purpose. `python -m app.qa.runner` would otherwise
execute runner twice — once as a submodule of this package, once as `__main__` —
and a module-level singleton like `MANAGER` does not survive being duplicated.
Importing Playwright or python-pptx is likewise deferred to the point of use, so
the app starts and reports their absence rather than failing to import.
"""
