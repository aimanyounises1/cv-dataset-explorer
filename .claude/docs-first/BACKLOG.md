# Retrieval evaluation

- [x] Preserve evaluation correctness when hubness availability or latency inputs change.
  - Done: `PYTHON_BIN=backend/.venv/bin/python .claude/docs-first/verify.sh`.
  - Libraries: NumPy, pytest and the Python standard library.
  - Sources: version-resolved installed documentation recorded in EVIDENCE.md.
  - Scope: fingerprint the actual applied penalty before cache lookup; retain retrieval exports for one-caption inputs without claiming a latency measurement.
  - Verified: 531 backend tests; Ruff; TypeScript/Vite build; 44 relative links; docs-first gate and all added-diff guard checks.
  - Limitations: existing backend environment reused read-only; no new model inference or full-corpus comparison.

- [x] Make retrieval rankings deterministic and evaluation artifacts reproducible.
  - Done: `PYTHON_BIN=backend/.venv/bin/python .claude/docs-first/verify.sh`.
  - Libraries: NumPy, Pydantic, pytest and the Python standard library.
  - Sources: installed package docstrings and official documentation in EVIDENCE.md.
  - Boundaries: synthetic regression measurements only; no new full-dataset performance claim.
  - Verified: 528 backend tests; Ruff; TypeScript/Vite build; 44 relative links;
    `git diff --check`; docs-first guard clean on every changed file.
  - Limitation: verification reused an existing installed backend environment;
    no fresh model download or full-corpus provider comparison was performed.
