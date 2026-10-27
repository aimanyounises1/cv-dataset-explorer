# Retrieval evaluation

- [x] Make retrieval rankings deterministic and evaluation artifacts reproducible.
  - Done: `PYTHON_BIN=backend/.venv/bin/python .claude/docs-first/verify.sh`.
  - Libraries: NumPy, Pydantic, pytest and the Python standard library.
  - Sources: installed package docstrings and official documentation in EVIDENCE.md.
  - Boundaries: synthetic regression measurements only; no new full-dataset performance claim.
  - Verified: 528 backend tests; Ruff; TypeScript/Vite build; 44 relative links;
    `git diff --check`; docs-first guard clean on every changed file.
  - Limitation: verification reused an existing installed backend environment;
    no fresh model download or full-corpus provider comparison was performed.
