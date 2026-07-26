# Contributing

This is a home-assignment repository, so the most useful contribution is usually
a review comment. If you do change something, this is what the change has to
survive and what the conventions are -- all of it commands that exist here, not
generic advice.

## Enough setup to run the checks

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.ingest --limit 300 --skip-embeddings   # small corpus, no model download
cd ../frontend && npm install
```

The test suite does not need that corpus -- it builds its own temporary database
and plants synthetic embeddings -- but the app does if you want to look at it.

## The checks

| Command | Working directory | What it catches |
| --- | --- | --- |
| `ruff check app tests` | `backend/` | lint and import order (line length 100, `E501` ignored) |
| `pytest` | `backend/` | API contracts, degraded modes, ranking and paging invariants, agent graph |
| `npm run build` | `frontend/` | the strict `tsc` type-check plus the production build |
| `python scripts/check_links.py` | repo root | a relative Markdown link that points nowhere |
| `python scripts/capabilities.py --check` | repo root | `docs/CAPABILITIES.md` drifting from the running system (needs the API up) |
| `python ../scripts/ui_smoke.py` | `backend/` | a view that renders empty, a control that stopped filtering, a console error (needs Chrome, both servers, an ingested corpus) |

The first four run in CI on every pull request. The last two cannot run there,
and [docs/TESTING.md](docs/TESTING.md) says why, along with everything else CI
does not cover.

## Conventions that are load-bearing

- **Filters belong inside the ranking.** Never filter an already-limited page:
  `build_filters` composes the `WHERE`, `filtered_id_set` becomes the candidate
  mask, and the lexical path splices the same clause in before `LIMIT`.
- **Name the basis of every score.** `cosine`, `cosine_adj`, `rrf` and `prism_ll`
  live on different scales. A response that publishes a number must publish what
  produced it, and a mode that has no number must publish none.
- **Optional layers degrade, they do not fail.** A missing artifact returns 200
  with a message naming the command that would produce it. Adding a capability
  means adding its availability probe too.
- **Never move a measured number without re-measuring it.** Every figure in the
  docs carries the protocol that produced it; the four protocols are separated in
  the README. If a change alters the benchmark definition, bump
  `PROTOCOL_VERSION` in `app/api/eval.py` so cached results cannot be read as
  comparable. If a claim gets weaker, say it got weaker.
- **Nothing generated is committed.** Images, `explorer.db`, `*.npy`, model
  weights, QA runs and reports all live under `backend/data/`, which is
  gitignored.
- **Requirement ids in test docstrings** (`FR-SE-U1`, `FR-EV-2`, "Criterion 2")
  refer to an internal requirements document that is deliberately not published
  (see `.gitignore`). Treat the docstring itself as the specification: each one
  states the failure it prevents.
- **Regenerate, do not hand-edit, generated files.** `docs/CAPABILITIES.md` comes
  from `python scripts/capabilities.py`; the screenshots come from
  `python scripts/screenshots.py`.

## Commits and pull requests

Commit subjects state intent in the present tense and stay under ~70 characters;
the body explains why, and what was measured. Pull requests use
[the checklist](.github/pull_request_template.md), which asks you to tick only
the checks you actually ran and to list the ones you could not.
