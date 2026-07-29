## What and why

<!-- The behaviour that changes, and the problem it solves. Link the issue if there is one. -->

## Verification

Tick only what you actually ran. The first four run in CI on this pull request;
the browser sweep remains local because it needs Chrome and an ingested corpus.

- [ ] `ruff check app tests` (in `backend/`)
- [ ] `pytest` (in `backend/`)
- [ ] `npm run build` (in `frontend/` — this is the `tsc` type-check plus the production build)
- [ ] `python scripts/check_links.py` (repo root)
- [ ] `python ../scripts/ui_smoke.py` (in `backend/`; needs Chrome, both servers, an ingested corpus)
- [ ] Not run, and why:

## If this touches ranking, scoring or evaluation

- [ ] every number in the diff names the protocol that produced it (in-app benchmark, offline harness, which split, which pool)
- [ ] no number was carried across protocols
- [ ] `PROTOCOL_VERSION` bumped if the benchmark definition changed, so cached results cannot be read as comparable
- [ ] a claim that got weaker is stated as weaker rather than reworded

## Risk

- [ ] no generated artifact in the diff (images, `explorer.db`, `*.npy`, model weights, QA output)
- [ ] optional layers still degrade with a message instead of failing
- [ ] existing API contracts, filters, paging depth and export shapes unchanged, or the change is called out above
