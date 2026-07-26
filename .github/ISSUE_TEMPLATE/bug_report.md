---
name: Bug report
about: Something behaves differently from what the README or the docs describe
labels: bug
---

**What happened, and what you expected instead**

**Where**

- [ ] an API response (paste the request URL)
- [ ] a view (paste the full URL — every filter lives in the query string, so the URL is the state)
- [ ] a batch CLI (`app.ingest`, `app.analyze`, `app.train_prism`, `app.enrich`)
- [ ] an optional layer: the assistant, the VLM enrichment, or the QA sweep

**Which layers are installed**

```bash
curl -s localhost:8000/api/health
curl -s localhost:8000/api/stats/overview   # reports embeddings_available
curl -s localhost:8000/api/chat/status      # only if the assistant is involved
```

This decides whether a missing feature is a bug or a documented degradation:
semantic search, the map, caption QA and the benchmark all require artifacts that
`app.ingest` and `app.analyze` produce.

**Environment**

Python and Node versions, OS, and whether inference is running on MPS, CUDA or
CPU.

**Logs**

The uvicorn traceback, or the browser console error.
