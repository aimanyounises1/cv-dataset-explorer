# 0003: Separate ingest-time analysis from request-time search

## Status

Accepted

## Context

A dataset exploration tool needs two different kinds of computation:
heavyweight one-off analysis (embeddings, projections, quality signals,
attributes) and cheap per-request work (answering a search or filter
query). Computing both in the same code path, on demand, would make every
search request's latency depend on whichever analysis happened to be
missing or stale.

## Decision

Ingest is a distinct phase (`python -m app.ingest`) that computes and
persists every embedding, projection, and quality signal once. Normal
search requests only read from SQLite and, at most, encode the incoming
query; they never recompute dataset-wide analysis inline.

## Consequences

Request latency stays predictable because it is bounded by a database read
and a single query encoding, not by whatever analysis a request happens to
trigger. It also makes the degraded-mode story simple: if an artifact is
missing, the fix is always "run ingest again," not a hidden recomputation
path that a request might trigger unexpectedly.

The trade-off is that ingest must be re-run deliberately after a change to
an embedding model, an analysis definition, or the source data, and the
application has to detect and report a missing artifact rather than paper
over it. That reporting is itself covered by a separate decision: see
[0004](0004-optional-capabilities-degrade-instead-of-failing.md).
