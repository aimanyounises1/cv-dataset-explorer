# 0004: Optional capabilities degrade instead of failing

## Status

Accepted

## Context

Several features depend on assets a reviewer may not have: local vision
models served through Ollama, Grounding DINO and SAM 2.1 weights, or the
LangGraph assistant's own dependencies. A tool that hard-fails when one of
these is absent would make the whole application unusable for anyone who
only wants the core gallery, search, and audit workflows.

## Decision

Every optional feature probes its own availability and returns a normal
200 response with a setup reason when its dependency is missing, instead
of raising an error that reaches the user as a broken page. The core
workflows (browse, search, inspect, audit, curate, export) never depend on
any optional model.

## Consequences

A reviewer can clone the repository, run the standard ingest, and use most
of the application without installing Ollama or downloading detector
weights. Adding a new optional capability means adding its availability
probe at the same time, not as an afterthought, since the UI relies on
that probe to explain why a feature is unavailable rather than showing a
stack trace.

The cost is that every optional integration point carries a small amount
of extra branching (an availability check plus a message) compared to a
version that simply assumes the dependency is present. That cost is
accepted deliberately, in the same spirit as the write boundary in
[0005](0005-human-review-as-the-write-boundary.md): a missing capability
should be visible and explained, not silently absent or fatally broken.
