# Architecture Decision Records

This directory records the reasoning behind choices that are easy to
second-guess in isolation but make sense against the constraints described in
[the main README](../../README.md#design-choices). Each record captures the
context at the time, the decision, and what it costs or unlocks, not just the
conclusion.

## Index

| ADR | Decision |
| --- | --- |
| [0001](0001-sqlite-and-exact-search-over-a-vector-database.md) | SQLite and exact NumPy cosine search instead of a vector database |
| [0002](0002-siglip2-as-the-default-retrieval-model.md) | SigLIP 2 as the default retrieval model, Qwen3-VL as an optional provider |
| [0003](0003-separate-ingest-time-analysis-from-request-time-search.md) | Ingest-time analysis is separated from request-time search |
| [0004](0004-optional-capabilities-degrade-instead-of-failing.md) | Optional capabilities degrade instead of failing the application |
| [0005](0005-human-review-as-the-write-boundary.md) | Human review is the boundary between a model suggestion and a saved change |

## Format

Each record uses four sections: Status, Context, Decision, and Consequences.
Status is almost always Accepted here, since this is a single-maintainer
project without a formal proposal stage. The value of a record is in its
Context and Consequences, not in an approval trail.

## When to add one

Add a record when a choice trades away something a reviewer would otherwise
expect, such as a vector database, a hosted model, or an automatic write, and
the trade-off is not obvious from the code alone. Skip it when the code
already makes the reasoning self-evident.
