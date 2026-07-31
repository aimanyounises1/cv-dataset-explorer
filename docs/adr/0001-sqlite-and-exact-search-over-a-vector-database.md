# 0001: SQLite and exact search instead of a vector database

## Status

Accepted

## Context

Semantic and hybrid search need a nearest-neighbour lookup over image and
text embeddings. The default instinct for a "search" feature is often an
external vector database or an approximate nearest-neighbour (ANN) index,
and adding one is a small amount of code against most client libraries.

The corpus here is fixed at 8,000 Flickr8k images. The application also
runs on a single developer machine, with no hosted service and no other
process that would justify a long-running database server.

## Decision

Embeddings and metadata are stored in SQLite, and ranking is computed with
exact NumPy cosine similarity over the full embedding matrix rather than an
approximate index. No external vector database, hosted search service, or
ANN library is part of the stack.

## Consequences

Exact search removes an entire class of tuning problems that ANN indexes
introduce, such as recall and speed trade-offs, index rebuild schedules,
and a second process to operate and back up. A query over 8,000 vectors is
fast enough on CPU that the exactness costs nothing observable.

This does not scale unbounded. A dataset one or two orders of magnitude
larger would make a full scan noticeably slower, and that would be the
trigger to measure the actual latency and consider a real ANN migration,
not a preemptive one. Swapping the ranking backend later is a contained
change because search already sits behind a single ranking service used by
the REST API, the exports, and the MCP surface.
