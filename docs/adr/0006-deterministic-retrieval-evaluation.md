# 0006: Deterministic retrieval evaluation

## Status

Accepted

## Context

A retrieval metric must describe the order users can actually inspect. The
previous evaluator counted only scores strictly greater than the target's
score. With four identical image vectors it assigned all four rank one,
although search could display only one first result. Search used a partial
partition, whose selection of tied rows depended on array layout.

A provider-comparison JSON also retained recall alone, losing the evaluation
protocol, query sample, model revision and configuration. Such a number cannot
support an independently inspectable model-choice claim.

## Decision

Use one helper for both paths: score descending, then stable item ID ascending.
Full lexicographic sorting selects ties consistently even at the top-k cutoff.
Optimistic shared ranks overstate visible retrieval; random tie order adds
variance; array-row order changes when artifacts are rewritten. Item IDs are
already the application's stable sample identity.

Protocol 11 computes exact semantic ranks from that order and keeps the
existing held-out lexical query and hubness-bank exclusions. Its response
records the ordered query IDs, corpus-caption digest, provider generation and
manifest, effective ranking configuration, encoding mode and measurement time.
The comparison script preserves this complete response and adds run/source
context to a caller-selected artifact path.

## Consequences

Sorting is O(N log N) instead of partial selection. The corpus is only 8,000
images, so a single obvious ordering contract is preferable here to a custom
partition implementation with a second tie-selection rule. No latency claim
is made for this change; measure it before generalizing to a larger corpus.

The deterministic synthetic regression yields semantic R@1 of 0.25 for four
identical vectors and checks that the endpoint order matches search. This is
a correctness check, not a model-quality benchmark. Protocol-10 results are
not comparable under ties; new performance claims require exported full-data
measurements. Stored-caption fallback remains visible and is not equivalent
to the live normalized query encoder.
