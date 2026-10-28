# 0002: SigLIP 2 as the default retrieval model

## Status

Accepted

## Context

Image-caption retrieval needs a joint embedding model, and there is no
shortage of candidates: CLIP variants, SigLIP, SigLIP 2, and increasingly
vision-language models such as Qwen3-VL that can also produce embeddings.
Picking one by reputation alone is not verifiable, and swapping the default
later without a benchmark would make any claim about retrieval quality
unfalsifiable.

## Decision

Keep SigLIP 2 as the established default while allowing Qwen3-VL behind the
same provider interface. The base requirements already support SigLIP 2;
Qwen3-VL requires an optional stack and a separately built index.

Earlier local comparisons motivated the original choice, but no committed
artifact binds their metrics to a provider generation, query sample and
protocol. They are historical observations, not a reproducible claim that
SigLIP 2 is currently better or faster. Before changing or defending the
default on quality grounds, export a new comparison following the
[evaluation guide](../evaluation.md).

## Consequences

The benchmark remains the way to revisit the default. Changes to its ranking
or metric definition bump PROTOCOL_VERSION so cached results cannot be read
as measurements under a new protocol. Protocol 11 fixes optimistic tie ranks
and records the inputs required to compare exported results.

Keeping a second provider behind the same interface costs some abstraction
overhead, but it means the default can be revisited later, given a newer
model or a different benchmark result, without a rewrite of the search or
ranking code, only a re-measurement.
