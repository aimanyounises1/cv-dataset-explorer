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

SigLIP 2 is the default embedding model, chosen because it scored better
than the alternatives on this project's own Flickr8k retrieval benchmark,
not because of general reputation. Qwen3-VL remains available as an
optional retrieval provider behind the same interface, for cases where a
reviewer wants to compare it directly.

## Consequences

The default is falsifiable: it names the benchmark that produced it, and a
change to that benchmark's definition bumps PROTOCOL_VERSION so a cached
result can never be misread as comparable to a new one. Anyone who disputes
the choice can re-run the same benchmark rather than argue about which
model should be better in the abstract.

Keeping a second provider behind the same interface costs some abstraction
overhead, but it means the default can be revisited later, given a newer
model or a different benchmark result, without a rewrite of the search or
ranking code, only a re-measurement.
