# 0005: Human review is the write boundary

## Status

Accepted

## Context

Vision-language models, detectors, and segmenters can all produce a
plausible-looking result: a caption, a class label, a box, or a mask. It
is tempting to let a confident model output become part of the dataset
directly, since that is what makes a tool feel automated. But a Flickr8k
copy already ships with undocumented gaps (see the README's Data
provenance section), and adding silent, unreviewed model writes on top of
that would make the dataset's provenance worse, not better.

## Decision

A model output is always presented as a proposal. Detection and
segmentation results are held in the browser until a reviewer explicitly
accepts them; nothing is written to the dataset from a vision or language
model without that explicit step. Read operations, including the
assistant's and the MCP surface's, never trigger a write on their own.

## Consequences

Every accepted mask, caption suggestion, or annotation in the dataset can
be traced back to a human decision, which keeps the provenance story
consistent with how the tool already treats the upstream data: caveats
stated plainly rather than hidden. The assistant and the automated
detectors can be as capable as they like without changing this guarantee.

The cost is friction: nothing gets faster by trusting a model output on
sight, and every write path needs an explicit acceptance step in the UI
rather than a shortcut. That friction is the point, not a limitation to
engineer away.
