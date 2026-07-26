# Research Program — Improving Retrieval Accuracy Beyond Zero-Shot

The platform ships with a measurement instrument (the benchmark: R@1/5/10, MRR,
median rank, per-mode candidate counts, query-caption exclusion, full-corpus
pool). This document uses it: a ranked set of falsifiable hypotheses for
drastically improving text→image recall, each with mechanism, published
evidence, cost, expected effect size, and the guard that keeps the measurement
honest. Everything below runs on a MacBook or a free Colab.

## Ground rules (before any hypothesis)

- **Split hygiene.** Anything trained or tuned sees the train split only;
  hyperparameters selected on val R@1; test touched once per hypothesis.
- **Noise floor.** R@1 over 1,000 queries is binomial: SE ≈ 1.6pp near 50%
  (95% CI ≈ ±3.1pp). Deltas under ~+2pp are noise at this sample size. Use all
  ~5,000 test-split captions as queries (CI ≈ ±1.4pp), paired bootstrap or
  McNemar for significance, and ≥3 seeds for anything trained
  ([Metric Learning Reality Check](https://arxiv.org/abs/2003.08505)).
- **Effect-size vocabulary.** +2pp = within noise. +5pp = solid, defensible.
  +10–16pp = matches published on-domain fine-tuning deltas
  ([arXiv:2406.17639](https://arxiv.org/pdf/2406.17639)) and qualifies as
  "drastic".
- **Pool honesty.** Published Flickr numbers use a 1,000-image gallery; ours
  ranks the full ~8k corpus (~8× harder). Never compare across pools; always
  state the pool. Reference points at 1k-pool T→I R@1: CLIP-L 68.7 · SigLIP
  B/16 77.9 · SigLIP-2 B/16 80.7 · fine-tuned BEiT-3 90.3.

## Ranked hypotheses

### H1 — Hubness correction (NNN). Training-free, ~10 lines of numpy.

**Claim.** Some gallery images are "hubs" that appear in the top-k of many
queries; subtracting a per-image bias estimated from a caption bank improves
R@1. Score `s(q,g) − α·mean(topk_b(B·g))`, with bank B = train-split caption
embeddings ([NNN, arXiv:2410.24114](https://arxiv.org/pdf/2410.24114)).

**Evidence.** SigLIP on Flickr30k: +1.9 R@1; COCO (bigger pool, like ours):
+3.1. Critical caveat found in the literature review: the famous "+8 to +10
hubness gains" ([QB-Norm](https://arxiv.org/abs/2112.12777),
[DBSN](https://arxiv.org/html/2508.02538)) are **transductive** — the
querybank is the test queries themselves. Using test captions as the bank is
leakage; our protocol forbids it. Inductive (train-bank) gains are the honest
ones: +0.7…+3.
**Prediction.** +2 to +5 R@1 (our pool is COCO-like). **Cost:** an afternoon,
MacBook. **Falsifier:** paired bootstrap CI crossing zero.
**Sweep:** α ∈ {0.25…1.5}, k ∈ {16…512} on val (paper optimum α=0.75, k=128).
**Guard:** report raw and corrected side by side; the transductive variant may
be reported only as a clearly-labelled upper bound.

### H2 — Residual adapter heads over frozen embeddings. The effort/gain sweet spot.

**Claim.** A small trained head per tower — `L2norm(x + α·MLP(x))`, zero-init
last layer, 768→256→768 — specializes the frozen embedding space to this
corpus. Because all embeddings are cached `.npy`, InfoNCE can use
**full-corpus negatives** (impossible in normal fine-tuning), on MPS in
seconds per epoch.

**Evidence.** CLIP-Adapter/TaskRes residual blending
([arXiv:2110.04544](https://arxiv.org/abs/2110.04544),
[arXiv:2211.10277](https://arxiv.org/abs/2211.10277)); frozen-feature heads
recover 85–95% of full fine-tuning ([FrEVL,
arXiv:2508.04469](https://arxiv.org/abs/2508.04469)); on-domain contrastive
fine-tuning moved CLIP Flickr30k T→I R@1 +16.4pp
([arXiv:2406.17639](https://arxiv.org/pdf/2406.17639)).
**Prediction.** +3 to +8 R@1 over zero-shot on our benchmark.
**Cost:** ~1 day incl. sweeps; MacBook only.
**Guards against the known failure (overfitting 6k images):** zero-init +
residual α tuned on val; weight decay (FrEVL loses 3pp without it); WiSE-FT
interpolation between original and adapted embeddings
([arXiv:2109.01903](https://arxiv.org/abs/2109.01903)); early stop on val R@1;
3 seeds.

### H3 — Hard-negative mining from our own index. Stacks on H2.

**Claim.** Mixing full-batch InfoNCE with top-10–50 *mined* negatives
(retrieved-but-wrong images per training caption, re-mined each epoch) sharpens
the decision boundary exactly where retrieval fails.
**Evidence.** VSE++ (+~17–23% relative R@1 at Flickr30k scale, partly with
frozen features, [arXiv:1707.05612](https://arxiv.org/abs/1707.05612)); ELIP
uses global hard mining as the limited-compute practice
([arXiv:2502.15682](https://arxiv.org/abs/2502.15682)).
**Prediction.** +1 to +3 R@1 on top of H2. **Cost:** hours.
**Guard:** mask sibling captions of the same image from the negative pool
(false negatives — 5 captions per image).

### H4 — LiT-style text-tower LoRA against cached image embeddings. Highest ceiling.

**Claim.** Training the text tower (LoRA r=16 on attention, ~4M params) with
the image tower *frozen as cached vectors* is exactly
[LiT](https://arxiv.org/abs/2111.07991) — and needs zero image forward passes.
The text side adapts to Flickr8k's caption register; the image side can't
overfit 6k images because it never trains.
**Evidence.** Locked-image tuning beat alternatives under budget constraints
(NLLB-CLIP); on-domain fine-tuning deltas of +10–16pp R@1 are the published
pattern.
**Prediction.** +5 to +15 R@1. **Cost:** minutes/epoch on Colab T4, ~1–3h on
M-series; the only hypothesis where Colab/GCP is worth it.
**Guard:** this is the highest-variance bet — lr 1e-4, ≤10 epochs, early stop
on val, WiSE-FT fallback, and report the val/test gap explicitly.

### H5 — Pseudo-caption multi-vector index (Doc2Query for images).

**Claim.** Generate 3–5 captions per image with a local VLM (Qwen2.5-VL via
Ollama, one overnight run), index them in both FTS and the dense index, and
score `max(cos(q,img), λ·max_k cos(q,pseudo_k))`. Expands each image's textual
"surface area" — the retrieval analog of
[docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf)
(BM25 MRR +50% on MS MARCO); training-free fusion of this kind reports ~+10%
top-1 ([F4-ITS, arXiv:2508.17037](https://arxiv.org/abs/2508.17037)).
**Prediction.** +3 to +8 R@1, largest on rare-entity/lexical queries.
**Cost:** offline GPU-hours; zero query latency.
**Guard (the circularity trap, from the literature review):** the *human*
captions must never enter the image-side index for benchmarking — sibling
captions of one image are near-paraphrases
([CxC](https://aclanthology.org/2021.eacl-main.249/)), so indexing them
measures caption-matching, not image understanding. Pseudo-captions only.
Filter hallucinated captions by SigLIP agreement before indexing
([Doc2Query−−](https://arxiv.org/pdf/2301.03266): unfiltered generated text
*hurts*).

### H6 — Cross-encoder reranking of top-20 ("deep search" mode).

**Claim.** ITC-retrieve → ITM-rerank is the classic recipe (BLIP-2 family sits
~15–20pp above plain cosine on Flickr T→I R@1). Local options: BLIP ITM head
(0.4B, ~50–100ms/pair on T4) or
[Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B).
**Prediction.** +5 to +10 R@1 — but too slow for interactive search on a
MacBook (0.5–2s/pair) → ship as an async "deep search" toggle.
**Guard:** rerankers can *undo* correct top-1
([INQUIRE](https://arxiv.org/pdf/2411.02537): small rerankers ≈ random);
blend `α·rerank + (1−α)·dense` and never demote below a rank floor; always A/B
against no-rerank.

### H7 — Heterogeneous encoder fusion (cheap, small).

Second bi-encoder (MobileCLIP2-S4 or nomic-embed-vision) fused by tuned
weighted-sum over min-max-normalized scores (beats RRF,
[Bruch et al.](https://arxiv.org/abs/2210.11934)). Prediction +1–3 R@1;
minutes to precompute. Worth doing as an ensemble floor under H2/H4.

## Explicitly rejected (evidence says no)

- **Transductive querybanks** — leakage dressed as a gain (+10.5 becomes +0.7
  done honestly, [DBSN Table 2](https://arxiv.org/html/2508.02538)).
- **Modality-gap mean-centering** — ≈+0.6 for SigLIP, negative after
  fine-tuning (NNN's replication). Superseded by H1, which it is a special
  case of (k=N, α=1).
- **LLM query-paraphrase ensembling** — gains are classification-only and
  anti-correlate with retriever strength; expect +0–1.
- **Human-caption centroid fusion for benchmarking** — circular (see H5 guard).
  Legitimate as a *product* feature; meaningless as a *measurement*.

## Sequencing and compute plan

| Phase | Hypotheses | Compute | Cumulative prediction |
|---|---|---|---|
| 0 | Freeze zero-shot baseline, 5k-query eval, CIs | MacBook, minutes | baseline |
| 1 | H1 (NNN) + H7 (fusion) | MacBook, 1 day | +3–6 |
| 2 | H2 (adapters) + H3 (hard negatives) | MacBook, 1–2 days | +5–10 |
| 3 | H4 (LiT-LoRA) **or** H5 (pseudo-captions) | Colab T4 / overnight | +8–15 |
| 4 | H6 (async rerank) as product mode | Colab or patient Mac | ceiling |

Gains do not add linearly — every phase re-measures against the same frozen
protocol, and each hypothesis is accepted/rejected on its paired-test CI, not
on whether we like it. The negative results are reported alongside the wins;
in a data-platform team, the instrument that *refutes* a plausible idea is
worth as much as the idea that survives.

## H★ — PRISM (original contribution)

Beyond the literature-derived hypotheses above, this project proposes one
method of its own: **PRISM — Per-image Retrieval via Inferred Speaker Models**
(spec, math, ablation ladder and falsification plan in
[PRISM.md](PRISM.md); runner: `python -m app.train_prism`). One sentence:
learn, from each training image's five-caption *cloud*, a tiny hypernetwork
that predicts a per-image distribution over human descriptions, rank by query
likelihood under that distribution, and derive hubness correction as the Bayes
prior — of which NNN (H1) is the flat-likelihood special case.
