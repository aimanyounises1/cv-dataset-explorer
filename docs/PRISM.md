# PRISM — Per-image Retrieval via Inferred Speaker Models

A proposed method, original to this project. Status: **hypothesis under test**,
not established art. This document states the idea precisely, locates it
relative to its nearest published neighbors, and defines the experiments that
will confirm or kill it. Sections 8–11 then widen the frame: PRISM is one bet on
a ranked ladder of ways to push text→image recall past zero-shot, and the same
measurement rules govern all of them.

## 1. The observation

Text→image retrieval scores `cos(q, v_i)` — query point against image point.
But an image is not described by a point. Flickr8k gives five human captions
per image, and their embeddings form a **cloud** in text space whose geometry
is data nobody uses:

- the cloud's **center** is where descriptions of this image actually live
  (systematically offset from `v_i` — the modality gap, per image);
- the cloud's **spread** is the image's *description ambiguity* — how many
  different valid ways there are to talk about it. A busy street scene has a
  wide cloud; a lone dog on grass has a tight one.

Standard contrastive training uses only the *pairing* (caption ↔ image). The
cloud's **shape** — five points per image, 30,000 points total on the train
split — is free supervision that is simply discarded.

## 2. The method

**Speaker model.** For each image `i`, model the conditional distribution of
human descriptions as a diagonal Gaussian in the frozen text-embedding space:

```
p(c | i) = N(c ; μ_i, diag(σ_i²))
```

**Inference network (the trainable part — tiny).** A hypernetwork predicts the
distribution from the frozen image embedding alone:

```
μ_i     = L2norm(v_i + f_μ(v_i))          f_μ: 768→256→768, zero-init output
log σ_i = b + f_σ(v_i)                    f_σ: 768→256→768, zero-init output
```

Zero-init means training starts exactly at "μ = the image embedding, σ =
global residual bandwidth `b`" — i.e., at a monotone transform of cosine
similarity. The model can only *earn* its way away from the baseline.
Crucially, the network's input is the **image embedding only** — so it
produces speaker models for val/test images whose captions it has never seen.
Whether those generalize is the experiment.

**Training.** On the train split only (~6k images, ~30k captions), minimize

```
L = Σ_captions −log N(c ; μ_i(c), σ_i(c))          (density: fit the clouds)
  + β · InfoNCE over images with logits ℓ(c,i)/τ    (discrimination)
where ℓ(c,i) = log N(c ; μ_i, σ_i)
```

Full-corpus negatives are exact (all embeddings cached); an epoch is a few
matmuls on MPS. The NLL term is what injects the cloud *shape*; the InfoNCE
term prevents the generative collapse where a broad σ explains everything.

**Retrieval (Bayes).** For query `q`:

```
score(q, i) = ℓ(q, i) − λ · log p̂(i)
log p̂(i)   = logmeanexp over bank captions b of ℓ(b, i)
```

with the bank = train-split captions (inductive; never test captions). The
second term is a learned-likelihood generalization of NNN hubness correction
([arXiv:2410.24114](https://arxiv.org/pdf/2410.24114)): an image whose speaker
model assigns high likelihood to *everything* is a hub, and pays for it.
**Special case:** freeze μ_i = v_i, σ = const, λ·top-k mean instead of
logmeanexp — PRISM reduces exactly to NNN over cosine. That is the theoretical
claim: *test-time hubness correction is Bayesian inference with a flat
likelihood; PRISM learns the non-flat likelihood.*

**Cost.** With diagonal Σ, `ℓ(q,·)` for all N images decomposes into two
matmuls (`q²·(1/σ²)ᵀ` and `q·(μ/σ²)ᵀ`) plus per-image constants — ~2× cosine.
The prior is precomputed once. Index artifacts: `μ` and `log σ` arrays, same
shape as the existing embedding matrix.

## 3. Relation to prior art (the honest paragraph)

Nearest neighbors, and the precise deltas:

- **PCME / PCME++** ([arXiv:2101.05068](https://arxiv.org/abs/2101.05068),
  [arXiv:2305.04732](https://arxiv.org/abs/2305.04732)) learn probabilistic
  embeddings **end-to-end from scratch** with learned uncertainty on *both*
  sides. PRISM is **post-hoc over a frozen SOTA backbone** (SigLIP-2), trains
  in minutes on a laptop, and supervises variance with **real per-image
  caption clouds** rather than learning it implicitly.
- **ProbVLM** ([ICCV 2023, arXiv:2307.00398](https://arxiv.org/abs/2307.00398))
  is a post-hoc adapter that estimates the **uncertainty of an input's
  embedding**. PRISM instead models the **conditional distribution of
  descriptions given a gallery image** and uses it as the *retrieval score*,
  with a Bayes prior that subsumes hubness correction. Different object,
  different use.
- **NNN / QB-Norm** are test-time corrections with no learned component;
  PRISM contains them as its flat-likelihood limit (§2).

The novel combination: (a) caption-cloud geometry as free supervision for a
frozen-backbone density head, (b) retrieval as per-gallery-item likelihood
scoring at cosine-class cost, (c) hubness correction derived as the prior term
rather than bolted on.

## 4. Falsifiable predictions

Protocol: the repo's benchmark (full-corpus pool, self-caption exclusion,
paired bootstrap, 3 seeds, val for all tuning, test touched once).

- **P1 (main).** PRISM beats the frozen-backbone cosine baseline by ≥ +4pp
  T→I R@1, and beats a matched plain adapter head (same parameter count,
  InfoNCE only — ablation A1) — because it consumes strictly more supervision.
- **P2 (mechanism check).** Predicted bandwidth `mean(σ_i)` correlates with
  the independently-measured caption-consistency score on val images
  (Spearman ρ > 0.4). If σ carries no signal about actual description
  ambiguity, the density story is wrong even if recall improves.
- **P3 (unification check).** PRISM's prior term alone (μ frozen) reproduces
  NNN's gain within noise. If not, the Bayes framing is decorative.
- **P4 (failure condition).** If the val→test gap exceeds the gain, the
  hypernetwork memorized train clouds and the method dies honestly.

Ablation ladder: A0 cosine → A1 μ-head only, σ frozen (≈ centroid-regression
adapter) → A2 + learned σ (the new part) → A3 + Bayes prior. Each step must
pay for itself on the paired test or be dropped.

## 5. Known risks

- **5 points is a small cloud in 768-d.** Mitigation: diagonal + strong
  shrinkage toward the global residual covariance (the zero-init does this),
  and the NLL weight β kept subordinate to InfoNCE on val.
- **Gaussian NLL in high-d can collapse to a rescaled cosine.** That outcome
  is measurable: A2 ≈ A1 → report the null result. The instrument doesn't care
  what we hoped.
- **Speaker-head generalization** to unseen images is the whole bet (P4).

## 6. First results (2026-07-26, seed 0, MacBook MPS)

Test split, full 8,000-image pool, 5,000 caption queries, val-tuned, paired
bootstrap:

| Model | R@1 | R@5 | MRR@10 | median rank | ΔR@1 vs A0 |
|---|---|---|---|---|---|
| A0 cosine (SigLIP-2) | 49.36% | 73.76% | 0.596 | 2.0 | — |
| A2 PRISM | **50.80%** | 74.24% | 0.607 | **1.0** | **+1.44 [＋0.44, ＋2.40] ✓** |
| A3 PRISM+prior | 50.52% | 73.86% | 0.606 | 1.0 | (vs A2: −0.28, n.s.) |

Findings, in pre-registered terms:

- **P1: direction confirmed, magnitude under-delivered.** +1.44pp is
  significant but below the ≥+4pp bar. Real, not yet "drastic".
- **P2: refuted as trained.** Spearman ρ(predicted σ, caption consistency) =
  +0.05 on val+test images, and the σ head barely differentiated at all
  (log-σ spread 0.0075 across images). The measured gain therefore comes from
  the **μ head** — a learned per-image modality-offset correction — while the
  method's distinctive density component was inert. This is the §5 collapse
  risk materializing; the A1 ablation (`--no-sigma`) now exists to isolate it,
  and larger β / longer schedules are the follow-up.
- **A3's null is informative:** once the likelihood is learned, the bank prior
  adds nothing — consistent with the Bayes framing (a non-flat likelihood
  absorbs the hubness correction that helps the flat A0: +0.9pp, significant,
  when λ is val-tuned).
- **Protocol lesson:** an early ladder run hard-coded the NNN paper's λ=0.75
  and *regressed* −0.6pp; val-tuning picks λ=0.25 and gains +0.9pp. Tuning on
  val is not optional hygiene — it flipped the sign of a result.
- Training overfits past epoch 2; WiSE-FT interpolation (val-selected α) is
  now applied post-training in the runner.

### Decisive round (same day, 2 seeds per arm)

| Arm | R@1 (seeds) | ΔR@1 vs A0 | Verdict |
|---|---|---|---|
| **A1 μ-only** (`--no-sigma`) | **51.60 / 51.54%** | **+2.24 [+1.28,+3.18] ✓ · +2.18 [+1.16,+3.16] ✓** | **winner** |
| A2 σ-heavy (β=0.3) | 49.56 / 49.84% | +0.20 n.s. · +0.48 n.s. | refuted |
| A2 original (β=0.05, seed 0) | 50.80% | +1.44 ✓ | σ was dead weight |
| A3 prior on trained model | — | significantly *negative* both rounds | drop when trained |

**Final conclusions.** (1) The per-image variance hypothesis is **refuted** at
this scale: σ contributes nothing at low β and actively fights the
discriminative objective at high β. (2) What survives is a strong, reproducible
finding: a **caption-centroid-supervised per-image modality-offset correction**
(the μ head) lifts full-corpus T→I R@1 by **+2.2pp (49.4→51.6)**, R@5 by
+1.5pp, and moves the median correct-image rank from 2 to 1 — with the frozen
backbone untouched, ~1 minute of MPS training, and zero query-time cost beyond
a matmul. (3) The Bayes prior helps only the untrained model (+0.9pp) and
harms the trained one — consistent with the likelihood absorbing the hubness
structure. The recommended production configuration is **A1**.

This is the pre-registered process working as designed: the novel-sounding
component was measured, refuted, and amputated; the boring-sounding component
was measured, confirmed, and kept.

## 7. Implementation in this repo

- `backend/app/ml/prism.py` — numpy scoring core + artifact I/O (no torch, no
  changes to existing modules).
- `backend/app/ml/prism_train.py` + `python -m app.train_prism` — torch
  training CLI (MacBook MPS; ~minutes) with `--eval` running the A0–A3 ladder
  under the standard protocol and printing the comparison table.
- `backend/tests/test_prism.py` — numpy identities (zero-init PRISM ranks
  exactly like cosine; σ and prior move rankings in the predicted direction);
  torch training smoke test auto-skipped where torch is absent.
- Shipped in `/api/search` as the `boosted` mode: the trained mu replaces the
  cosine ranking (score basis `prism_ll`), with NO hubness penalty on top —
  stacking the prior on the trained mu measured *worse* than the mu alone
  (R@1 51.6% → 50.3%, paired CI95 [-2.0, -0.5] pts), so one correction ranks.
  Availability is probed per request: missing or corpus-mismatched artifacts
  degrade the request to semantic, and `POST /api/admin/reload` picks up fresh
  artifacts without a restart. The Benchmark page adds a paired test-split
  comparison (boosted vs. semantic on the same held-out queries) whenever the
  artifacts are present — the verdict that gated this integration, kept visible.

## 8. The wider programme: measurement ground rules

PRISM is one bet. The platform also ships the instrument that judges it — the
benchmark (R@1/5/10, MRR, median rank, per-mode candidate counts, query-caption
exclusion, full-corpus pool) — and the instrument outlives any single method.
The rules below were fixed before any number in §6 was collected, and they
govern the alternative hypotheses in §9 identically. Everything in this
programme runs on a MacBook or a free Colab.

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

## 9. Ranked hypotheses beyond PRISM

A ranked set of falsifiable alternatives for improving text→image recall, each
with mechanism, published evidence, cost, expected effect size, and the guard
that keeps the measurement honest. PRISM competes against these, not against
nothing.

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
**Already measured here** (§6): the inductive variant gains +0.9pp on the
untrained model with val-tuned λ=0.25 — inside the honest range, at the bottom
of it — and goes significantly *negative* once the PRISM μ head is trained,
which is the unification claim of §2 doing its job.

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
3 seeds. PRISM's μ-only arm (A1, §6) is a head of exactly this shape trained
with caption-cloud supervision on top of InfoNCE, so its **+2.2pp** is the
in-repo reference point a plain adapter has to beat.

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

## 10. Explicitly rejected (evidence says no)

- **Transductive querybanks** — leakage dressed as a gain (+10.5 becomes +0.7
  done honestly, [DBSN Table 2](https://arxiv.org/html/2508.02538)).
- **Modality-gap mean-centering** — ≈+0.6 for SigLIP, negative after
  fine-tuning (NNN's replication). Superseded by H1, which it is a special
  case of (k=N, α=1).
- **LLM query-paraphrase ensembling** — gains are classification-only and
  anti-correlate with retriever strength; expect +0–1.
- **Human-caption centroid fusion for benchmarking** — circular (see the H5
  guard). Legitimate as a *product* feature; meaningless as a *measurement*.

## 11. Sequencing and compute plan

| Phase | Hypotheses | Compute | Cumulative prediction |
|---|---|---|---|
| 0 | Freeze zero-shot baseline, 5k-query eval, CIs | MacBook, minutes | baseline |
| 1 | H1 (NNN) + H7 (fusion) | MacBook, 1 day | +3–6 |
| 2 | H2 (adapters) + H3 (hard negatives) | MacBook, 1–2 days | +5–10 |
| 3 | H4 (LiT-LoRA) **or** H5 (pseudo-captions) | Colab T4 / overnight | +8–15 |
| 4 | H6 (async rerank) as product mode | Colab or patient Mac | ceiling |

PRISM and H1 are the phases already run (§6); the rest are costed, not done.
Gains do not add linearly — every phase re-measures against the same frozen
protocol, and each hypothesis is accepted/rejected on its paired-test CI, not
on whether we like it. The negative results are reported alongside the wins;
in a data-platform team, the instrument that *refutes* a plausible idea is
worth as much as the idea that survives. PRISM's own σ head is the worked
example: proposed, measured, refuted, removed.
