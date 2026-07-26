"""PRISM experiment runner. See docs/PRISM.md.

    python -m app.train_prism            # train + run the A0-A3 ablation ladder
    python -m app.train_prism --eval     # ladder only, from saved artifacts

Protocol: tune on val, test touched by this command only. Prints the
comparison table and writes cache/prism_eval.json. Deliberately not wired
into /api/search until the verdict is in.
"""
import argparse
import json
import logging
import sys

import numpy as np

from . import config, db
from .ml.index import EmbeddingIndex
from .ml.prism import PrismIndex, paired_bootstrap_delta, recall_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_prism")

PRIOR_WEIGHTS = (0.25, 0.5, 0.75, 1.0)


def load_data():
    img = EmbeddingIndex.load("image")
    cap = EmbeddingIndex.load("caption")
    if img is None or cap is None:
        logger.error("Need image + caption embeddings first: run `python -m app.ingest` "
                     "and `python -m app.analyze`.")
        return None
    conn = db.connect()
    split_of = {r["id"]: r["split"] for r in conn.execute("SELECT id, split FROM samples")}
    cap_sample = {r["id"]: r["sample_id"]
                  for r in conn.execute("SELECT id, sample_id FROM captions")}
    conn.close()

    img_row = {int(i): k for k, i in enumerate(img.ids)}
    keep = [k for k, cid in enumerate(cap.ids)
            if cap_sample.get(int(cid)) in img_row]
    cap_embs = cap.embeddings[keep]
    cap_img_row = np.array([img_row[cap_sample[int(cap.ids[k])]] for k in keep])
    cap_split = np.array([split_of[cap_sample[int(cap.ids[k])]] for k in keep])
    img_split = np.array([split_of.get(int(i), "?") for i in img.ids])
    return img, cap_embs, cap_img_row, cap_split, img_split


def wise_select(img, trained: PrismIndex, queries_val, targets_val) -> PrismIndex:
    """WiSE-FT-style interpolation between the zero-shot and trained speaker
    parameters, selected on val. Recovers the early-epoch peak when the head
    overfits past it; alpha=1.0 (no interpolation) competes on equal terms."""
    best_alpha, best_r1, best_index = 1.0, -1.0, trained
    global_ls = trained.log_sigma.mean(axis=0, keepdims=True)
    for alpha in (0.25, 0.5, 0.75, 1.0):
        mu = alpha * trained.mu + (1 - alpha) * img.embeddings
        mu = mu / np.linalg.norm(mu, axis=1, keepdims=True).clip(1e-8)
        ls = alpha * trained.log_sigma + (1 - alpha) * global_ls
        cand = PrismIndex(trained.ids, mu, ls)
        r1 = float((cand.rank_of(queries_val, targets_val) < 1).mean())
        logger.info("WiSE alpha=%.2f val R@1=%.4f", alpha, r1)
        if r1 > best_r1:
            best_alpha, best_r1, best_index = alpha, r1, cand
    logger.info("WiSE selected alpha=%.2f", best_alpha)
    return best_index


def evaluate_ladder(img, cap_embs, cap_img_row, cap_split, img_split,
                    trained: PrismIndex | None, label: str = "A2_prism"):
    """A0 cosine-equivalent -> (A1/A2 trained) -> A3 +prior. Val tunes lambda,
    test decides."""
    test_q = cap_split == "test"
    val_q = cap_split == "validation"
    queries_test = cap_embs[test_q]
    targets_test = img.ids[cap_img_row[test_q]]
    queries_val = cap_embs[val_q]
    targets_val = img.ids[cap_img_row[val_q]]
    bank = cap_embs[cap_split == "train"]        # inductive prior bank

    results = {}
    a0 = PrismIndex.identity_from_embeddings(img.ids, img.embeddings)
    ranks_a0 = a0.rank_of(queries_test, targets_test)
    results["A0_cosine"] = recall_table(ranks_a0)

    ladder = {"A0": ranks_a0}
    if trained is not None:
        ranks_a2 = trained.rank_of(queries_test, targets_test)
        results[label] = recall_table(ranks_a2)
        results[f"{label}_vs_A0"] = paired_bootstrap_delta(ranks_a0, ranks_a2)
        ladder["A2"] = ranks_a2

        # A3: pick lambda on val, apply once to test.
        best_w, best_val = None, -1.0
        for w in PRIOR_WEIGHTS:
            trained.set_prior(bank, weight=w)
            val_r1 = float((trained.rank_of(queries_val, targets_val) < 1).mean())
            if val_r1 > best_val:
                best_w, best_val = w, val_r1
        trained.set_prior(bank, weight=best_w)
        ranks_a3 = trained.rank_of(queries_test, targets_test)
        trained.clear_prior()
        results["A3_prism_prior"] = {**recall_table(ranks_a3), "lambda": best_w}
        results["A3_vs_A2"] = paired_bootstrap_delta(ladder["A2"], ranks_a3)

    # NNN-limit check (P3): prior over the *identity* model. Lambda tuned on
    # val like every other knob — a hard-coded paper optimum over-corrects
    # here (measured: lambda=0.75 costs ~1.5pp vs lambda=0.25 on this data).
    best_w0, best_v0 = PRIOR_WEIGHTS[0], -1.0
    for w in PRIOR_WEIGHTS:
        a0.set_prior(bank, weight=w)
        v = float((a0.rank_of(queries_val, targets_val) < 1).mean())
        if v > best_v0:
            best_w0, best_v0 = w, v
    a0.set_prior(bank, weight=best_w0)
    ranks_nnn = a0.rank_of(queries_test, targets_test)
    a0.clear_prior()
    results["A0_plus_prior(NNN-limit)"] = {**recall_table(ranks_nnn), "lambda": best_w0}
    results["NNN_vs_A0"] = paired_bootstrap_delta(ranks_a0, ranks_nnn)

    results["protocol"] = {
        "pool_size": int(len(img.ids)), "test_queries": int(test_q.sum()),
        "bank": "train-split captions (inductive)",
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate PRISM (docs/PRISM.md).")
    parser.add_argument("--eval", action="store_true", help="Skip training; use saved artifacts")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-sigma", action="store_true",
                        help="Ablation A1: mu head only, sigma stays global")
    args = parser.parse_args()

    data = load_data()
    if data is None:
        return 1
    img, cap_embs, cap_img_row, cap_split, img_split = data
    label = "A1_prism_mu_only" if args.no_sigma else "A2_prism"

    trained = None
    if args.eval:
        trained = PrismIndex.load()
        if trained is None:
            logger.warning("No saved PRISM artifacts; evaluating baselines only.")
    else:
        from .ml.prism_train import SpeakerConfig, train_speaker

        cfg = SpeakerConfig()
        cfg.epochs, cfg.beta, cfg.seed = args.epochs, args.beta, args.seed
        cfg.learn_sigma = not args.no_sigma
        train_rows = np.nonzero(img_split == "train")[0]
        val_mask = cap_split == "validation"
        mu, log_sigma, info = train_speaker(
            img.ids, img.embeddings, cap_embs, cap_img_row,
            train_rows, val_mask, cfg)
        trained = PrismIndex(img.ids, mu, log_sigma)
        # WiSE-FT interpolation, selected on val (recovers early-epoch peaks).
        trained = wise_select(img, trained, cap_embs[val_mask],
                              img.ids[cap_img_row[val_mask]])
        trained.save(meta={"cfg": {"epochs": cfg.epochs, "beta": cfg.beta,
                                   "tau": cfg.tau, "seed": cfg.seed,
                                   "learn_sigma": cfg.learn_sigma}, **{
                           k: v for k, v in info.items() if k != "history"}})
        logger.info("Best val R@1 during training: %.4f", info["best_val_r1"])

    results = evaluate_ladder(img, cap_embs, cap_img_row, cap_split, img_split,
                              trained, label=label)
    config.ensure_dirs()
    out = config.CACHE_DIR / "prism_eval.json"
    out.write_text(json.dumps(results, indent=2))

    print("\n=== PRISM ablation ladder (test split, full-corpus pool) ===")
    for name, table in results.items():
        if isinstance(table, dict) and "R@1" in table:
            print(f"{name:28s} " + "  ".join(f"{k}={v}" for k, v in table.items()))
    for name, d in results.items():
        if isinstance(d, dict) and "_vs_" in name and "delta" in d:
            print(f"{name:28s} ΔR@1={d['delta']:+0.4f}  CI95={d['ci95']}  "
                  f"{'SIGNIFICANT' if d['significant'] else 'not significant'}")
    print(f"\nFull results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
