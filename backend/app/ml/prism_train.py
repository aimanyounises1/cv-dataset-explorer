"""PRISM training (torch; MPS/CUDA/CPU). See docs/PRISM.md.

Trains the speaker hypernetwork on the train split's caption clouds, selects
on val R@1, and materializes speaker models (mu, log_sigma) for the FULL
corpus — legitimately, since the network's input is the image embedding only.

Deliberately isolated: imports nothing from the API layer, changes no shared
module. The product does not ship this hypothesis until the benchmark says so.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


class SpeakerConfig:
    hidden: int = 256
    tau: float = 0.05          # InfoNCE temperature over log-likelihood logits
    beta: float = 0.05         # weight of the density (NLL) term
    learn_sigma: bool = True   # False = A1 ablation (mu head only, global sigma)
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 40
    batch_captions: int = 1024
    log_sigma_clamp: tuple = (-4.0, 2.0)
    patience: int = 6          # early-stop epochs without val R@1 improvement
    seed: int = 0


def _build_model(dim: int, cfg: SpeakerConfig, residual_log_sigma: np.ndarray):
    import torch
    from torch import nn

    torch.manual_seed(cfg.seed)

    def head() -> nn.Sequential:
        m = nn.Sequential(nn.Linear(dim, cfg.hidden), nn.GELU(),
                          nn.Linear(cfg.hidden, dim))
        nn.init.zeros_(m[-1].weight)   # zero-init: start exactly at baseline
        nn.init.zeros_(m[-1].bias)
        return m

    class Speaker(nn.Module):
        def __init__(self):
            super().__init__()
            self.f_mu = head()
            self.f_sigma = head()
            # Global log-bandwidth, initialized from the measured per-dim std
            # of (caption - paired image) residuals on the train split.
            self.b = nn.Parameter(torch.tensor(residual_log_sigma, dtype=torch.float32))

        def forward(self, v):  # v: (n, d) frozen image embeddings
            mu = v + self.f_mu(v)
            mu = mu / mu.norm(dim=1, keepdim=True).clamp_min(1e-8)
            per_image = self.f_sigma(v) if cfg.learn_sigma else torch.zeros_like(v)
            log_sigma = (self.b + per_image).clamp(*cfg.log_sigma_clamp)
            return mu, log_sigma

    return Speaker()


def _ll(q, mu, log_sigma):
    """(m,d) queries vs (n,d) speaker params -> (m,n) log-likelihood (torch)."""
    import torch

    r = torch.exp(-2.0 * log_sigma)
    quad = (q * q) @ r.T - 2.0 * (q @ (mu * r).T) \
        + (mu * mu * r).sum(1) + (2.0 * log_sigma).sum(1)
    return -0.5 * quad


def train_speaker(
    image_ids: np.ndarray, image_embs: np.ndarray,
    caption_embs: np.ndarray, caption_image_row: np.ndarray,
    train_rows: np.ndarray, val_caption_mask: np.ndarray,
    cfg: SpeakerConfig = SpeakerConfig(),
):
    """Returns (mu, log_sigma) for ALL images + training history.

    caption_image_row[j] = row index (into image_embs) of caption j's image.
    train_rows = image rows whose captions may be trained on.
    val_caption_mask = captions used for early stopping (their images excluded
    from training rows by the caller).
    """
    import torch

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training PRISM speaker head on %s", device)

    V = torch.tensor(image_embs, dtype=torch.float32, device=device)
    C = torch.tensor(caption_embs, dtype=torch.float32, device=device)
    cap_row = torch.tensor(caption_image_row, dtype=torch.long, device=device)

    train_row_set = torch.tensor(train_rows, dtype=torch.long, device=device)
    # Position of each train image row within the train gallery (for CE labels).
    pos_in_train = torch.full((len(image_embs),), -1, dtype=torch.long, device=device)
    pos_in_train[train_row_set] = torch.arange(len(train_row_set), device=device)

    train_cap_idx = torch.nonzero(
        (pos_in_train[cap_row] >= 0)
        & ~torch.tensor(val_caption_mask, device=device), as_tuple=False).squeeze(1)
    val_cap_idx = torch.nonzero(
        torch.tensor(val_caption_mask, device=device), as_tuple=False).squeeze(1)

    # Global residual bandwidth init (per-dim std of caption - paired image).
    resid = (C[train_cap_idx] - V[cap_row[train_cap_idx]]).cpu().numpy()
    residual_log_sigma = np.log(resid.std(axis=0).clip(1e-3))

    model = _build_model(V.shape[1], cfg, residual_log_sigma).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    def val_r1() -> float:
        model.eval()
        with torch.no_grad():
            mu, ls = model(V)
            hits = 0
            for start in range(0, len(val_cap_idx), 1024):
                idx = val_cap_idx[start:start + 1024]
                s = _ll(C[idx], mu, ls)                      # vs FULL gallery
                hits += int((s.argmax(dim=1) == cap_row[idx]).sum())
        model.train()
        return hits / max(len(val_cap_idx), 1)

    best = {"r1": -1.0, "state": None, "epoch": -1}
    history = []
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    for epoch in range(cfg.epochs):
        perm = train_cap_idx[torch.randperm(len(train_cap_idx), generator=gen)]
        total, batches = 0.0, 0
        for start in range(0, len(perm), cfg.batch_captions):
            idx = perm[start:start + cfg.batch_captions]
            mu_t, ls_t = model(V[train_row_set])             # all train images
            logits = _ll(C[idx], mu_t, ls_t)                 # (b, n_train)
            labels = pos_in_train[cap_row[idx]]
            infonce = torch.nn.functional.cross_entropy(logits / cfg.tau, labels)
            nll = -logits[torch.arange(len(idx), device=device), labels].mean()
            loss = infonce + cfg.beta * nll
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            batches += 1
        r1 = val_r1()
        history.append({"epoch": epoch, "loss": round(total / max(batches, 1), 4),
                        "val_r1": round(r1, 4)})
        logger.info("epoch %d loss %.4f val R@1 %.4f", epoch, total / max(batches, 1), r1)
        if r1 > best["r1"]:
            best = {"r1": r1, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= cfg.patience:
            logger.info("Early stop at epoch %d (best %.4f @ %d)",
                        epoch, best["r1"], best["epoch"])
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        mu, ls = model(V)
    return (mu.cpu().numpy().astype(np.float32),
            ls.cpu().numpy().astype(np.float32),
            {"best_val_r1": best["r1"], "best_epoch": best["epoch"],
             "history": history, "device": device})
