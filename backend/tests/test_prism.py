"""PRISM mechanism tests (numpy — always run) + a torch training smoke test
(skipped where torch is absent, e.g. light CI). See docs/PRISM.md §4."""
import numpy as np
import pytest

from app.ml.prism import PrismIndex, paired_bootstrap_delta, recall_table


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# -- Identity anchor: zero-init PRISM must rank exactly like cosine ----------

def test_identity_prism_ranks_like_cosine():
    rng = np.random.default_rng(0)
    embs = rng.normal(size=(50, 16)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    ids = np.arange(100, 150)
    prism = PrismIndex.identity_from_embeddings(ids, embs)

    q = _unit(rng.normal(size=16))
    cosine_order = ids[np.argsort(-(embs @ q))]
    prism_order = [sid for sid, _ in prism.search(q, top_k=50)]
    assert list(cosine_order) == prism_order


def test_identity_holds_for_any_constant_bandwidth():
    rng = np.random.default_rng(1)
    embs = rng.normal(size=(20, 8)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    ids = np.arange(20)
    q = _unit(rng.normal(size=8))
    order_1 = [s for s, _ in PrismIndex.identity_from_embeddings(ids, embs, 1.0).search(q, 20)]
    order_3 = [s for s, _ in PrismIndex.identity_from_embeddings(ids, embs, 3.0).search(q, 20)]
    assert order_1 == order_3  # constant sigma is rank-irrelevant


# -- The sigma mechanism: spread absorbs deviation ---------------------------

def test_wider_sigma_raises_likelihood_of_far_queries():
    # One image, one dimension: query deviates by 3 from mu.
    ids = np.array([1])
    mu = np.zeros((1, 1), dtype=np.float32)
    narrow = PrismIndex(ids, mu, np.log(np.full((1, 1), 1.0, dtype=np.float32)))
    wide = PrismIndex(ids, mu, np.log(np.full((1, 1), 3.0, dtype=np.float32)))
    q = np.array([[3.0]], dtype=np.float32)
    # -(9/2) vs -(9/18) - log(3): the wide speaker explains the deviation.
    assert wide.log_likelihood(q)[0, 0] > narrow.log_likelihood(q)[0, 0]


def test_sigma_is_a_tradeoff_not_a_free_lunch():
    # For a query AT mu, widening sigma strictly lowers likelihood (log-sigma
    # penalty) — a broad speaker cannot dominate everywhere. This is what the
    # InfoNCE term relies on to prevent generative collapse.
    ids = np.array([1])
    mu = np.zeros((1, 1), dtype=np.float32)
    narrow = PrismIndex(ids, mu, np.log(np.full((1, 1), 1.0, dtype=np.float32)))
    wide = PrismIndex(ids, mu, np.log(np.full((1, 1), 3.0, dtype=np.float32)))
    q = np.zeros((1, 1), dtype=np.float32)
    assert narrow.log_likelihood(q)[0, 0] > wide.log_likelihood(q)[0, 0]


# -- The Bayes prior: hubs pay -----------------------------------------------

def test_prior_demotes_hub_images():
    # Gallery: two specialists and one "hub" near the bank's center of mass.
    e1, e2 = np.eye(2, dtype=np.float32)
    hub = _unit(e1 + e2)
    embs = np.stack([e1, e2, hub])
    ids = np.array([10, 20, 30])
    prism = PrismIndex.identity_from_embeddings(ids, embs)

    rng = np.random.default_rng(2)
    bank = np.stack([_unit(e1 + e2 + 0.3 * rng.normal(size=2)) for _ in range(200)])

    q = _unit(np.array([0.8, 0.6]))          # slightly e1-leaning, hub-adjacent
    before = [s for s, _ in prism.search(q, 3)]
    assert before[0] == 30                    # hub wins raw likelihood
    prism.set_prior(bank.astype(np.float32), weight=1.0)
    after = [s for s, _ in prism.search(q, 3)]
    assert after[0] != 30                     # prior demotes the hub
    prism.clear_prior()
    assert [s for s, _ in prism.search(q, 3)] == before


# -- Metrics helpers ----------------------------------------------------------

def test_rank_of_and_recall_table():
    embs = np.eye(4, dtype=np.float32)
    ids = np.array([1, 2, 3, 4])
    prism = PrismIndex.identity_from_embeddings(ids, embs)
    queries = np.eye(4, dtype=np.float32)
    ranks = prism.rank_of(queries, np.array([1, 2, 3, 4]))
    assert (ranks == 0).all()
    table = recall_table(ranks)
    assert table["R@1"] == 1.0 and table["median_rank"] == 1.0


def test_paired_bootstrap_detects_a_real_difference():
    ranks_bad = np.array([5] * 200)
    ranks_good = np.array([0] * 200)
    d = paired_bootstrap_delta(ranks_bad, ranks_good, k=1)
    assert d["delta"] == 1.0 and d["significant"] is True


def test_paired_bootstrap_rejects_noise():
    rng = np.random.default_rng(3)
    ranks_a = rng.integers(0, 10, 500)
    d = paired_bootstrap_delta(ranks_a, ranks_a, k=1)
    assert d["delta"] == 0.0 and d["significant"] is False


# -- Torch training smoke test (P1 in miniature) ------------------------------

def test_training_learns_a_systematic_modality_offset():
    torch = pytest.importorskip("torch")  # noqa: F841  (skip in light CI)
    from app.ml.prism_train import SpeakerConfig, train_speaker

    # Synthetic world: caption clouds sit at a fixed ROTATION of the image
    # embedding — a systematic modality gap that cosine cannot correct but a
    # learned mu-head can. 60 images on the unit circle, 5 captions each.
    rng = np.random.default_rng(4)
    n, theta = 60, 0.5
    angles = rng.uniform(0, 2 * np.pi, size=n)
    images = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    caps, cap_row = [], []
    for i in range(n):
        center = images[i] @ rot.T
        for _ in range(5):
            caps.append(_unit(center + 0.05 * rng.normal(size=2)))
            cap_row.append(i)
    caps = np.stack(caps).astype(np.float32)
    cap_row = np.array(cap_row)

    train_rows = np.arange(0, 40)
    val_mask = (cap_row >= 40) & (cap_row < 50)   # val images' captions
    cfg = SpeakerConfig()
    cfg.epochs, cfg.patience, cfg.batch_captions = 60, 60, 64

    mu, log_sigma, info = train_speaker(
        np.arange(n), images, caps, cap_row, train_rows, val_mask, cfg)

    # Test on images 50-59 — never trained, never used for selection.
    from app.ml.prism import PrismIndex
    test_mask = cap_row >= 50
    q, targets = caps[test_mask], cap_row[test_mask]
    baseline = PrismIndex.identity_from_embeddings(np.arange(n), images)
    trained = PrismIndex(np.arange(n), mu, log_sigma)
    r1_base = float((baseline.rank_of(q, targets) < 1).mean())
    r1_trained = float((trained.rank_of(q, targets) < 1).mean())
    assert r1_trained > r1_base  # the head generalizes the rotation to unseen images


# -- Serving cache: the seam /api/search's boosted mode trusts ----------------

def test_serving_cache_probes_validates_and_reloads(tmp_path, monkeypatch):
    from app import config
    from app.ml import prism as prism_mod

    monkeypatch.setattr(config, "EMB_DIR", tmp_path)
    prism_mod.invalidate_prism()
    # No artifacts on disk: boosted mode reports unavailable, never raises.
    assert prism_mod.get_prism_index() is None

    ids = np.arange(5, dtype=np.int64)
    mu = np.eye(5, 8, dtype=np.float32)
    PrismIndex(ids, mu, np.zeros_like(mu)).save(meta={"cfg": "test"})

    # Absence is cached until an explicit invalidate — the same lifecycle as
    # every other index, and what POST /api/admin/reload exists to trigger.
    assert prism_mod.get_prism_index() is None
    prism_mod.invalidate_prism()
    loaded = prism_mod.get_prism_index()
    assert loaded is not None and np.array_equal(loaded.ids, ids)

    # Artifacts from a different corpus must refuse to rank, not rank wrongly:
    # the id sets disagree, so every score would describe the wrong image.
    class OtherCorpus:
        ids = np.arange(99, 104)
        embeddings = np.zeros((5, 8), np.float32)  # same space, different ids

    prism_mod.invalidate_prism()
    assert prism_mod.get_prism_index(OtherCorpus()) is None

    # Artifacts from a different EMBEDDING SPACE must also refuse: after a
    # backbone swap the ids are unchanged, so only the dimension betrays that
    # these mu vectors no longer live where the queries do. Without the guard
    # the first boosted query dies on a shape mismatch instead of falling back.
    class SwappedBackbone:
        ids = np.arange(5, dtype=np.int64)          # same corpus…
        embeddings = np.zeros((5, 16), np.float32)  # …different space (16 ≠ 8)

    prism_mod.invalidate_prism()
    assert prism_mod.get_prism_index(SwappedBackbone()) is None
