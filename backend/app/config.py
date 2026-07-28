"""Central configuration. Everything is overridable via environment variables
so the app can be pointed at a different data directory, model, or VLM without
code changes.
"""
import os
from pathlib import Path

# Data layout ---------------------------------------------------------------
DATA_DIR = Path(os.environ.get("CVDE_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
IMAGES_DIR = DATA_DIR / "images"
THUMBS_DIR = DATA_DIR / "thumbs"
EMB_DIR = DATA_DIR / "embeddings"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "explorer.db"

# Embedding model ------------------------------------------------------------
# SigLIP 2 base: strong zero-shot retrieval, ~1.5GB, runs on MPS/CUDA/CPU.
# CVDE_EMBED_MODEL keeps its historical meaning: the SigLIP checkpoint.
EMBED_MODEL = os.environ.get("CVDE_EMBED_MODEL", "google/siglip2-base-patch16-256")
EMBED_BATCH_SIZE = int(os.environ.get("CVDE_EMBED_BATCH", "32"))

# Retrieval provider ----------------------------------------------------------
# Default is "siglip2" because it MEASURES better on this corpus's own
# benchmark (R@1 55.2% vs 50.2%, ~5x the encode speed — scripts/
# bench_providers.py; parameter count is not quality). "qwen3_vl" is the
# explicit opt-in multimodal alternative: Qwen/Qwen3-VL-Embedding-2B
# in-process through sentence-transformers (Ollama serves the language models
# only — it cannot host an image-embedding model), falling back to siglip2
# with a visible, named reason whenever its stack, weights or index are
# missing. "mock" is the deterministic test provider.
EMBED_PROVIDER = os.environ.get("CVDE_EMBED_PROVIDER", "siglip2")
QWEN_EMBED_MODEL = os.environ.get("CVDE_QWEN_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
QWEN_EMBED_BATCH = int(os.environ.get("CVDE_QWEN_EMBED_BATCH", "8"))


def emb_dir_for(provider: str) -> Path:
    """siglip2 keeps the original flat embeddings layout (an existing install
    keeps working, untouched); every other provider gets its own subdirectory
    so two vector spaces can never mix in one index."""
    return EMB_DIR if provider == "siglip2" else EMB_DIR / provider

# Optional local VLM enrichment + assistant (via Ollama) ---------------------
OLLAMA_URL = os.environ.get("CVDE_OLLAMA_URL", "http://localhost:11434")
VLM_MODEL = os.environ.get("CVDE_VLM_MODEL", "qwen2.5vl:7b")
CHAT_MODEL = os.environ.get("CVDE_CHAT_MODEL", "qwen3:8b")  # needs tool calling

# Agent bounds. A local 8B model asked a vague question will sometimes loop
# through tools indefinitely, and one stalled Ollama request would otherwise hold
# the HTTP request open until the client gives up. Three nested bounds, outermost
# last: one model call, one specialist lane, and the tool-call loop inside a lane.
OLLAMA_TIMEOUT = float(os.environ.get("CVDE_OLLAMA_TIMEOUT", "120"))
AGENT_LANE_TIMEOUT = float(os.environ.get("CVDE_AGENT_LANE_TIMEOUT", "240"))
AGENT_RECURSION_LIMIT = int(os.environ.get("CVDE_AGENT_RECURSION_LIMIT", "25"))
# Reports the assistant generates are written here so they can be downloaded
# after the turn that produced them has scrolled away.
REPORTS_DIR = DATA_DIR / "reports"

# Retrieval ------------------------------------------------------------------
# Reciprocal-rank-fusion constant. 60 is the value Cormack et al. (SIGIR 2009)
# fixed "during a pilot investigation" — a convention, not a tuned optimum for
# this corpus, so it is configurable and logged with every fused search.
RRF_K = int(os.environ.get("CVDE_RRF_K", "60"))

# Ranking depth held constant across pages of the same query. RRF ranks depend
# on how deep the candidate lists go, so fusing to `offset + page_size` would
# re-rank on every page and let items shift between them. Fusing to a fixed
# depth instead makes paging stable for any page within it.
SEARCH_DEPTH = int(os.environ.get("CVDE_SEARCH_DEPTH", "300"))

# Hubness correction (inverted softmax) --------------------------------------
# A bi-encoder's image vectors are not uniformly reachable: this corpus has a
# 0.556 mean pairwise cosine, and some images sit close to almost every query,
# so they surface for text they do not depict. The correction subtracts a
# per-image "how close is this to queries in general" scalar, estimated once
# offline from a bank of held-out captions.
#
# beta = 0 disables it entirely and restores the previous ranking exactly.
# beta and the temperature were tuned on a dev split that is disjoint from both
# the bank and the benchmark sample — see docs and `python -m app.ml.hubness`.
# Lineage: QB-Norm (Bogolin et al., CVPR 2022); the numbers here are ours.
HUBNESS_BETA = float(os.environ.get("CVDE_HUBNESS_BETA", "0.3"))
HUBNESS_TEMPERATURE = float(os.environ.get("CVDE_HUBNESS_T", "0.02"))
HUBNESS_BANK_SIZE = int(os.environ.get("CVDE_HUBNESS_BANK", "3000"))
# Build the ~32 kB artifact on first use when it is missing. The build costs one
# (bank x corpus) matmul plus encoding the bank (~5 s for 3,000 captions); set
# to 0 to require an explicit `python -m app.ml.hubness` instead.
HUBNESS_AUTOBUILD = os.environ.get("CVDE_HUBNESS_AUTOBUILD", "1") not in ("0", "false")

# A query term matching this fraction of images or more is too common for
# keyword ranking to discriminate, and the UI says so. Calibrated on Flickr8k,
# where the most common content word ("dog") matches 22.6% of images and no
# content word reaches FTS5's 50% IDF clamp.
DF_WARN_FRACTION = float(os.environ.get("CVDE_DF_WARN_FRACTION", "0.20"))

# Zero-shot attributes: how decisive the winning label must be before it is
# recorded at all. Gated on the top-1 minus top-2 margin rather than on the
# winner's probability, because probability is not comparable across groups of
# different sizes — `setting` has two labels, so its argmax is >= 0.5 by
# construction and no probability floor could ever reject one, while
# `environment` has seven and a 0.30 winner may be a coin toss. A margin means
# the same thing in both. Below it, the group is left unlabelled and reported
# as abstained rather than guessed.
ATTR_MIN_MARGIN = float(os.environ.get("CVDE_ATTR_MIN_MARGIN", "0.10"))

# Autonomous UI QA -----------------------------------------------------------
# Screenshots, decks and reports from a QA sweep are build artifacts, so they
# live under DATA_DIR (gitignored) and are served read-only like the thumbnails.
QA_DIR = DATA_DIR / "qa"

# What the QA browser drives. `localhost` and not `127.0.0.1`: the Vite dev
# server listens on IPv6 only, so the numeric form connects to nothing.
QA_BASE_URL = os.environ.get("CVDE_QA_BASE_URL", "http://localhost:5173")
QA_API_URL = os.environ.get("CVDE_QA_API_URL", "http://localhost:8000")

# Prefix the report puts in front of artifact paths. The default assumes
# StaticFiles is mounted on /media/qa alongside the image mounts; the router
# also serves the same files under /api/qa/artifact, so pointing this at that
# prefix works without a mount.
QA_URL_PREFIX = os.environ.get("CVDE_QA_URL_PREFIX", "/media/qa")

# Misc -----------------------------------------------------------------------
THUMB_SIZE = 320
DUPLICATE_THRESHOLD = float(os.environ.get("CVDE_DUP_THRESHOLD", "0.95"))


def ensure_dirs() -> None:
    for d in (DATA_DIR, IMAGES_DIR, THUMBS_DIR, EMB_DIR, CACHE_DIR, QA_DIR,
              REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
