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
EMBED_MODEL = os.environ.get("CVDE_EMBED_MODEL", "google/siglip2-base-patch16-256")
EMBED_BATCH_SIZE = int(os.environ.get("CVDE_EMBED_BATCH", "32"))

# Optional local VLM enrichment + assistant (via Ollama) ---------------------
OLLAMA_URL = os.environ.get("CVDE_OLLAMA_URL", "http://localhost:11434")
VLM_MODEL = os.environ.get("CVDE_VLM_MODEL", "qwen2.5vl:7b")
CHAT_MODEL = os.environ.get("CVDE_CHAT_MODEL", "qwen3:8b")  # needs tool calling

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

# A query term matching this fraction of images or more is too common for
# keyword ranking to discriminate, and the UI says so. Calibrated on Flickr8k,
# where the most common content word ("dog") matches 22.6% of images and no
# content word reaches FTS5's 50% IDF clamp — see docs/REQUIREMENTS.md §13.14.
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

# Misc -----------------------------------------------------------------------
THUMB_SIZE = 320
DUPLICATE_THRESHOLD = float(os.environ.get("CVDE_DUP_THRESHOLD", "0.95"))


def ensure_dirs() -> None:
    for d in (DATA_DIR, IMAGES_DIR, THUMBS_DIR, EMB_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
