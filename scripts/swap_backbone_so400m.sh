#!/usr/bin/env bash
# Backbone swap: SigLIP2-base-patch16-256  ->  SigLIP2-so400m-patch14-384
#
# Re-embeds the corpus with the stronger backbone and re-runs analysis.
# Everything is env-var
# driven — no code changes. Runs on the Mac (MPS); expect ~1–2 hours total,
# dominated by embedding 8,000 images through a 1.1B-param vision tower.
#
# The db (captions, splits, YOUR CURATION TAGS) is preserved: ingest re-embeds
# but skips the download/db stages that are already done. Backups are taken
# anyway — restoring them (see bottom) undoes the whole swap.
#
# IMPORTANT: CVDE_EMBED_MODEL must be set BOTH here and in the shell that
# serves the API afterwards. The server encodes queries live; serving a
# so400m index with base-encoded queries cannot work (the eval caches
# self-protect, but semantic search would 500).
set -euo pipefail

MODEL="google/siglip2-so400m-patch14-384"
cd "$(dirname "$0")/../backend"

STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="data/backup_base_backbone_${STAMP}"
echo "==> Backing up current embeddings + db to ${BACKUP}"
mkdir -p "${BACKUP}"
cp data/explorer.db "${BACKUP}/"
cp data/embeddings/*.npy data/embeddings/*.json "${BACKUP}/" 2>/dev/null || true
cp -r data/cache "${BACKUP}/cache" 2>/dev/null || true

export CVDE_EMBED_MODEL="${MODEL}"
# so400m at 384px is memory-hungrier than base at 256px; a smaller batch keeps
# 16GB Macs out of MPS OOM. Raise it if you have headroom.
export CVDE_EMBED_BATCH="${CVDE_EMBED_BATCH:-16}"

echo "==> [1/3] Re-embedding images with ${MODEL} (the long step)"
python -m app.ingest

echo "==> [2/3] Re-embedding captions + agreement/attributes/axes/UMAP"
python -m app.analyze

echo "==> [3/3] Done. Now serve with the SAME env var:"
echo
echo "    export CVDE_EMBED_MODEL=${MODEL}"
echo "    uvicorn app.main:app --port 8000"
echo
echo "    (or, against a running server: curl -X POST localhost:8000/api/admin/reload)"
echo
echo "Then compare accuracy:"
echo "  - The ladder above prints new-backbone A0 (zero-shot) vs A1 (trained):"
echo "    old numbers were A0 49.4% / A1 51.6% R@1 on the 8k pool."
echo "  - The app's Benchmark page re-runs automatically (caches key on the"
echo "    embedding files) and shows the paired test-split rows."
echo
echo "To UNDO the swap: stop the API, then"
echo "    cp ${BACKUP}/explorer.db data/ && cp ${BACKUP}/*.npy ${BACKUP}/*.json data/embeddings/"
echo "    and serve without CVDE_EMBED_MODEL set."
