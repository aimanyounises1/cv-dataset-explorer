#!/usr/bin/env bash
# Backbone swap: SigLIP2-base-patch16-256  ->  SigLIP2-so400m-patch14-384
#
# Re-embeds the corpus with the stronger backbone and runs the dependent
# analysis in the same manifest-committing ingest. Everything is env-var
# driven — no code changes. Runs on the Mac (MPS); expect ~1–2 hours total,
# dominated by embedding 8,000 images through a 1.1B-param vision tower.
#
# The db (captions, splits, YOUR CURATION TAGS) is preserved: ingest re-embeds
# but skips the download/db stages that are already done. Backups are taken
# anyway — restoring them (see bottom) undoes the whole swap.
#
# IMPORTANT: CVDE_EMBED_MODEL must be set BOTH here and in the shell that
# serves the API afterwards. The server encodes queries live; serving a
# so400m index with base-encoded queries cannot work. The manifest guard refuses
# that mismatch and reports the rebuild command instead of serving wrong ranks.
set -euo pipefail

MODEL="google/siglip2-so400m-patch14-384"
cd "$(dirname "$0")/../backend"
PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing backend virtualenv. Follow the README setup first." >&2
  exit 1
fi

DATA_DIR="$("${PYTHON}" -c \
  'from app import config; print(config.DATA_DIR.expanduser().resolve())')"
DB_PATH="${DATA_DIR}/explorer.db"
INDEX_DIR="${DATA_DIR}/embeddings"
if [[ ! -f "${DB_PATH}" ]]; then
  echo "Missing corpus database: ${DB_PATH}" >&2
  exit 1
fi
if [[ ! -d "${INDEX_DIR}" ]]; then
  echo "Missing embedding directory: ${INDEX_DIR}" >&2
  exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="${DATA_DIR}/backup_base_backbone_${STAMP}"
echo "==> Backing up current embeddings + db to ${BACKUP}"
mkdir -p "${BACKUP}"
# SQLite runs in WAL mode, so copying explorer.db alone can omit committed
# curation that still lives in explorer.db-wal. The supported backup API takes
# one consistent snapshot while a local API continues to read or write.
"${PYTHON}" - "${DB_PATH}" "${BACKUP}/explorer.db" <<'PY'
import sqlite3
import sys

source = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
destination = sqlite3.connect(sys.argv[2])
try:
    source.backup(destination)
finally:
    destination.close()
    source.close()
PY
shopt -s nullglob
INDEX_ARTIFACTS=("${INDEX_DIR}"/*.npy "${INDEX_DIR}"/*.json)
if ((${#INDEX_ARTIFACTS[@]})); then
  cp "${INDEX_ARTIFACTS[@]}" "${BACKUP}/"
fi
if [[ -d "${DATA_DIR}/cache" ]]; then
  cp -r "${DATA_DIR}/cache" "${BACKUP}/cache"
fi

export CVDE_EMBED_MODEL="${MODEL}"
# so400m at 384px is memory-hungrier than base at 256px; a smaller batch keeps
# 16GB Macs out of MPS OOM. Raise it if you have headroom.
export CVDE_EMBED_BATCH="${CVDE_EMBED_BATCH:-16}"

echo "==> Rebuilding the bound SigLIP index and analysis with ${MODEL}"
"${PYTHON}" -m app.ingest --provider siglip2

echo "==> Done. Now serve with the SAME env var:"
echo
echo "    export CVDE_EMBED_MODEL=${MODEL}"
echo "    .venv/bin/python -m uvicorn app.main:app --port 8000"
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
echo "    rm -f '${DB_PATH}-wal' '${DB_PATH}-shm'"
echo "    cp '${BACKUP}/explorer.db' '${DB_PATH}'"
echo "    cp '${BACKUP}'/*.npy '${BACKUP}'/*.json '${INDEX_DIR}/'"
echo "    and serve without CVDE_EMBED_MODEL set."
