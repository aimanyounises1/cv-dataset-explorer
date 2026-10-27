#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
PYTHON_BIN=${PYTHON_BIN:-backend/.venv/bin/python}
PYTHON_BIN=$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')
(cd backend && "$PYTHON_BIN" -m ruff check app tests ../scripts/bench_providers.py)
(cd backend && "$PYTHON_BIN" -m pytest -q)
(cd frontend && npm run build)
"$PYTHON_BIN" scripts/check_links.py
git diff --check
echo "docs-first verification: PASS"
