"""Test isolation: point the app at a fresh temp data dir BEFORE any test
module imports `app.*` (conftest is imported first, so this is the one place
where the env override is guaranteed to precede `app.config` evaluation)."""
import os
import tempfile

os.environ["CVDE_DATA_DIR"] = tempfile.mkdtemp(prefix="cvde-test-")
# Pin the retrieval provider: the suite's expectations were written against the
# SigLIP stack and must not change because a Qwen index exists on the dev
# machine. Provider resolution has its own dedicated tests (test_providers.py).
os.environ["CVDE_EMBED_PROVIDER"] = "siglip2"
