"""Test isolation: point the app at a fresh temp data dir BEFORE any test
module imports `app.*` (conftest is imported first, so this is the one place
where the env override is guaranteed to precede `app.config` evaluation)."""
import os
import tempfile

os.environ["CVDE_DATA_DIR"] = tempfile.mkdtemp(prefix="cvde-test-")
