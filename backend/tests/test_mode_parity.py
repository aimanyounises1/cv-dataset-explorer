"""Every search mode the gallery can produce must survive into export.

`mode` is validated by a regex on two separate endpoints, and they once
drifted: a mode was added to `/api/search` and not to `/api/export`, so the
gallery happily rendered a result set while the CSV / JSONL / JSON buttons
sitting beside it returned 422. A mode the user can select and cannot take
away is a broken button they can see.

The patterns are read out of the OpenAPI schema rather than hardcoded here, so
adding a fifth mode to search without adding it to export fails this test instead
of shipping.
"""
import re

import pytest
from fastapi.testclient import TestClient

from app.main import app

MODE_PATH = "/api/search"
EXPORT_PATH = "/api/export"


def _mode_pattern(schema: dict, path: str) -> str:
    for param in schema["paths"][path]["get"]["parameters"]:
        if param["name"] == "mode":
            s = param["schema"]
            # Optional/defaulted params may be wrapped in anyOf.
            for cand in [s, *s.get("anyOf", [])]:
                if "pattern" in cand:
                    return cand["pattern"]
    raise AssertionError(f"no `mode` pattern on GET {path}")


@pytest.fixture(scope="module")
def schema():
    return app.openapi()


def test_search_and_export_accept_the_same_modes(schema):
    assert _mode_pattern(schema, MODE_PATH) == _mode_pattern(schema, EXPORT_PATH), (
        "GET /api/search and GET /api/export disagree about which modes exist; "
        "a mode the gallery can produce would 422 on the export button")


def test_removed_boosted_mode_stays_removed(schema):
    """PRISM/boosted was removed deliberately; a request for it must 422, not
    silently rank as something else."""
    assert "boosted" not in _mode_pattern(schema, MODE_PATH)
    r = TestClient(app).get("/api/search", params={"q": "dog", "mode": "boosted"})
    assert r.status_code == 422


def test_every_declared_mode_is_actually_accepted(schema):
    """The pattern is not the contract on its own — the handler has to take it.

    Runs against an empty corpus on purpose: the modes that need embeddings must
    degrade with a message rather than reject the request, so a 200 here is the
    assertion and the result count is irrelevant.
    """
    modes = re.findall(r"\w+", _mode_pattern(schema, MODE_PATH).strip("^$()"))
    assert set(modes) == {"semantic", "keyword", "hybrid"}, f"parsed {modes}"

    with TestClient(app) as client:
        for mode in modes:
            for path, params in (
                (MODE_PATH, {"q": "dog", "mode": mode, "top_k": 5}),
                (EXPORT_PATH, {"q": "dog", "mode": mode, "format": "csv"}),
            ):
                r = client.get(path, params=params)
                assert r.status_code == 200, (
                    f"GET {path} rejected declared mode {mode!r}: "
                    f"{r.status_code} {r.text[:200]}")
