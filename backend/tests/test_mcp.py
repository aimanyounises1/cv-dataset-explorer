"""The local MCP server at POST /mcp: JSON-RPC handshake, tool listing,
invocation through the same service layer as the REST API, and the mutation
boundary — propose_tag returns a token and provably writes nothing.

    cd backend && pytest tests/test_mcp.py
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app


def rpc(client, method, params=None, id_=1):
    return client.post("/mcp", json={"jsonrpc": "2.0", "id": id_,
                                     "method": method,
                                     "params": params or {}})


@pytest.fixture(scope="module")
def ctx():
    conn = db.connect()
    db.init_db(conn)
    sids = []
    for i in range(3):
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
            "VALUES ('flickr8k', ?, 'train', 320, 240, 1)", (f"mcp_{i}.jpg",))
        sids.append(cur.lastrowid)
        text = f"mcp probe caption about a striped zeppelin {i}"
        ccur = conn.execute(
            "INSERT INTO captions(sample_id, idx, text, agreement) "
            "VALUES (?, 0, ?, ?)", (sids[-1], text, 0.05 + i / 100))
        conn.execute(
            "INSERT INTO captions_fts(rowid, text) VALUES (?, ?)",
            (ccur.lastrowid, text))
    conn.commit()
    with TestClient(app) as client:
        yield client, sids
    for sid in sids:
        conn.execute("DELETE FROM captions WHERE sample_id = ?", (sid,))
        conn.execute("DELETE FROM samples WHERE id = ?", (sid,))
    conn.execute("DELETE FROM activity_events WHERE kind = 'mcp_tag_proposal'")
    conn.commit()
    conn.close()


def test_handshake(ctx):
    client, _ = ctx
    r = rpc(client, "initialize", {"protocolVersion": "2025-06-18",
                                   "capabilities": {},
                                   "clientInfo": {"name": "probe"}})
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["protocolVersion"] and result["serverInfo"]["name"]
    assert "tools" in result["capabilities"]
    # the initialized notification is acknowledged with 202 and no body
    n = ctx[0].post("/mcp", json={"jsonrpc": "2.0",
                                  "method": "notifications/initialized"})
    assert n.status_code == 202


def test_tools_list_names_schemas_and_readonly_hints(ctx):
    client, _ = ctx
    tools = rpc(client, "tools/list").json()["result"]["tools"]
    byname = {t["name"]: t for t in tools}
    assert set(byname) == {"search_images", "get_sample", "find_similar",
                           "dataset_stats", "audit_captions", "get_album",
                           "propose_tag"}
    for t in tools:
        assert t["inputSchema"]["type"] == "object"
    assert byname["propose_tag"]["annotations"]["readOnlyHint"] is False
    assert byname["search_images"]["annotations"]["readOnlyHint"] is True


def test_search_tool_uses_the_real_service_layer(ctx):
    client, sids = ctx
    r = rpc(client, "tools/call",
            {"name": "search_images",
             "arguments": {"query": "striped zeppelin", "mode": "keyword"}})
    payload = json.loads(r.json()["result"]["content"][0]["text"])
    assert payload["mode_used"] == "keyword"
    assert any(c["sample_id"] in sids for c in payload["results"])


def test_get_sample_and_audit(ctx):
    client, sids = ctx
    detail = json.loads(rpc(client, "tools/call",
        {"name": "get_sample", "arguments": {"sample_id": sids[0]}}
    ).json()["result"]["content"][0]["text"])
    assert detail["sample_id"] == sids[0] and len(detail["captions"]) == 1
    audit = json.loads(rpc(client, "tools/call",
        {"name": "audit_captions", "arguments": {"limit": 3}}
    ).json()["result"]["content"][0]["text"])
    assert audit["captions"][0]["agreement"] <= audit["captions"][-1]["agreement"]


def test_propose_tag_returns_token_and_writes_nothing(ctx):
    client, sids = ctx
    r = rpc(client, "tools/call",
            {"name": "propose_tag",
             "arguments": {"sample_ids": [sids[0], 999_999_999],
                           "tag": "MCP-Probe", "reason": "test"}})
    payload = json.loads(r.json()["result"]["content"][0]["text"])
    assert payload["proposal_token"].startswith("mcp-")
    assert payload["status"] == "awaiting-approval"
    assert payload["candidates"] == 1
    conn = db.connect()
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM sample_tags WHERE sample_id = ?",
            (sids[0],)).fetchone()[0] == 0, "a proposal must write no tags"
        assert conn.execute(
            "SELECT COUNT(*) FROM tags WHERE name = 'mcp-probe'"
        ).fetchone()[0] == 0
        row = conn.execute(
            "SELECT payload FROM activity_events WHERE kind = 'mcp_tag_proposal' "
            "ORDER BY id DESC LIMIT 1").fetchone()
        assert row is not None
        assert json.loads(row["payload"])["token"] == payload["proposal_token"]
    finally:
        conn.close()


def test_protocol_errors(ctx):
    client, _ = ctx
    assert rpc(client, "no/such/method").json()["error"]["code"] == -32601
    bad = client.post("/mcp", content=b"{not json",
                      headers={"Content-Type": "application/json"})
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == -32700
    unknown = rpc(client, "tools/call", {"name": "rm_rf", "arguments": {}})
    assert unknown.json()["error"]["code"] == -32602


def test_get_mcp_is_a_signpost(ctx):
    client, _ = ctx
    info = client.get("/mcp").json()
    assert info["endpoint"] == "POST /mcp" and "propose_tag" in info["tools"]
