"""A local MCP server at POST /mcp — the platform's tools for ANY local agent.

Implements the Model Context Protocol's Streamable HTTP transport in its
stateless form: JSON-RPC 2.0 over plain POST, no SDK dependency, mounted at
/mcp (no /api prefix — MCP clients expect a root-ish endpoint). The tools call
the SAME service layer as the REST API and the in-app assistant, which is the
whole point: retrieval stays deterministic and testable, and every client —
REST, LangGraph, an external MCP client — sees identical answers.

Two hard rules, both load-bearing:

* **Read-only by construction.** Six of the seven tools only read. The one
  mutating intent, `propose_tag`, writes NOTHING to curation data: it
  validates the ids, records a proposal event in the activity trail, and
  returns a proposal token. The tag is applied only when a human approves it
  inside the app (the assistant's proposal flow or manual tagging) — an MCP
  client cannot curate the dataset by itself.
* **Honest degradation, same as everywhere.** Tools that need embeddings
  report the named reason instead of empty results dressed as answers.

See docs/MCP.md for the handshake, client configuration and examples.
"""
import json
import logging
import secrets
import sqlite3
from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from .. import config, db
from .deps import record_activity, thumb_url

logger = logging.getLogger(__name__)

router = APIRouter()

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "cv-dataset-explorer", "version": "1.0"}
MAX_BODY = 64 * 1024


# ------------------------------------------------------------------ tool impls
# Thin wrappers over the service layer. Deliberately NOT the agent's LangChain
# tools: those need the optional langchain stack, and the MCP surface must work
# on the light install.

def _cards(items) -> list[dict]:
    return [{"sample_id": it.id, "caption": it.caption, "score": it.score,
             "thumb_url": it.thumb_url} for it in items]


def _t_search(conn, args: dict) -> dict:
    from .search import run_search

    res = run_search(conn, str(args["query"]),
                     mode=str(args.get("mode", "hybrid")),
                     top_k=min(int(args.get("top_k", 8)), 30))
    return {"mode_used": res.mode_used, "score_basis": res.score_basis,
            "degraded": res.degraded, "message": res.message,
            "results": _cards(res.items)}


def _t_sample(conn, args: dict) -> dict:
    sid = int(args["sample_id"])
    row = conn.execute("SELECT * FROM samples WHERE id = ?", (sid,)).fetchone()
    if row is None:
        return {"error": f"sample {sid} does not exist"}
    captions = [{"text": r["text"], "agreement": r["agreement"]}
                for r in conn.execute(
                    "SELECT text, agreement FROM captions WHERE sample_id = ? "
                    "ORDER BY idx", (sid,))]
    tags = [r["name"] for r in conn.execute(
        "SELECT t.name FROM tags t JOIN sample_tags st ON st.tag_id = t.id "
        "WHERE st.sample_id = ?", (sid,))]
    return {"sample_id": sid, "filename": row["filename"], "split": row["split"],
            "width": row["width"], "height": row["height"],
            "captions": captions, "tags": tags,
            "axes": {a: row[a] for a in ("legibility", "rarity", "difficulty",
                                         "clutter")},
            "thumb_url": thumb_url(row["filename"])}


def _t_similar(conn, args: dict) -> dict:
    from ..ml.index import get_index

    index = get_index()
    if index is None:
        return {"error": "no embedding index — run `python -m app.ingest`"}
    sid = int(args["sample_id"])
    hits = index.similar_to(sid, top_k=min(int(args.get("top_k", 8)), 30))
    if not hits:
        return {"error": f"sample {sid} is not in the index"}
    qmarks = ",".join("?" * len(hits))
    rows = {r["id"]: r for r in conn.execute(
        f"SELECT id, filename FROM samples WHERE id IN ({qmarks})",
        [h[0] for h in hits])}
    return {"score_basis": "cosine", "results": [
        {"sample_id": i, "score": round(s, 4),
         "thumb_url": thumb_url(rows[i]["filename"]) if i in rows else None}
        for i, s in hits]}


def _t_stats(conn, args: dict) -> dict:
    from ..ml import providers

    st = providers.resolve()
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    splits = {r["split"]: r["n"] for r in conn.execute(
        "SELECT split, COUNT(*) AS n FROM samples GROUP BY split")}
    return {"total_samples": total,
            "total_captions": conn.execute(
                "SELECT COUNT(*) FROM captions").fetchone()[0],
            "splits": splits,
            "retrieval_provider": st.active, "retrieval_model": st.model_id,
            "retrieval_fallback_reason": st.fallback_reason}


def _t_audit_captions(conn, args: dict) -> dict:
    rows = conn.execute(
        "SELECT c.sample_id, c.text, c.agreement FROM captions c "
        "WHERE c.agreement IS NOT NULL ORDER BY c.agreement ASC LIMIT ?",
        (min(int(args.get("limit", 8)), 50),)).fetchall()
    if not rows:
        return {"error": "no agreement scores — run `python -m app.analyze`"}
    return {"note": "lowest image-caption agreement first (ingest-time SigLIP "
                    "cosine); low = likely annotation problem",
            "captions": [{"sample_id": r["sample_id"], "text": r["text"],
                          "agreement": round(r["agreement"], 4)} for r in rows]}


def _t_album(conn, args: dict) -> dict:
    aid = int(args["album_id"])
    row = conn.execute("SELECT * FROM albums WHERE id = ?", (aid,)).fetchone()
    if row is None:
        return {"error": f"album {aid} does not exist"}
    items = [r["sample_id"] for r in conn.execute(
        "SELECT sample_id FROM album_items WHERE album_id = ? "
        "ORDER BY position, sample_id", (aid,))]
    return {"album_id": aid, "name": row["name"], "summary": row["summary"],
            "category": row["category"], "origin": row["origin"],
            "item_count": len(items), "sample_ids": items}


def _t_propose_tag(conn, args: dict) -> dict:
    ids = [int(i) for i in list(args["sample_ids"])[:200]]
    tag = str(args["tag"]).strip().lower()
    if not tag or not ids:
        return {"error": "need a tag name and at least one sample id"}
    qmarks = ",".join("?" * len(ids))
    real = {r["id"] for r in conn.execute(
        f"SELECT id FROM samples WHERE id IN ({qmarks})", ids)}
    keep = [i for i in ids if i in real]
    if not keep:
        return {"error": "none of those sample ids exist — nothing proposed"}
    token = f"mcp-{secrets.token_hex(8)}"
    # A log entry, not a mutation: sample_tags is untouched until a human
    # applies the tag inside the app.
    record_activity(conn, "mcp_tag_proposal",
                    {"token": token, "tag": tag, "sample_ids": keep,
                     "reason": str(args.get("reason", ""))[:500],
                     "source": "mcp"})
    conn.commit()
    return {"proposal_token": token, "status": "awaiting-approval",
            "candidates": len(keep),
            "note": "Nothing was written. The proposal is visible in the "
                    "app's History; a human applies the tag there — this "
                    "token only identifies the request."}


TOOLS: dict[str, dict] = {
    "search_images": {
        "fn": _t_search, "readonly": True,
        "description": "Search the dataset by natural language. Modes: hybrid "
                       "(default), semantic, keyword.",
        "schema": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string"},
            "mode": {"type": "string",
                     "enum": ["hybrid", "semantic", "keyword"]},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 30}}}},
    "get_sample": {
        "fn": _t_sample, "readonly": True,
        "description": "One sample's captions, agreement scores, tags, "
                       "difficulty axes and metadata.",
        "schema": {"type": "object", "required": ["sample_id"], "properties": {
            "sample_id": {"type": "integer"}}}},
    "find_similar": {
        "fn": _t_similar, "readonly": True,
        "description": "Nearest neighbours of a sample in the active "
                       "embedding index.",
        "schema": {"type": "object", "required": ["sample_id"], "properties": {
            "sample_id": {"type": "integer"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 30}}}},
    "dataset_stats": {
        "fn": _t_stats, "readonly": True,
        "description": "Corpus counts, splits, and the ACTIVE retrieval "
                       "provider with its fallback reason.",
        "schema": {"type": "object", "properties": {}}},
    "audit_captions": {
        "fn": _t_audit_captions, "readonly": True,
        "description": "Captions least supported by their image — likely "
                       "annotation problems, lowest agreement first.",
        "schema": {"type": "object", "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 50}}}},
    "get_album": {
        "fn": _t_album, "readonly": True,
        "description": "One album's metadata and ordered member ids.",
        "schema": {"type": "object", "required": ["album_id"], "properties": {
            "album_id": {"type": "integer"}}}},
    "propose_tag": {
        "fn": _t_propose_tag, "readonly": False,
        "description": "PROPOSE tagging samples. Writes nothing: returns a "
                       "proposal token; a human approves inside the app.",
        "schema": {"type": "object", "required": ["sample_ids", "tag"],
                   "properties": {
                       "sample_ids": {"type": "array",
                                      "items": {"type": "integer"}},
                       "tag": {"type": "string"},
                       "reason": {"type": "string"}}}},
}


# --------------------------------------------------------------- JSON-RPC core

def _rpc_error(id_: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_,
            "error": {"code": code, "message": message}}


def _handle(payload: dict, conn: sqlite3.Connection) -> Optional[dict]:
    """One JSON-RPC request → response dict, or None for notifications."""
    method = payload.get("method")
    id_ = payload.get("id")
    params = payload.get("params") or {}
    if id_ is None:                       # notification: acknowledged, no body
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": id_, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
            "instructions": "Read-only dataset tools plus propose_tag, which "
                            "never writes without in-app human approval."}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": id_, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": id_, "result": {"tools": [
            {"name": name, "description": t["description"],
             "inputSchema": t["schema"],
             "annotations": {"readOnlyHint": t["readonly"]}}
            for name, t in TOOLS.items()]}}
    if method == "tools/call":
        name = params.get("name")
        tool = TOOLS.get(name or "")
        if tool is None:
            return _rpc_error(id_, -32602, f"unknown tool {name!r}")
        try:
            result = tool["fn"](conn, params.get("arguments") or {})
        except (KeyError, TypeError, ValueError) as exc:
            return _rpc_error(id_, -32602, f"bad arguments for {name}: {exc}")
        return {"jsonrpc": "2.0", "id": id_, "result": {
            "content": [{"type": "text", "text": json.dumps(result)}],
            "isError": "error" in result}}
    return _rpc_error(id_, -32601, f"method {method!r} not found")


@router.post("/mcp")
async def mcp_endpoint(request: Request):
    """Model Context Protocol endpoint (Streamable HTTP, stateless JSON mode).

    Speaks JSON-RPC 2.0: initialize, ping, tools/list, tools/call. Read-only
    tools over the same service layer as the REST API; the one mutating
    intent returns a proposal token and writes nothing. See docs/MCP.md.
    """
    body = await request.body()
    if len(body) > MAX_BODY:
        return JSONResponse(_rpc_error(None, -32600, "request too large"),
                            status_code=400)
    try:
        payload = json.loads(body)
    except ValueError:
        return JSONResponse(_rpc_error(None, -32700, "parse error"),
                            status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse(_rpc_error(None, -32600, "batch not supported"),
                            status_code=400)
    with db.get_db() as conn:
        out = _handle(payload, conn)
    if out is None:
        return Response(status_code=202)
    return JSONResponse(out)


@router.get("/mcp")
def mcp_info():
    """Human-facing hint; the protocol itself runs over POST."""
    return {"protocol": "Model Context Protocol (Streamable HTTP, stateless)",
            "endpoint": "POST /mcp", "tools": sorted(TOOLS),
            "docs": "docs/MCP.md",
            "note": "SSE streaming is not implemented — plain JSON responses."}
