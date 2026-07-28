# MCP server

The platform's tools, exposed to any local agent over the
[Model Context Protocol](https://modelcontextprotocol.io): JSON-RPC 2.0 at
`POST http://localhost:8000/mcp` (Streamable HTTP transport, stateless JSON
mode — plain JSON responses, no SSE stream). No SDK dependency; the endpoint
is ~250 lines over the same service layer the REST API and the in-app
assistant use, so every client sees identical answers.

## Tools

| Tool | Reads | Writes |
|---|---|---|
| `search_images` | ranked search, any mode, with `score_basis` and degradation message | — |
| `get_sample` | captions + agreement, tags, difficulty axes, metadata | — |
| `find_similar` | nearest neighbours in the ACTIVE embedding index | — |
| `list_annotations` | paginated saved rectangles and object masks, including label ancestry and mask URLs | — |
| `find_similar_to_annotation` | retrieval from one accepted object mask, with its exact score basis | — |
| `dataset_stats` | corpus counts, splits, active retrieval provider + fallback reason | — |
| `audit_captions` | lowest image–caption agreement first (ingest-time SigLIP scores) | — |
| `get_album` | album metadata + ordered member ids | — |

**Strictly read-only.** Every tool carries `readOnlyHint: true` and none can
mutate anything: an MCP client can search, inspect and audit this dataset,
never curate it. Curation — tagging, albums, annotations — is a human act
performed inside the app, where the assistant's own proposals also wait for a
click.

## Handshake, by hand

```bash
curl -s localhost:8000/mcp -H 'Content-Type: application/json' -d '{
  "jsonrpc": "2.0", "id": 1, "method": "initialize",
  "params": {"protocolVersion": "2025-06-18", "capabilities": {},
             "clientInfo": {"name": "curl"}}}'

curl -s localhost:8000/mcp -H 'Content-Type: application/json' \
  -d '{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}'

curl -s localhost:8000/mcp -H 'Content-Type: application/json' -d '{
  "jsonrpc": "2.0", "id": 3, "method": "tools/call",
  "params": {"name": "search_images",
             "arguments": {"query": "a dog jumping into water", "top_k": 3}}}'
```

## Connecting clients

Any MCP client that speaks Streamable HTTP connects with just the URL:

- **Claude Code / claude.ai clients**: add an HTTP MCP server with URL
  `http://localhost:8000/mcp`.
- **LangGraph / LangChain**: `pip install langchain-mcp-adapters`, then
  `MultiServerMCPClient({"cvde": {"transport": "streamable_http", "url":
  "http://localhost:8000/mcp"}})` yields the eight tools as LangChain tools.
  (The in-app assistant does NOT go through MCP — it calls the service layer
  directly; MCP is the door for agents living outside this process.)
- **Anything else**: POST the three requests above; notifications (requests
  without `id`) are acknowledged with `202`.

## Limits, honestly

Stateless JSON only — no SSE streaming, no sessions, no resumability; requests
over 64 KB are refused; batch requests are not supported. The endpoint binds
to the same local server as everything else and adds no authentication: it is
a local, single-user tool by the assignment's own constraint, and exposing it
beyond localhost is the operator's decision, not the default.

Tests: `backend/tests/test_mcp.py` (handshake, listing, invocation through the
real search stack, protocol errors, and the strictly-read-only guarantee).
