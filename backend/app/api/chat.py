"""Assistant endpoint: runs the multi-agent LangGraph orchestration.

Beyond invoking the graph, this endpoint does three things the graph should not
know about:

* **Collects render blocks.** Tools return `{"blocks": [...]}`; the reply carries
  them in the order produced, so the chat transcript can render charts, tables and
  diagrams rather than paragraphs describing them. Blocks are validated here
  against the block union, so a malformed one fails as a 500 with a traceback
  instead of as an empty box in the browser.
* **Persists reports.** A report is worth more than the turn that produced it, so
  it is written to disk and the block gains download URLs. The tool that builds a
  report cannot do this: it does not know where it will be served from.
* **Reports partial failure.** Which lanes ran and which died reaches the client,
  because an answer assembled from half the specialists should look like one.

Optional stack (LangChain/LangGraph + a local Ollama chat model): missing pieces
produce a 503 with exact setup instructions, never a crash.
"""
import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import FileResponse

from .. import config
from ..schemas import ChatMessage, ChatResponse, ChatTraceStep
from .deps import first_captions, get_conn, row_to_card

router = APIRouter()
logger = logging.getLogger(__name__)

_graph = None
_lock = threading.Lock()

IMAGE_TOOLS = {"search_images", "find_similar", "suspect_captions", "inspect_album",
               "rare_slice_examples", "get_sample_details", "show_images"}
SETUP_HELP = (
    "The assistant needs the optional agent stack: "
    "1) pip install -r requirements-agent.txt  "
    "2) install Ollama (https://ollama.com) and run `ollama pull "
    f"{config.CHAT_MODEL}`  3) retry."
)


def _get_graph():
    global _graph
    if _graph is None:
        with _lock:
            if _graph is None:
                try:
                    from ..agent.graph import build_graph
                except ImportError as exc:
                    raise HTTPException(503, f"{SETUP_HELP} (import error: {exc})") from exc
                _graph = build_graph()
    return _graph


async def _check_ollama() -> None:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{config.OLLAMA_URL}/api/tags")
        resp.raise_for_status()
        models = {m.get("name", "") for m in resp.json().get("models", [])}
        wanted = config.CHAT_MODEL
        aliases = {wanted}
        if ":" not in wanted:
            aliases.add(f"{wanted}:latest")
        if models.isdisjoint(aliases):
            raise HTTPException(
                503, f"Ollama is running but model '{config.CHAT_MODEL}' is not "
                     f"pulled. Run: ollama pull {config.CHAT_MODEL}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            503, f"Ollama is not reachable at {config.OLLAMA_URL}. {SETUP_HELP}") from exc


def _validated_blocks(raw: list[dict]) -> list[dict]:
    """Check every collected block against the block union.

    Validation happens at the boundary rather than being trusted from the tools,
    because a block that fails to render is invisible: the browser draws nothing
    and the user reads it as "the assistant had no answer". Failing loudly here
    turns that into a traceback with the offending payload in it.
    """
    from pydantic import TypeAdapter

    from ..agent.blocks import Block

    adapter = TypeAdapter(list[Block])
    return [b.model_dump(mode="json") for b in adapter.validate_python(raw)]


def _dedupe_blocks(raw: list[dict]) -> list[dict]:
    """Collapse blocks that are byte-identical, keeping the first.

    A local model asked to "plot the splits" will sometimes call the same tool
    twice in one turn, and two parallel lanes can independently chart the same
    thing — both were observed. Either way the user gets the same picture twice,
    which reads as a rendering bug. Deduplicating on the payload rather than on
    the tool name is what makes the second case work: two lanes calling different
    tools that happen to produce the same chart still collapse to one.
    """
    seen, out = set(), []
    for block in raw:
        key = json.dumps(block, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(block)
    return out


def _persist_report(block: dict) -> dict:
    """Write a report block to disk and attach download URLs.

    Both formats are written: JSON because it is the block itself and can be
    re-rendered exactly, Markdown because a report that can only be read inside
    the tool that made it is not much of a deliverable.
    """
    from ..agent.report_md import report_to_markdown

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rid = uuid.uuid4().hex[:12]
    (config.REPORTS_DIR / f"{rid}.json").write_text(
        json.dumps(block, indent=2), encoding="utf-8")
    (config.REPORTS_DIR / f"{rid}.md").write_text(
        report_to_markdown(block), encoding="utf-8")
    return {**block,
            "download_json": f"/api/reports/{rid}.json",
            "download_md": f"/api/reports/{rid}.md"}


@router.get("/reports/{name}")
def download_report(name: str):
    """Serve a generated report. Names are generated hex ids, and the suffix is
    whitelisted, so this cannot be walked out of the reports directory."""
    if not name.replace(".", "").isalnum() or not name.endswith((".json", ".md")):
        raise HTTPException(400, "Bad report name.")
    path = config.REPORTS_DIR / name
    if not path.is_file():
        raise HTTPException(404, "Report not found — reports do not survive a "
                                 "data directory being cleared.")
    media = "application/json" if name.endswith(".json") else "text/markdown"
    return FileResponse(path, media_type=media, filename=f"cvde-report-{name}")


@router.get("/agent/graph")
def agent_topology():
    """The live agent topology, read from the registry.

    The assistant page draws this rather than shipping a picture of the
    architecture, so the diagram cannot drift from the code that runs.
    """
    try:
        from ..agent import registry
    except ImportError as exc:
        raise HTTPException(503, f"{SETUP_HELP} (import error: {exc})") from exc
    return {
        "specialists": [
            {"name": s.name, "summary": s.summary, "cost": s.cost,
             "tools": [getattr(t, "name", str(t)) for t in s.tools]}
            for s in registry.SPECIALISTS
        ],
        "max_parallel_lanes": registry.MAX_PARALLEL_LANES,
        "model": config.CHAT_MODEL,
        "lane_timeout_s": config.AGENT_LANE_TIMEOUT,
        # Both bounds, because quoting the lane alone understates what a
        # waiting person is promised: the lane bound is per-specialist, the
        # turn bound is the one they actually experience. The capabilities doc
        # is generated from this payload, so a bound it cannot read is a bound
        # the documentation cannot state.
        "turn_budget_s": config.AGENT_TURN_BUDGET,
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(
    messages: list[ChatMessage] = Body(..., embed=True),
    conn: sqlite3.Connection = Depends(get_conn),
):
    if not messages or messages[-1].role != "user":
        raise HTTPException(400, "Last message must be from the user.")
    started = time.monotonic()
    deadline = asyncio.get_running_loop().time() + config.AGENT_TURN_BUDGET
    try:
        async with asyncio.timeout_at(deadline):
            await _check_ollama()
            graph = _get_graph()

            from langchain_core.messages import AIMessage, HumanMessage

            lc_messages = [
                HumanMessage(m.content) if m.role == "user" else AIMessage(m.content)
                for m in messages[-10:]  # bound context
            ]
            result = await graph.ainvoke(
                {"messages": lc_messages, "routes": [], "retries": 0,
                 "claims_to_verify": [], "lanes_ok": [], "lanes_failed": []},
                config={"recursion_limit": 40},
            )
            elapsed = time.monotonic() - started
            return _build_response(conn, lc_messages, result, elapsed)
    except HTTPException:
        raise
    except TimeoutError as exc:
        logger.warning("Assistant exceeded the %.0fs turn timeout",
                       config.AGENT_TURN_BUDGET)
        raise HTTPException(
            504,
            f"Assistant exceeded the {config.AGENT_TURN_BUDGET:.0f}s turn timeout.",
        ) from exc
    except Exception as exc:
        logger.exception("Agent run failed")
        raise HTTPException(500, f"Assistant run failed: {exc}") from exc


# Human phrasing per graph node for the live trace. A node missing here still
# streams — as its own name — so a new specialist can never silence the trace.
_STEP_LABELS = {
    "orchestrate": "Reading the request and choosing specialist lanes",
    "retrieval": "Searching and inspecting the corpus",
    "insights": "Analyzing dataset signals",
    "visualization": "Building a visual answer",
    "qa": "Driving the application in Chrome",
    "synthesize": "Quality-gating the answer",
}


@router.post("/chat/stream")
async def chat_stream(
    messages: list[ChatMessage] = Body(..., embed=True),
    conn: sqlite3.Connection = Depends(get_conn),
):
    """The same orchestration as POST /chat, with the steps streamed as they
    happen: NDJSON lines `{"type":"step",...}` from LangGraph's own update
    stream — real node transitions, never a staged animation — then one
    `{"type":"final","response":{...}}` carrying the exact POST /chat payload.
    """
    if not messages or messages[-1].role != "user":
        raise HTTPException(400, "Last message must be from the user.")
    started = time.monotonic()
    deadline = asyncio.get_running_loop().time() + config.AGENT_TURN_BUDGET
    try:
        async with asyncio.timeout_at(deadline):
            await _check_ollama()
            graph = _get_graph()

            from langchain_core.messages import AIMessage, HumanMessage

            lc_messages = [
                HumanMessage(m.content) if m.role == "user" else AIMessage(m.content)
                for m in messages[-10:]
            ]
            state = {"messages": lc_messages, "routes": [], "retries": 0,
                     "claims_to_verify": [], "lanes_ok": [], "lanes_failed": []}
    except HTTPException:
        raise
    except TimeoutError as exc:
        raise HTTPException(
            504,
            f"Assistant exceeded the {config.AGENT_TURN_BUDGET:.0f}s turn timeout.",
        ) from exc

    async def gen():
        emit = lambda d: json.dumps(d, separators=(",", ":")) + "\n"  # noqa: E731
        final_state = None
        try:
            async with asyncio.timeout_at(deadline):
                async for mode, chunk in graph.astream(
                        state,
                        config={"recursion_limit": 40},
                        stream_mode=["updates", "values"]):
                    if mode == "values":
                        final_state = chunk
                        continue
                    for node in chunk:
                        yield emit({"type": "step", "node": node,
                                    "label": _STEP_LABELS.get(node, node),
                                    "t": round(time.monotonic() - started, 2)})
                if final_state is None:
                    yield emit({"type": "error",
                                "detail": "The graph produced no final state."})
                    return
                resp = _build_response(
                    conn, lc_messages, final_state, time.monotonic() - started
                )
                yield emit({
                    "type": "final",
                    "response": resp.model_dump(mode="json"),
                })
        except TimeoutError:
            logger.warning("Streaming assistant exceeded the %.0fs turn timeout",
                           config.AGENT_TURN_BUDGET)
            yield emit({
                "type": "error",
                "detail": (
                    f"Assistant exceeded the "
                    f"{config.AGENT_TURN_BUDGET:.0f}s turn timeout."
                ),
            })
            return
        except Exception as exc:                          # noqa: BLE001
            logger.exception("Streaming agent run failed")
            yield emit({"type": "error",
                        "detail": f"Assistant run failed: {exc}"})
            return

    from fastapi.responses import StreamingResponse

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _build_response(conn, lc_messages, result, elapsed) -> ChatResponse:
    """Everything between a finished graph state and the wire response:
    trace walk, block validation, report persistence, sample cards, reply
    extraction. One implementation, shared by the blocking endpoint and the
    streaming one — two copies would drift on the first change."""
    from langchain_core.messages import AIMessage, ToolMessage

    new_messages = result["messages"][len(lc_messages):]

    # Trace + collected payloads. Walked in order so blocks appear in the reply
    # in the order the specialists produced them, which is the order the user's
    # request implied.
    trace: list[ChatTraceStep] = []
    sample_ids: list[int] = []
    raw_blocks: list[dict] = []
    current_agent = "assistant"
    known_lanes = {"orchestrator", "synthesizer", "final"}
    try:
        from ..agent import registry

        known_lanes |= {s.name for s in registry.SPECIALISTS}
    except ImportError:                                   # pragma: no cover
        pass

    for msg in new_messages:
        name = getattr(msg, "name", None)
        if isinstance(msg, AIMessage):
            if name in known_lanes:
                current_agent = name
            for call in getattr(msg, "tool_calls", None) or []:
                trace.append(ChatTraceStep(
                    agent=current_agent, tool=call["name"],
                    input=json.dumps(call.get("args", {}))[:200]))
        elif isinstance(msg, ToolMessage):
            payload = _tool_payload(msg)
            if payload is None:
                continue
            if getattr(msg, "name", "") in IMAGE_TOOLS:
                sample_ids += [int(s) for s in payload.get("sample_ids", []) or []]
            for block in payload.get("blocks", []) or []:
                raw_blocks.append(block)

    reply = ""
    for msg in reversed(new_messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            raw = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Retry feedback is a typed SynthesisDecision field and therefore
            # never enters message content. ChatOllama is configured with its
            # documented `reasoning=False` mode, so content is already the
            # user-facing answer rather than a reasoning-tag transport.
            cleaned = raw.strip()
            # Internal notes are addressed to the graph, not to the reader:
            # `[orchestrator → retrieval] …`, `[retrieval ran out of time …]`,
            # `[quality gate] …`. They are AIMessages with content and no tool
            # calls, so without this they satisfy the loop and become the answer
            # — measured: "Show me the 12 images with the worst caption
            # agreement" returned the twelve cards and the reply
            # "[orchestrator → retrieval]". `_fallback_answer` in the graph has
            # skipped bracketed content all along; this path never did.
            if cleaned.startswith("["):
                continue
            reply = cleaned
            if reply:
                break

    # Deduplicated before persisting, so a report produced twice writes one file.
    blocks = _validated_blocks(_dedupe_blocks(raw_blocks))
    blocks = [_persist_report(b) if b.get("kind") == "report" else b for b in blocks]

    # Attach renderable cards for any samples the tools surfaced. An images block
    # already carries its own ids, so this is the fallback for the retrieval tools
    # that predate blocks and simply report what they found.
    seen, ordered_ids = set(), []
    for sid in sample_ids:
        if sid not in seen:
            seen.add(sid)
            ordered_ids.append(sid)
    ordered_ids = ordered_ids[:24]
    cards = []
    if ordered_ids:
        qmarks = ",".join("?" * len(ordered_ids))
        rows = {r["id"]: r for r in conn.execute(
            f"SELECT * FROM samples WHERE id IN ({qmarks})", ordered_ids)}
        caps = first_captions(conn, ordered_ids)
        cards = [row_to_card(rows[sid], caption=caps.get(sid))
                 for sid in ordered_ids if sid in rows]

    lanes = list(dict.fromkeys(result.get("lanes_ok") or []))
    failed = list(dict.fromkeys(result.get("lanes_failed") or []))
    if not reply:
        reply = ("The specialists produced no text answer."
                 + (f" Failed lanes: {', '.join(failed)}." if failed else ""))

    return ChatResponse(reply=reply, samples=cards, trace=trace, blocks=blocks,
                        lanes=lanes, lanes_failed=failed,
                        elapsed_s=round(elapsed, 2))


def _tool_payload(msg: Any) -> dict | None:
    """A tool's JSON output, or None when it did not return JSON.

    Tools are free to return prose — several do — so a parse failure is an
    expected outcome here, not an error to report.
    """
    content = getattr(msg, "content", None)
    if not isinstance(content, str) or not content.strip().startswith("{"):
        return None
    try:
        payload = json.loads(content)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


@router.get("/chat/status")
async def chat_status():
    """Availability probe so the UI can show setup instructions proactively."""
    try:
        await _check_ollama()
        from ..agent import registry
        return {"available": True, "model": config.CHAT_MODEL,
                "specialists": [s.name for s in registry.SPECIALISTS]}
    except HTTPException as exc:
        return {"available": False, "model": config.CHAT_MODEL, "reason": exc.detail}
    except ImportError:
        return {"available": False, "model": config.CHAT_MODEL, "reason": SETUP_HELP}
