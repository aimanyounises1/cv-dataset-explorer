"""Assistant endpoint: runs the Fugu-style LangGraph orchestration.

Optional stack (LangChain/LangGraph + a local Ollama chat model): missing
pieces produce a 503 with exact setup instructions, never a crash.
"""
import json
import logging
import sqlite3
import threading

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException

from .. import config
from ..schemas import ChatMessage, ChatResponse, ChatTraceStep
from .deps import first_captions, get_conn, row_to_card

router = APIRouter()
logger = logging.getLogger(__name__)

_graph = None
_lock = threading.Lock()

IMAGE_TOOLS = {"search_images", "find_similar", "suspect_captions",
               "rare_slice_examples", "get_sample_details"}
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


def _check_ollama() -> None:
    try:
        resp = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        if not any(m.startswith(config.CHAT_MODEL.split(":")[0]) for m in models):
            raise HTTPException(
                503, f"Ollama is running but model '{config.CHAT_MODEL}' is not "
                     f"pulled. Run: ollama pull {config.CHAT_MODEL}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            503, f"Ollama is not reachable at {config.OLLAMA_URL}. {SETUP_HELP}") from exc


@router.post("/chat", response_model=ChatResponse)
def chat(
    messages: list[ChatMessage] = Body(..., embed=True),
    conn: sqlite3.Connection = Depends(get_conn),
):
    if not messages or messages[-1].role != "user":
        raise HTTPException(400, "Last message must be from the user.")
    _check_ollama()
    graph = _get_graph()

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    lc_messages = [
        HumanMessage(m.content) if m.role == "user" else AIMessage(m.content)
        for m in messages[-10:]  # bound context
    ]
    try:
        result = graph.invoke({"messages": lc_messages, "route": "", "retries": 0},
                              config={"recursion_limit": 40})
    except Exception as exc:
        logger.exception("Agent run failed")
        raise HTTPException(500, f"Assistant run failed: {exc}") from exc

    new_messages = result["messages"][len(lc_messages):]

    # Trace: every tool call the specialists made, in order.
    trace, sample_ids, current_agent = [], [], "assistant"
    for msg in new_messages:
        name = getattr(msg, "name", None)
        if isinstance(msg, AIMessage):
            if name in ("retrieval", "insights", "orchestrator", "synthesizer"):
                current_agent = name
            for call in getattr(msg, "tool_calls", None) or []:
                trace.append(ChatTraceStep(
                    agent=current_agent, tool=call["name"],
                    input=json.dumps(call.get("args", {}))[:200]))
        elif isinstance(msg, ToolMessage) and getattr(msg, "name", "") in IMAGE_TOOLS:
            try:
                payload = json.loads(msg.content)
                sample_ids += [int(s) for s in payload.get("sample_ids", [])]
            except Exception:
                pass

    reply = ""
    for msg in reversed(new_messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            reply = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # Attach renderable cards for any samples the tools surfaced.
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

    return ChatResponse(reply=reply or "(no reply produced)", samples=cards, trace=trace)


@router.get("/chat/status")
def chat_status():
    """Availability probe so the UI can show setup instructions proactively."""
    try:
        _check_ollama()
        from ..agent import graph as _  # noqa: F401  (import check only)
        return {"available": True, "model": config.CHAT_MODEL}
    except HTTPException as exc:
        return {"available": False, "model": config.CHAT_MODEL, "reason": exc.detail}
    except ImportError:
        return {"available": False, "model": config.CHAT_MODEL, "reason": SETUP_HELP}
