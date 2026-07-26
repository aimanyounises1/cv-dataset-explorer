"""Multi-agent orchestration: one interface, several specialists, run in parallel.

    user → orchestrator ─┬→ retrieval      (search / similar / inspect / tag)
                         ├→ insights       (stats / coverage / caption QA)
                         ├→ visualization  (charts / diagrams / reports)
                         └→ qa             (drive the app, screenshot, report)
                                   ↓  (lanes run concurrently, results merged)
                             synthesizer ──(quality gate)──→ answer
                                   └───── one retry with feedback ─────┘

The specialists are not written here — they come from `registry.SPECIALISTS`, and
every part of this module is derived from that list: the nodes, the routing menu,
the fan-out edges, and what the synthesizer is told. See `registry` for why.

Three properties this graph has to hold, and where each is enforced:

* **Parallel, not merely concurrent-looking.** When the orchestrator selects two
  cheap lanes, LangGraph runs both nodes in one superstep on separate threads and
  merges their message updates through the `add_messages` reducer. Latency is the
  slower lane, not the sum.
* **A failed lane must not fail the turn.** Every lane is wrapped: an exception
  or a timeout becomes a recorded failure and the synthesizer is told which lanes
  died, so a partial answer says so instead of quietly omitting half the work.
* **Bounded.** Per-lane wall clock, a model-level request timeout, one retry, and
  a recursion cap. A local model that starts looping should cost a bounded amount
  of time, not the whole request.

All models run locally through Ollama. The deterministic search stack stays the
platform's retrieval engine — agents are consumers of the same service functions
the REST API uses, not a replacement for them.
"""
import json
import logging
import operator
import re
import threading
import time
from typing import Annotated, Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

from .. import config
from . import registry

logger = logging.getLogger(__name__)


def orchestrator_prompt() -> str:
    """Built from the registry, so a new specialist is routable the moment it is
    registered — there is no second list of agent names to keep in step."""
    return f"""You are the orchestrator of a computer-vision dataset exploration \
platform (Flickr8k: 8,000 images, 5 captions each). Decide which specialist \
should handle the user's latest request:

{registry.routing_menu()}

Usually one specialist is right. Choose two only when the request genuinely has \
two halves (for example "how are captions distributed and show me the worst \
ones"). Never choose more than {registry.MAX_PARALLEL_LANES}.

Reply with ONLY a JSON object:
{{"routes": ["<specialist>"], "brief": "<one-line instruction for them>"}}"""


SYNTHESIZER_PROMPT = """You are the quality gate and final voice of a dataset \
assistant. Review the conversation: does the specialists' work actually answer \
the user's latest request?

Charts, tables and reports the specialists produced are already displayed to the \
user. Refer to them, do not re-describe them, and never restate a table as prose.

- If YES: write the final user-facing answer. Start by naming what was produced \
and the single most useful thing in it — "The report covers X; the notable \
finding is Y" — then stop. Ground every claim in a tool result, mention sample \
ids rather than inventing links, and never end by offering further options: \
answer the question that was asked.
- If NO (wrong direction, missing the point, tool errors went unaddressed): \
reply with exactly `RETRY: <one-line feedback on what to do differently>`."""


class AgentState(MessagesState):
    # Lists, because the orchestrator may select more than one lane.
    routes: list[str]
    retries: int
    # Written concurrently by parallel lanes, so both need an additive reducer;
    # without one LangGraph rejects the second writer in the same superstep.
    lanes_ok: Annotated[list[str], operator.add]
    lanes_failed: Annotated[list[str], operator.add]


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ORPHAN_CLOSE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove a reasoning-model's private thinking from a user-facing answer.

    Qwen3 emits `<think>…</think>` around its scratchpad. Matched pairs are the
    easy case; the one that actually reached the screen was an *orphan* closing
    tag, because the opening tag had been consumed as a separate streamed chunk.
    That left "No further action needed.\\n</think>" sitting above the answer in
    the chat. So an unmatched `</think>` is treated as "everything before this was
    thinking", which is what it means.

    A stray *opening* tag with no close is the opposite case and is left alone:
    discarding everything after it would throw away the whole answer.
    """
    if not text:
        return text
    cleaned = _THINK_BLOCK.sub("", text)
    if "</think>" in cleaned.lower():
        cleaned = _ORPHAN_CLOSE.sub("", cleaned, count=1)
    return cleaned.strip()


def _model() -> ChatOllama:
    # The request timeout is the outer bound on a single model call. Without it a
    # stalled Ollama pins the lane until the lane timeout fires, which is a much
    # blunter instrument.
    return ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL,
                      temperature=0.1,
                      client_kwargs={"timeout": config.OLLAMA_TIMEOUT})


def _parse_routes(text: str) -> tuple[list[str], str]:
    """Extract routes + brief from the orchestrator's reply.

    A 8-billion-parameter local model wraps JSON in prose, emits `route` instead
    of `routes`, or names a specialist that does not exist. Each of those is
    normal, so each is handled rather than raised: an unroutable reply falls back
    to retrieval, which is the safe default because it is read-only and cheap.
    """
    known = registry.by_name()
    try:
        data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, IndexError):
        logger.warning("Orchestrator returned non-JSON; defaulting to retrieval.")
        return ["retrieval"], ""

    raw = data.get("routes", data.get("route", []))
    if isinstance(raw, str):
        raw = [raw]
    routes, seen = [], set()
    for r in raw:
        r = str(r).strip().lower()
        if r in seen:
            continue
        seen.add(r)
        if r in known or r == "direct":
            routes.append(r)

    if not routes:
        logger.warning("Orchestrator named no known specialist (%r); using retrieval.",
                       raw)
        return ["retrieval"], str(data.get("brief", ""))

    # "direct" means no specialist is needed, so it cannot be combined with one.
    if "direct" in routes:
        routes = ["direct"] if len(routes) == 1 else [r for r in routes if r != "direct"]

    # An expensive lane runs alone. Booting a browser as a side effect of a
    # question about captions is not a trade the user agreed to.
    expensive = [r for r in routes if r in known and known[r].cost == "expensive"]
    if expensive:
        routes = expensive[:1]
    else:
        routes = routes[:registry.MAX_PARALLEL_LANES]

    return routes, str(data.get("brief", ""))


def build_graph(model=None, specialists=None):
    """Compile the orchestration graph.

    Both arguments exist for tests: injecting a stub model and stub specialists
    is what makes the graph's own behaviour — parallel fan-out, lane isolation,
    lane timeouts — testable without a running Ollama and without waiting on an
    8B model to think. Production callers pass neither.
    """
    model = model or _model()
    specialists = list(specialists if specialists is not None else registry.SPECIALISTS)
    agents = {
        s.name: s.agent or create_react_agent(model, s.tools, prompt=s.prompt,
                                              name=s.name)
        for s in specialists
    }

    def orchestrate(state: AgentState):
        try:
            reply = model.invoke([SystemMessage(orchestrator_prompt())] + state["messages"])
            routes, brief = _parse_routes(reply.content or "")
        except Exception as exc:
            # The orchestrator itself failing must not end the turn: retrieval is
            # read-only and answers most requests acceptably.
            logger.warning("Orchestrator call failed (%s); defaulting to retrieval.", exc)
            routes, brief = ["retrieval"], ""

        msgs = []
        if routes != ["direct"]:
            lanes = " ‖ ".join(routes)
            msgs.append(AIMessage(
                content=f"[orchestrator → {lanes}] {brief}".rstrip(),
                name="orchestrator"))
        return {"routes": routes, "messages": msgs}

    def fan_out(state: AgentState) -> list[str]:
        """Returning a list makes LangGraph run those nodes in one parallel
        superstep. A single-element list is the ordinary one-specialist case."""
        routes = state.get("routes") or ["retrieval"]
        if routes == ["direct"]:
            return ["synthesize"]
        return routes

    def make_lane(spec: registry.Specialist):
        agent = agents[spec.name]

        def run(state: AgentState):
            started = time.monotonic()
            base = len(state["messages"])
            box: dict[str, Any] = {}

            def work():
                try:
                    box["result"] = agent.invoke(
                        {"messages": state["messages"]},
                        {"recursion_limit": config.AGENT_RECURSION_LIMIT})
                except BaseException as exc:                  # noqa: BLE001
                    box["error"] = exc

            # A daemon thread joined with a timeout, deliberately not a
            # ThreadPoolExecutor: the executor's context manager calls
            # shutdown(wait=True) on exit, which blocks on precisely the hung
            # lane the timeout exists to escape. That version was written first
            # and measured — a 0.3s lane timeout still took 30s to return, so
            # the timeout was decorative. A daemon thread is also abandoned
            # cleanly: it cannot delay interpreter shutdown if it never finishes.
            thread = threading.Thread(target=work, name=f"lane-{spec.name}",
                                      daemon=True)
            thread.start()
            thread.join(timeout=config.AGENT_LANE_TIMEOUT)

            if thread.is_alive():
                logger.warning("lane %s timed out after %ss",
                               spec.name, config.AGENT_LANE_TIMEOUT)
                return {
                    "lanes_failed": [spec.name],
                    "messages": [AIMessage(
                        content=f"[{spec.name} timed out after "
                                f"{config.AGENT_LANE_TIMEOUT}s and produced nothing]",
                        name=spec.name)]}
            if "error" in box:
                exc = box["error"]
                logger.warning("lane %s failed: %s: %s", spec.name,
                               type(exc).__name__, exc)
                return {
                    "lanes_failed": [spec.name],
                    "messages": [AIMessage(
                        content=f"[{spec.name} failed: {type(exc).__name__}: {exc}]",
                        name=spec.name)]}

            new = box["result"]["messages"][base:]
            logger.info("lane %s finished in %.1fs (%d messages)",
                        spec.name, time.monotonic() - started, len(new))
            return {"messages": new, "lanes_ok": [spec.name]}

        return run

    def synthesize(state: AgentState):
        prompt = SYNTHESIZER_PROMPT
        failed = state.get("lanes_failed") or []
        if failed:
            # Told explicitly, because the alternative is an answer that silently
            # covers half the request and reads as if it covered all of it.
            prompt += (f"\n\nIMPORTANT: these specialists failed and returned "
                       f"nothing usable: {', '.join(failed)}. Answer with what the "
                       f"others produced and state plainly, in one short sentence, "
                       f"what could not be checked.")
        try:
            reply = model.invoke([SystemMessage(prompt)] + state["messages"])
            content = strip_reasoning(reply.content or "")
        except Exception as exc:
            logger.warning("Synthesizer failed (%s); returning the lanes' own output.", exc)
            return {"messages": [AIMessage(
                content=_fallback_answer(state, exc), name="final")]}

        if content.startswith("RETRY:") and state.get("retries", 0) < 1:
            return {"retries": state.get("retries", 0) + 1,
                    "messages": [AIMessage(content=f"[quality gate] {content}",
                                           name="synthesizer")]}
        return {"messages": [AIMessage(content=content.removeprefix("RETRY:").strip(),
                                       name="final")]}

    def route_after_synthesize(state: AgentState):
        last = state["messages"][-1]
        return "orchestrate" if getattr(last, "name", "") == "synthesizer" else END

    graph = StateGraph(AgentState)
    graph.add_node("orchestrate", orchestrate)
    for spec in specialists:
        graph.add_node(spec.name, make_lane(spec))
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "orchestrate")
    graph.add_conditional_edges(
        "orchestrate", fan_out,
        {**{s.name: s.name for s in specialists}, "synthesize": "synthesize"})
    for spec in specialists:
        graph.add_edge(spec.name, "synthesize")
    graph.add_conditional_edges("synthesize", route_after_synthesize,
                                {"orchestrate": "orchestrate", END: END})
    return graph.compile()


def _fallback_answer(state: AgentState, exc: Exception) -> str:
    """What to say when the synthesizer itself is unavailable.

    The specialists' work is already done and is often the whole answer, so it is
    handed over directly rather than discarded — with the failure named, because
    an answer that skipped its own quality gate should say so.
    """
    for msg in reversed(state["messages"]):
        content = strip_reasoning(getattr(msg, "content", "") or "")
        if isinstance(content, str) and content.strip() and not content.startswith("["):
            return (f"{content.strip()}\n\n(The final review step was unavailable — "
                    f"{type(exc).__name__} — so this is the specialist's own answer, "
                    f"unverified.)")
    return (f"The specialists ran but the final review step failed "
            f"({type(exc).__name__}: {exc}).")
