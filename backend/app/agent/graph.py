"""Fugu-style orchestration (after Sakana's Fugu: a multi-agent system behind
a single interface): an orchestrator analyzes the request and dispatches
specialist agents; a synthesizer verifies the result against the request and
either finalizes the answer or sends it back for one corrective round.

    user → orchestrator ─┬→ retrieval agent (search / similar / inspect / tag)
                         └→ insights agent  (stats / coverage / caption QA)
                                   ↓
                             synthesizer ──(quality gate)──→ answer
                                   └───── one retry with feedback ─────┘

All models run locally through Ollama. The deterministic search stack stays
the platform's retrieval engine — agents are consumers of the same service
functions the REST API uses, not a replacement for them.
"""
import json
import logging
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

from .. import config
from .tools import INSIGHT_TOOLS, RETRIEVAL_TOOLS

logger = logging.getLogger(__name__)

ORCHESTRATOR_PROMPT = """You are the orchestrator of a computer-vision dataset \
exploration platform (Flickr8k: 8k images, 5 captions each). Decide which \
specialist should handle the user's latest request:

- "retrieval": finding/showing images, searching, similar images, inspecting or \
tagging specific samples.
- "insights": dataset statistics, attribute coverage, rare slices, caption \
quality / annotation errors.
- "direct": greetings or questions about what you can do — no data access needed.

Reply with ONLY a JSON object: {"route": "retrieval" | "insights" | "direct", \
"brief": "<one-line instruction for the specialist>"}"""

RETRIEVAL_PROMPT = """You are the retrieval specialist for a CV dataset platform. \
Use your tools to find, inspect, compare, or tag dataset samples. Prefer hybrid \
search. Keep answers concise; mention sample ids you found — the UI renders the \
images automatically, so never fabricate image markdown or URLs."""

INSIGHTS_PROMPT = """You are the dataset-insights specialist for a CV dataset \
platform. Use your tools to report statistics, attribute coverage, rare slices, \
and caption-quality findings. Ground every number in a tool result — never \
estimate. Keep answers concise and analytical."""

SYNTHESIZER_PROMPT = """You are the quality gate and final voice of a dataset \
assistant. Review the conversation: does the specialist's work actually answer \
the user's latest request?

- If YES: write the final user-facing answer (concise, friendly, grounded in \
the tool results; mention sample ids rather than inventing links).
- If NO (wrong direction, missing the point, tool errors went unaddressed): \
reply with exactly `RETRY: <one-line feedback on what to do differently>`."""


class AgentState(MessagesState):
    route: str
    retries: int


def _model() -> ChatOllama:
    return ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL,
                      temperature=0.1)


def build_graph():
    model = _model()
    retrieval_agent = create_react_agent(model, RETRIEVAL_TOOLS,
                                         prompt=RETRIEVAL_PROMPT, name="retrieval")
    insights_agent = create_react_agent(model, INSIGHT_TOOLS,
                                        prompt=INSIGHTS_PROMPT, name="insights")

    def orchestrate(state: AgentState):
        reply = model.invoke([SystemMessage(ORCHESTRATOR_PROMPT)] + state["messages"])
        route, brief = "retrieval", ""
        try:
            text = reply.content
            data = json.loads(text[text.index("{"): text.rindex("}") + 1])
            route = data.get("route", "retrieval")
            brief = data.get("brief", "")
        except Exception:
            logger.warning("Orchestrator returned non-JSON; defaulting to retrieval.")
        if route not in ("retrieval", "insights", "direct"):
            route = "retrieval"
        msgs = []
        if brief and route != "direct":
            msgs.append(AIMessage(content=f"[orchestrator → {route}] {brief}",
                                  name="orchestrator"))
        return {"route": route, "messages": msgs}

    def route_after_orchestrator(state: AgentState) -> Literal["retrieval", "insights", "synthesize"]:
        return state["route"] if state["route"] in ("retrieval", "insights") else "synthesize"

    def run_retrieval(state: AgentState):
        result = retrieval_agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"][len(state["messages"]):]}

    def run_insights(state: AgentState):
        result = insights_agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"][len(state["messages"]):]}

    def synthesize(state: AgentState):
        reply = model.invoke([SystemMessage(SYNTHESIZER_PROMPT)] + state["messages"])
        content = (reply.content or "").strip()
        if content.startswith("RETRY:") and state.get("retries", 0) < 1:
            return {"retries": state.get("retries", 0) + 1,
                    "messages": [AIMessage(content=f"[quality gate] {content}",
                                           name="synthesizer")]}
        return {"messages": [AIMessage(content=content.removeprefix("RETRY:").strip(),
                                       name="final")]}

    def route_after_synthesize(state: AgentState) -> Literal["orchestrate", "__end__"]:
        last = state["messages"][-1]
        return "orchestrate" if getattr(last, "name", "") == "synthesizer" else END

    graph = StateGraph(AgentState)
    graph.add_node("orchestrate", orchestrate)
    graph.add_node("retrieval", run_retrieval)
    graph.add_node("insights", run_insights)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "orchestrate")
    graph.add_conditional_edges("orchestrate", route_after_orchestrator,
                                {"retrieval": "retrieval", "insights": "insights",
                                 "synthesize": "synthesize"})
    graph.add_edge("retrieval", "synthesize")
    graph.add_edge("insights", "synthesize")
    graph.add_conditional_edges("synthesize", route_after_synthesize,
                                {"orchestrate": "orchestrate", END: END})
    return graph.compile()
