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
  cheap lanes, LangGraph runs both async nodes in one superstep and merges their
  message updates through the `add_messages` reducer. Latency is the slower lane,
  not the sum.
* **A failed lane must not fail the turn.** Each lane is an isolated LangGraph
  subgraph whose documented node error handler turns an exception or native node
  timeout into state. The synthesizer can therefore name partial failure.
* **Bounded.** LangGraph supplies the per-node wall-clock limits, Ollama supplies
  a finite generation cap, and the HTTP boundary owns the total turn timeout.

All models run locally through Ollama. The deterministic search stack stays the
platform's retrieval engine — agents are consumers of the same service functions
the REST API uses, not a replacement for them.
"""
import logging
import operator
import time
from collections.abc import Sequence
from typing import Annotated, Literal, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.errors import NodeError, NodeTimeoutError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field, StringConstraints, create_model

from .. import config
from . import registry

logger = logging.getLogger(__name__)


def orchestrator_prompt() -> str:
    """Built from the registry, so a new specialist is routable the moment it is
    registered — there is no second list of agent names to keep in step."""
    return f"""You are the orchestrator of a local computer-vision dataset \
exploration platform. Decide which specialist should handle the user's latest \
request:

{registry.routing_menu()}

Usually one specialist is right. Choose two only when the request genuinely has \
two halves needing DIFFERENT tools (for example "chart the split sizes and show \
me examples from the smallest split"). Never choose more than \
{registry.MAX_PARALLEL_LANES}.

ONE MEASUREMENT, ONE LANE. When the request asks about a measured quantity — \
caption agreement, coverage, statistics, difficulty — route it to the single \
specialist that owns that measurement, and do NOT add a second lane even if the \
request also says "show me" or "give me the worst ones". A second lane answers \
with its own scores on a different scale, and two sets of numbers about the same \
question cannot be combined into one honest answer.

Answer with the routing decision itself — the reply is schema-constrained, so \
name the specialist(s), give them one line of instruction, and classify relevant \
user phrases in `propositions`. Do not copy facts from this system message or \
prior assistant messages. Do not decide propositions with keyword rules; \
interpret the meaning of the complete latest request.

Use `discovery_request` when the quoted fact or value is what the user is asking \
you to find, show, count, inspect, compare, or verify. A question and an \
imperative are discovery requests, not claims that need a refusal gate.

Use `asserted_premise` only when the user treats the quoted fact as already true \
while asking for a DIFFERENT operation, such as explaining or summarizing it. \
Only asserted premises require current-turn tool evidence before they may be \
repeated as facts.

Examples:
- "How many images and captions are in the dataset?" → discovery_request
- "Show me the worst caption matches" → discovery_request
- "Summarize the 30% improvement the correction produced" → asserted_premise

Never turn a question into a generic assertion such as "the dataset contains a \
specific number"; the user asked you to discover that value. Include at most \
three propositions. Every `quote` must be an exact, contiguous quote from the \
latest user's text; never paraphrase it."""


SYNTHESIZER_PROMPT = """You are the quality gate and final voice of a dataset \
assistant. Review the conversation: does the specialists' work actually answer \
the user's latest request?

Charts, tables and reports the specialists produced are already displayed to the \
user. Refer to them, do not re-describe them, and never restate a table as prose.

- If YES: choose the `answer` outcome and write the final user-facing answer. \
Start by naming what was produced \
and the single most useful thing in it — "The report covers X; the notable \
finding is Y" — then stop. Ground every claim in a tool result, mention sample \
ids rather than inventing links, and never end by offering further options: \
answer the question that was asked.
- If NO (wrong direction, missing the point, tool errors went unaddressed): \
choose the `retry` outcome and put one line of corrective instruction in \
`retry_feedback`. The output contract makes answer and retry mutually exclusive.

NUMBERS. Tool results carry a `score_basis` with a `score_meaning` beside it. \
Quote a score only together with the basis that produced it, in the form \
<value> (<basis>) — never a bare figure. Use the values from the tool results \
and no others. NEVER merge, average, rank across or relabel \
numbers that came from different lanes or different bases: a reciprocal-rank \
fusion score and a cosine share no scale, and an image-caption agreement \
measures a different thing from either. If two lanes report figures, keep them \
in separate sentences with their own bases.

A PREMISE IS NOT EVIDENCE. When the user's question asserts a fact — a number, \
a finding, an improvement, an event ("the 30% gain the correction produced", \
"the leak you found earlier") — treat it as a claim to check, never as \
something established. Repeat it only if a tool result in this conversation \
contains it. If nothing does, say so in the first sentence, plainly: this \
dataset does not record that, and here is what IS measured. Do not explain, \
rationalise or find a reason for a figure you cannot see; do not offer a tag, \
an album or any other action built on one; and never invent sample ids to \
illustrate it. A confident answer to a false premise is the most damaging \
thing this assistant can produce, because it looks exactly like a true one.

When the user asks HOW MANY, or for a count, size, split or any other figure, \
the answer must BE those figures, restated from the tool result — not a \
description of what the tool can do and not a statement that the capability is \
available. If the number is genuinely absent from every tool result, say that \
plainly instead of substituting capability status for it."""


class ClaimAssessment(BaseModel):
    """The synthesizer's assessment and cited excerpt for one routed premise."""

    claim: str = Field(description="Exact asserted premise supplied by the router.")
    status: Literal["supported", "not_supported"] = Field(
        description=(
            "The synthesizer's judgment of whether a current-turn tool result "
            "supports the claim."
        )
    )
    evidence: str = Field(
        default="",
        description=(
            "An exact excerpt copied from a current-turn tool result when "
            "supported; empty when not_supported."
        ),
    )


NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class AnswerDecision(BaseModel):
    """The quality gate accepted the evidence and is answering the user."""

    outcome: Literal["answer"]
    answer: NonEmptyText
    claim_assessments: list[ClaimAssessment] = Field(
        default_factory=list,
        description=(
            "One assessment for every asserted premise supplied by the router. "
            "Never mark a claim supported without an exact tool-result excerpt."
        ),
    )


class RetryDecision(BaseModel):
    """The quality gate rejected the work and requests one bounded retry."""

    outcome: Literal["retry"]
    retry_feedback: NonEmptyText


class SynthesisDecision(BaseModel):
    """Required, discriminated quality-gate output.

    An empty object is intentionally invalid. The model must either provide a
    non-empty answer or a non-empty retry instruction; there is no ambiguous
    third state for application code to repair.
    """

    decision: Annotated[
        AnswerDecision | RetryDecision,
        Field(discriminator="outcome"),
    ]


class AgentState(MessagesState):
    # Lists, because the orchestrator may select more than one lane.
    routes: list[str]
    retries: int
    # Semantic output of the structured router. Specialists must establish
    # these from tools before the synthesizer may repeat them as facts.
    claims_to_verify: list[str]
    # Written concurrently by parallel lanes, so both need an additive reducer;
    # without one LangGraph rejects the second writer in the same superstep.
    lanes_ok: Annotated[list[str], operator.add]
    lanes_failed: Annotated[list[str], operator.add]


def _unverified_claims(
    claims: Sequence[str],
    assessments: Sequence[ClaimAssessment],
    messages,
) -> list[str]:
    """Claims lacking a supported assessment with an exact tool-result excerpt.

    This deterministic gate is deliberately structural: it verifies that the
    synthesizer cited text which actually occurs in a current-turn ToolMessage.
    It does not decide whether that excerpt semantically entails the claim; that
    judgment remains explicit in the synthesizer's typed assessment.
    """
    evidence = "\n".join(
        message.content
        for message in messages
        if isinstance(message, ToolMessage) and isinstance(message.content, str)
    ).casefold()
    by_claim = {item.claim.strip().casefold(): item for item in assessments}
    unverified: list[str] = []
    for claim in claims:
        item = by_claim.get(claim.strip().casefold())
        excerpt = item.evidence.strip() if item else ""
        if (
            item is None
            or item.status != "supported"
            or not excerpt
            or excerpt.casefold() not in evidence
        ):
            unverified.append(claim)
    return unverified


def _claim_refusal(claims: Sequence[str]) -> str:
    listed = "\n".join(f"- {claim}" for claim in claims)
    return (
        "I could not verify the following premise"
        f"{'s' if len(claims) != 1 else ''} from this turn's tool results:\n"
        f"{listed}\n\nI have not used "
        f"{'those premises' if len(claims) != 1 else 'that premise'} as fact."
    )


def _model() -> ChatOllama:
    # The request timeout bounds one model call; LangGraph separately bounds the
    # async node containing it, and FastAPI bounds the complete turn.
    # num_ctx is pinned rather than left to each model's default. Ollama sizes
    # the KV cache from the context length, so an unpinned model can reserve a
    # multiple of its own weights: qwen3:30b-a3b defaulted to a 262,144-token
    # window and 44 GB resident, against 40,960 and 11 GB for qwen3:8b. That is
    # not a comparison, and on a laptop it is not a safe default either.
    # num_predict is the server-side cap on GENERATION. Async cancellation ends
    # our request; this cap guarantees Ollama itself cannot decode indefinitely.
    return ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL,
                      temperature=0.1, reasoning=False,
                      num_ctx=config.OLLAMA_NUM_CTX,
                      num_predict=config.OLLAMA_NUM_PREDICT,
                      client_kwargs={"timeout": config.OLLAMA_TIMEOUT})


class UserProposition(BaseModel):
    """One exact user phrase, typed by how truth is treated in the request."""

    quote: str = Field(
        description="Exact contiguous quote from the latest user message.",
    )
    kind: Literal["discovery_request", "asserted_premise"] = Field(
        description=(
            "discovery_request when the quoted fact or value is what the user "
            "asks the assistant to find, show, count, inspect, compare, or "
            "verify; asserted_premise when the user presupposes it is already "
            "true while asking for a different operation such as summarizing "
            "or explaining it."
        ),
    )


def route_schema() -> type[BaseModel]:
    """The router's output contract, built from the registry.

    The specialist names become a real enum in the JSON schema the model is
    constrained to, so "which lanes exist" is stated once — a new specialist is
    routable, and enumerable, the moment it is registered. Structured output
    replaces parsing a name out of prose: the model is held to this shape by
    Ollama's schema-constrained decoding rather than asked politely for JSON.

    One lane is the SHAPE of the answer, not a list with a cap: a `list` field
    invites filling, and measurably did. Asked for `routes: list[lane]`, qwen3:8b
    chose two lanes for a caption-quality question in 3 runs of 3 — sending a
    retrieval lane to fetch search scores for a question about agreement scores,
    which is exactly how a reciprocal rank ends up printed as an agreement. The
    same model given `route` plus an optional `also` picks one, because one is
    what the type asks for and a second becomes a deliberate act.
    """
    lanes = Literal[tuple(registry.by_name()) + ("direct",)]   # type: ignore[valid-type]
    return create_model(
        "Route",
        __doc__=("Which specialist should handle the user's latest request, and "
                 "the one-line instruction they should act on."),
        route=(lanes, Field(description="The ONE specialist that should handle "
                                        "this request.")),
        also=(Optional[lanes],                             # noqa: UP045
              Field(default=None,
                    description="A second specialist ONLY when the request has "
                                "two halves needing different tools. Null "
                                "otherwise — one specialist is the normal case, "
                                "and two specialists reporting numbers about the "
                                "same question cannot be combined.")),
        brief=(str, Field(default="", description="One line telling the "
                                                  "specialist what to do.")),
        propositions=(
            list[UserProposition],
            Field(
                default_factory=list,
                max_length=3,
                description=(
                    "Classify at most three relevant factual phrases from the "
                    "latest user. Include ordinary questions and imperatives as "
                    "discovery_request; include a fact only as asserted_premise "
                    "when the user treats it as already true while requesting a "
                    "different operation. Every quote must be exact and "
                    "contiguous; never paraphrase."
                ),
            ),
        ),
    )


def normalise_routes(routes: Sequence[str], brief: str = "") -> tuple[list[str], str]:
    """Apply the routing rules the schema cannot express.

    Validation guarantees the SHAPE (known names, at least one, at most the cap);
    these are the domain rules on top of it, kept pure so they can be tested
    without a model. An empty or unusable selection falls back to retrieval,
    which is the safe default because it is read-only and cheap.
    """
    known = registry.by_name()
    seen: set[str] = set()
    picked: list[str] = []
    for r in routes:
        name = str(r).strip().lower()
        if name in seen or (name not in known and name != "direct"):
            continue
        seen.add(name)
        picked.append(name)

    if not picked:
        logger.warning("Router named no known specialist (%r); using retrieval.", routes)
        return ["retrieval"], brief

    # "direct" means no specialist is needed, so it cannot be combined with one.
    if "direct" in picked:
        picked = ["direct"] if len(picked) == 1 else [r for r in picked if r != "direct"]

    # An expensive lane runs alone. Booting a browser as a side effect of a
    # question about captions is not a trade the user agreed to.
    expensive = [r for r in picked if r in known and known[r].cost == "expensive"]
    if expensive:
        return expensive[:1], brief
    return picked[:registry.MAX_PARALLEL_LANES], brief


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
        s.name: s.agent or create_agent(model, s.tools, system_prompt=s.prompt,
                                        name=s.name)
        for s in specialists
    }

    router = model.with_structured_output(route_schema())
    synthesizer = model.with_structured_output(SynthesisDecision)

    async def orchestrate(state: AgentState):
        try:
            # Schema-constrained decoding, not prose parsed for a name: the model
            # can only answer with lanes that exist, and validation — not a
            # substring search — is what decides the reply was usable.
            decision = await router.ainvoke(
                [SystemMessage(orchestrator_prompt())] + state["messages"])
            picked = [decision.route] + ([decision.also] if decision.also else [])
            routes, brief = normalise_routes(picked, decision.brief)
            latest_user = next(
                (
                    message.content
                    for message in reversed(state["messages"])
                    if (
                        getattr(message, "type", "") in ("human", "user")
                        and isinstance(message.content, str)
                    )
                ),
                "",
            )
            claims: list[str] = []
            for proposition in decision.propositions:
                candidate = proposition.quote.strip()
                if (
                    proposition.kind == "asserted_premise"
                    and candidate
                    and candidate in latest_user
                    and candidate not in claims
                ):
                    claims.append(candidate)
        except Exception as exc:
            # Structured output can still fail — a model that ignores the schema,
            # a validation error, an Ollama hiccup. None of those may end the
            # turn: retrieval is read-only and answers most requests acceptably.
            logger.warning("Routing failed (%s); defaulting to retrieval.", exc)
            routes, brief, claims = ["retrieval"], "", []

        if claims:
            checklist = "; ".join(claims)
            brief = (
                f"{brief} — Claims requiring tool evidence before they may be "
                f"stated as facts: {checklist}. Verify them with your tools; if "
                f"the available evidence does not establish one, say so plainly."
            ).strip(" —")

        msgs = []
        if routes != ["direct"]:
            lanes = " ‖ ".join(routes)
            msgs.append(AIMessage(
                content=f"[orchestrator → {lanes}] {brief}".rstrip(),
                name="orchestrator"))
        return {"routes": routes, "claims_to_verify": claims, "messages": msgs}

    def route_failure(state: AgentState, error: NodeError):
        logger.warning("Routing node failed: %s", error.error)
        return Command(
            update={
                "routes": ["retrieval"],
                "claims_to_verify": [],
                "messages": [AIMessage(
                    content="[orchestrator → retrieval] The routing model was "
                            "unavailable; use the read-only retrieval lane.",
                    name="orchestrator",
                )],
            },
            goto="retrieval",
        )

    def fan_out(state: AgentState) -> list[str]:
        """Returning a list makes LangGraph run those nodes in one parallel
        superstep. A single-element list is the ordinary one-specialist case."""
        routes = state.get("routes") or ["retrieval"]
        if routes == ["direct"]:
            return ["synthesize"]
        return routes

    def make_lane(spec: registry.Specialist):
        agent = agents[spec.name]

        async def run(state: AgentState):
            started = time.monotonic()
            base = len(state["messages"])
            result = await agent.ainvoke(
                {"messages": state["messages"]},
                {"recursion_limit": config.AGENT_RECURSION_LIMIT},
            )
            new = result["messages"][base:]
            logger.info("lane %s finished in %.1fs (%d messages)",
                        spec.name, time.monotonic() - started, len(new))
            return {"messages": new, "lanes_ok": [spec.name]}

        def lane_failure(state: AgentState, error: NodeError):
            exc = error.error
            if isinstance(exc, NodeTimeoutError):
                detail = f"timed out after {exc.timeout:.0f}s"
            else:
                detail = f"failed: {type(exc).__name__}: {exc}"
            logger.warning("lane %s %s", spec.name, detail)
            return {
                "lanes_failed": [spec.name],
                "messages": [AIMessage(
                    content=f"[{spec.name} {detail}; its work was not included]",
                    name=spec.name,
                )],
            }

        # A lane is the independently recoverable unit, so its timeout and
        # NodeError handler belong to a subgraph boundary. The parent fan-out
        # receives only the branch delta and can merge sibling results normally.
        lane_graph = StateGraph(AgentState)
        lane_graph.add_node(
            "work",
            run,
            timeout=config.AGENT_LANE_TIMEOUT,
            error_handler=lane_failure,
        )
        lane_graph.add_edge(START, "work")
        lane_graph.add_edge("work", END)
        isolated = lane_graph.compile()

        async def outer_lane(state: AgentState):
            """Expose only this lane's delta to the parallel parent superstep."""
            message_base = len(state["messages"])
            ok_base = len(state.get("lanes_ok") or [])
            failed_base = len(state.get("lanes_failed") or [])
            result = await isolated.ainvoke(state)
            return {
                "messages": result["messages"][message_base:],
                "lanes_ok": (result.get("lanes_ok") or [])[ok_base:],
                "lanes_failed": (result.get("lanes_failed") or [])[failed_base:],
            }

        return outer_lane

    async def synthesize(state: AgentState):
        prompt = SYNTHESIZER_PROMPT
        failed = state.get("lanes_failed") or []
        if failed:
            # Told explicitly, because the alternative is an answer that silently
            # covers half the request and reads as if it covered all of it.
            prompt += (f"\n\nIMPORTANT: these specialists failed and returned "
                       f"nothing usable: {', '.join(failed)}. Answer with what the "
                       f"others produced and state plainly, in one short sentence, "
                       f"what could not be checked.")
        claims = state.get("claims_to_verify") or []
        if claims:
            prompt += (
                "\n\nCLAIMS REQUIRING VERIFICATION. The structured router "
                "identified these user-supplied premises:\n- "
                + "\n- ".join(claims)
                + "\nRepeat a claim as fact only when a tool result from this "
                  "turn establishes it. Otherwise state that it could not be "
                  "verified, then answer only from the evidence returned."
                  "\nAssess EVERY premise listed above in `claim_assessments`, "
                  "including the ones you find unsupported — an omitted "
                  "assessment counts as unsupported and cannot help the user. "
                  "An unsupported premise is not a reason to request a retry. "
                  "Choose the answer outcome and assess it as `not_supported`; "
                  "the deterministic gate will return the refusal."
            )
        try:
            decision = await synthesizer.ainvoke(
                [SystemMessage(prompt)] + state["messages"]
            )
        except Exception as exc:
            logger.warning("Synthesizer failed (%s); returning the lanes' own output.", exc)
            if claims:
                return {"messages": [AIMessage(
                    content=_claim_refusal(claims), name="final"
                )]}
            return {"messages": [AIMessage(
                content=_fallback_answer(state, exc), name="final")]}

        outcome = decision.decision
        if isinstance(outcome, RetryDecision):
            if state.get("retries", 0) < 1:
                return {
                    "retries": state.get("retries", 0) + 1,
                    "messages": [
                        AIMessage(
                            content=f"[quality gate] {outcome.retry_feedback}",
                            name="synthesizer",
                        )
                    ],
                }
            if claims:
                content = _claim_refusal(claims)
            else:
                content = _fallback_answer(
                    state,
                    RuntimeError("quality gate rejected the retry"),
                )
            return {"messages": [AIMessage(content=content, name="final")]}

        unverified = _unverified_claims(
            claims, outcome.claim_assessments, state["messages"]
        )
        if unverified:
            return {"messages": [AIMessage(
                content=_claim_refusal(unverified), name="final"
            )]}
        return {"messages": [AIMessage(content=outcome.answer, name="final")]}

    def synth_failure(state: AgentState, error: NodeError):
        logger.warning("Synthesizer node failed: %s", error.error)
        claims = state.get("claims_to_verify") or []
        if claims:
            content = _claim_refusal(claims)
        else:
            content = _fallback_answer(state, error.error)
        return {"messages": [AIMessage(
            content=content, name="final"
        )]}

    def route_after_synthesize(state: AgentState):
        last = state["messages"][-1]
        return "orchestrate" if getattr(last, "name", "") == "synthesizer" else END

    graph = StateGraph(AgentState)
    graph.add_node(
        "orchestrate",
        orchestrate,
        timeout=config.OLLAMA_TIMEOUT,
        error_handler=route_failure,
    )
    for spec in specialists:
        graph.add_node(spec.name, make_lane(spec))
    graph.add_node(
        "synthesize",
        synthesize,
        timeout=config.OLLAMA_TIMEOUT,
        error_handler=synth_failure,
    )

    graph.add_edge(START, "orchestrate")
    graph.add_conditional_edges(
        "orchestrate", fan_out,
        {**{s.name: s.name for s in specialists}, "synthesize": "synthesize"})
    for spec in specialists:
        graph.add_edge(spec.name, "synthesize")
    graph.add_conditional_edges("synthesize", route_after_synthesize,
                                {"orchestrate": "orchestrate", END: END})
    return graph.compile()


def _lane_answer(state: AgentState) -> str:
    """The last thing a specialist said to the user in its own voice.

    Messages carrying tool calls are skipped along with the tool results
    themselves: a `search_images` payload is the evidence behind an answer, not
    an answer, and handing raw JSON to a reader is not an improvement on saying
    nothing. Bracketed lines are skipped too — those are the graph narrating
    itself, never a reply.
    """
    successful = set(state.get("lanes_ok") or [])
    for msg in reversed(state["messages"]):
        if (
            not isinstance(msg, AIMessage)
            or getattr(msg, "name", "") not in successful
            or getattr(msg, "tool_calls", None)
        ):
            continue
        raw = msg.content
        content = raw.strip() if isinstance(raw, str) else ""
        if content and not content.startswith("["):
            return content
    return ""


def _fallback_answer(state: AgentState, exc: Exception) -> str:
    """What to say when the synthesizer itself is unavailable.

    The specialists' work is already done and is often the whole answer, so it is
    handed over directly rather than discarded — with the failure named, because
    an answer that skipped its own quality gate should say so.
    """
    content = _lane_answer(state)
    if content:
        return (f"{content}\n\n(The final review step was unavailable — "
                f"{type(exc).__name__} — so this is the specialist's own answer, "
                f"unverified.)")
    return (f"The specialists ran but the final review step failed "
            f"({type(exc).__name__}: {exc}).")
