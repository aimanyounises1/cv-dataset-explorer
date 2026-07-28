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
import logging
import operator
import re
import threading
import time
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field, create_model

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
name the specialist(s) and give them one line of instruction."""


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
reply with exactly `RETRY: <one-line feedback on what to do differently>`.

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


class AgentState(MessagesState):
    # Lists, because the orchestrator may select more than one lane.
    routes: list[str]
    retries: int
    # `time.monotonic()` at which this turn must be over. Set once, by the
    # orchestrator, and read by every step after it — the bound belongs to the
    # turn, not to each step, or the per-step bounds add up to the wall clock a
    # person actually waits.
    deadline: float
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


# Markdown on either side of the token: a model that has been told to write
# `RETRY:` will happily emit "**RETRY:**", "> RETRY:" or "- _RETRY:_".
_RETRY_LINE = re.compile(r"^[ \t>*_-]*RETRY:[ \t*_]*(.*?)[ \t*_]*$",
                         re.IGNORECASE | re.MULTILINE)


def _split_retry(content: str) -> tuple[str, str]:
    """(retry feedback, answer) — the control token never reaches the reader.

    `RETRY:` is the quality gate's private word for "send this back", and the
    handler used to look for it only at position 0. A model that writes a full
    answer and *then* appends "RETRY: check whether agreement metrics exist"
    slipped straight through: the retry never fired, and the user was shown the
    agent's instruction to itself in the same voice as the answer. So the token
    is recognised wherever it appears, and stripped from the text either way —
    an internal control word is never part of a reply, even when the retry
    budget is spent.
    """
    hits = _RETRY_LINE.findall(content)
    if not hits:
        return "", content.strip()
    answer = _RETRY_LINE.sub("", content).strip()
    return (hits[0].strip() or "revise the answer"), answer


ic = r"\d[\d,]*(?:\.\d+)?"          # 30, 1,000, 0.31 — separators included
_FIGURE = re.compile(rf"{ic}\s?%|\b{ic}\b")
del ic


def _ungrounded_figures(messages) -> list[str]:
    """Figures the USER asserted that appear in no tool result.

    A leading premise is the assistant's worst failure mode: asked to
    "summarize the 30% accuracy improvement the hubness correction produced",
    an 8B model explained a number that exists nowhere, and once built a tag
    proposal on invented ids to illustrate it. A general instruction not to
    trust premises did not hold — three runs of three still asserted the
    figure.

    So the check is arithmetic rather than advice: take the numbers out of the
    latest question, look for each in everything the tools returned this turn,
    and hand the synthesizer the ones that are missing BY NAME. A specific
    "30% appears in no tool result" is a far harder instruction to talk past
    than "be careful about premises".
    """
    last_user = ""
    evidence: list[str] = []
    for m in messages:
        role = getattr(m, "type", "") or getattr(m, "role", "")
        text = m.content if isinstance(getattr(m, "content", None), str) else ""
        if role in ("human", "user"):
            last_user = text
            continue
        # The orchestrator's own warning QUOTES the ungrounded figure ("NOTE: 30%
        # appear in no tool result"). Counting it as evidence made the check
        # conclude, on the second pass, that 30% was grounded — the warning
        # silencing the warning. A note is not a measurement, and no tool wrote
        # it, so it is not evidence.
        if getattr(m, "name", "") == "orchestrator":
            continue
        evidence.append(text)
    if not last_user:
        return []
    # Separators are formatting, not value: the tools write 1000 where a person
    # writes 1,000, and the two must compare equal.
    hay = " ".join(evidence).replace(",", "")
    out: list[str] = []
    for raw in _FIGURE.findall(last_user):
        token = raw.strip()
        bare = token.rstrip("%").strip().replace(",", "")
        # A percentage is always a claim; a bare one- or two-digit number is
        # usually a quantity being asked FOR ("show me 5") rather than asserted,
        # and flagging those would hedge ordinary questions.
        if "%" not in token and len(bare.replace(".", "")) < 3:
            continue
        # As a NUMBER, not a substring. Matching "30" anywhere in the evidence
        # is satisfied by sample id 4301 or a score of 0.308, so every figure
        # looked grounded and the check never fired — measured: the note was
        # absent on the very question it exists for.
        if not bare or token in out:
            continue
        if not re.search(rf"(?<![\d.]){re.escape(bare)}(?![\d.])", hay):
            out.append(token)
    return out


def _remaining(state: AgentState) -> float:
    """Seconds left in this turn's budget.

    A turn with no deadline recorded — a graph invoked directly by a test, or
    state carried over from an older run — gets the full budget rather than a
    negative one, so the bound can only ever help.
    """
    deadline = state.get("deadline")
    if not deadline:
        return config.AGENT_TURN_BUDGET
    return deadline - time.monotonic()


def lane_budget(remaining: float) -> float:
    """How long a lane may run, given what is left of the turn.

    The reserve is what keeps the synthesizer inside the budget instead of
    starting a fresh full-length model call after the lanes have already spent
    the wall clock — the 240 + 120 = 360s turn this function exists to prevent.
    The floor is because a lane cut to two seconds is a lane guaranteed to
    produce nothing, and an honest short answer beats a pointlessly started one.
    """
    # The ceiling is applied LAST, so the floor can never raise a lane above the
    # per-lane limit that was set deliberately: with the clamps the other way
    # round, configuring a 0.3s lane produced a 20s one.
    return min(config.AGENT_LANE_TIMEOUT,
               max(config.AGENT_LANE_MIN, remaining - config.AGENT_SYNTH_RESERVE))


# "the duplicate cluster you removed last week", "the bias you measured".
_PAST_CLAIM = re.compile(
    r"\byou(?:'ve|\s+have)?\s+(?:already\s+|previously\s+|earlier\s+)?"
    r"(removed|deleted|cleaned|fixed|found|discovered|detected|identified|"
    r"flagged|measured|computed|calculated|ran|built|made|created|chose|"
    r"selected|noticed|observed|reported|concluded|verified|confirmed)\b",
    re.IGNORECASE)


def _unsupported_premises(messages) -> list[str]:
    """Claims about things the assistant supposedly DID, which it cannot check.

    The figure check is arithmetic over digits, so a fabricated *event* with no
    number in it walked straight past it: "Recap the duplicate cluster you
    removed last week and confirm the corpus is clean now" was adopted in 3 of
    4 runs, two of which invented "8,500 → 8,000" and declared the corpus clean.

    A premise of this shape can never be grounded, and that is a fact about the
    architecture rather than a guess: each turn starts from the conversation
    plus this turn's tool results, and no tool reports what was done to the
    corpus in the past. So an attribution of a past action to the assistant is
    always something to refuse, whatever it claims.
    """
    for m in reversed(messages):
        role = getattr(m, "type", "") or getattr(m, "role", "")
        if role not in ("human", "user"):
            continue
        text = m.content if isinstance(getattr(m, "content", None), str) else ""
        return sorted({f"you {v.lower()}" for v in _PAST_CLAIM.findall(text)})
    return []


def _model() -> ChatOllama:
    # The request timeout is the outer bound on a single model call. Without it a
    # stalled Ollama pins the lane until the lane timeout fires, which is a much
    # blunter instrument.
    # num_ctx is pinned rather than left to each model's default. Ollama sizes
    # the KV cache from the context length, so an unpinned model can reserve a
    # multiple of its own weights: qwen3:30b-a3b defaulted to a 262,144-token
    # window and 44 GB resident, against 40,960 and 11 GB for qwen3:8b. That is
    # not a comparison, and on a laptop it is not a safe default either.
    # num_predict is the cap on GENERATION, and it is the only bound here that
    # stops the model rather than stopping our wait for it. The request timeout
    # and the lane timeout both abandon a thread; Ollama carries on decoding the
    # request nobody is reading any more, and with one slot on a laptop that
    # runaway blocks every later turn until it finishes. See config for the
    # measurement that made this non-optional.
    return ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL,
                      temperature=0.1, num_ctx=config.OLLAMA_NUM_CTX,
                      num_predict=config.OLLAMA_NUM_PREDICT,
                      client_kwargs={"timeout": config.OLLAMA_TIMEOUT})


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

    def orchestrate(state: AgentState):
        try:
            # Schema-constrained decoding, not prose parsed for a name: the model
            # can only answer with lanes that exist, and validation — not a
            # substring search — is what decides the reply was usable.
            decision = router.invoke(
                [SystemMessage(orchestrator_prompt())] + state["messages"])
            picked = [decision.route] + ([decision.also] if decision.also else [])
            routes, brief = normalise_routes(picked, decision.brief)
        except Exception as exc:
            # Structured output can still fail — a model that ignores the schema,
            # a validation error, an Ollama hiccup. None of those may end the
            # turn: retrieval is read-only and answers most requests acceptably.
            logger.warning("Routing failed (%s); defaulting to retrieval.", exc)
            routes, brief = ["retrieval"], ""

        # The warning belongs where the fabrication STARTS. Told only to the
        # synthesizer, it asked one model to contradict the lane whose text it
        # was summarising, and lost: the insights lane wrote "the notable
        # finding is a 30% accuracy improvement" and the summary repeated it.
        # Carried in the brief, every lane sees it before it writes anything.
        loose = _ungrounded_figures(state["messages"])
        if loose:
            brief = (f"{brief} — NOTE: {', '.join(loose)} appear in the question "
                     f"and in no tool result. Do not treat "
                     f"{'it' if len(loose) == 1 else 'them'} as measured, do not "
                     f"explain or infer "
                     f"{'it' if len(loose) == 1 else 'them'} from other data, and "
                     f"say plainly that this dataset does not record "
                     f"{'it' if len(loose) == 1 else 'them'}.").strip(" —")
        claims = _unsupported_premises(state["messages"])
        if claims:
            brief = (f"{brief} — NOTE: the question refers to something this "
                     f"assistant supposedly did ({', '.join(claims)}). It keeps "
                     f"no record of past actions and no tool reports them, so "
                     f"that cannot be confirmed. Say so plainly, do not recap or "
                     f"invent what happened, and answer only from what the tools "
                     f"return now.").strip(" —")

        msgs = []
        if routes != ["direct"]:
            lanes = " ‖ ".join(routes)
            msgs.append(AIMessage(
                content=f"[orchestrator → {lanes}] {brief}".rstrip(),
                name="orchestrator"))
        # Stamped once per turn, by the first node that runs. Every step after
        # this reads it rather than starting its own clock.
        return {"routes": routes, "messages": msgs,
                "deadline": state.get("deadline")
                            or time.monotonic() + config.AGENT_TURN_BUDGET}

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

            # LangGraph's own per-node bound, add_node(timeout=...), is the
            # first choice and is not available here: it is async-only, and
            # setting it on a sync node fails at compile time because sync
            # Python cannot be cancelled in-process. These lanes are sync
            # because everything under them is — SQLite, NumPy and the SigLIP
            # forward pass — and so are both chat endpoints, which drive the
            # graph through invoke() and stream().
            #
            # So the bound is a daemon thread joined with a timeout,
            # deliberately not a ThreadPoolExecutor: the executor's context
            # manager calls shutdown(wait=True) on exit, which blocks on
            # precisely the hung lane the timeout exists to escape. That version
            # was written first and measured — a 0.3s lane timeout still took
            # 30s to return, so the timeout was decorative. A daemon thread is
            # also abandoned cleanly: it cannot delay interpreter shutdown if it
            # never finishes.
            #
            # What this bound cannot do is stop the work: an abandoned thread
            # never reaches Ollama, which keeps generating on a request nobody
            # is waiting for. OLLAMA_NUM_PREDICT is what ends that generation
            # itself; see the note on it in config.py.
            budget = lane_budget(_remaining(state))
            thread = threading.Thread(target=work, name=f"lane-{spec.name}",
                                      daemon=True)
            thread.start()
            thread.join(timeout=budget)

            if thread.is_alive():
                logger.warning("lane %s timed out after %.0fs", spec.name, budget)
                return {
                    "lanes_failed": [spec.name],
                    "messages": [AIMessage(
                        content=f"[{spec.name} ran out of time after "
                                f"{budget:.0f}s and its work was not included]",
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
        loose = _ungrounded_figures(state["messages"])
        if loose:
            prompt += (
                f"\n\nUNGROUNDED FIGURES — checked, not guessed: "
                f"{', '.join(loose)} appear in the user's question and in NO tool "
                f"result this turn. Open your answer by saying that this dataset "
                f"does not record "
                f"{'that figure' if len(loose) == 1 else 'those figures'}, then "
                f"answer with what IS measured. Do not repeat "
                f"{', '.join(loose)} as fact, do not explain or justify "
                f"{'it' if len(loose) == 1 else 'them'}, and propose no tag, album "
                f"or other action built on "
                f"{'it' if len(loose) == 1 else 'them'}.")
        # Starting a model call the budget cannot cover is how a turn that had
        # already failed slowly went on to fail slowly a second time. With too
        # little left, the lanes' own work is handed over immediately — it is
        # real output, and shipping it late is strictly worse than shipping it.
        left = _remaining(state)
        if left < config.AGENT_SYNTH_MIN:
            logger.warning("skipping synthesis: %.0fs left of the turn budget", left)
            return {"messages": [AIMessage(
                content=_fallback_answer(state, TimeoutError(
                    f"only {left:.0f}s of the {config.AGENT_TURN_BUDGET:.0f}s turn "
                    f"budget remained")), name="final")]}
        try:
            reply = model.invoke([SystemMessage(prompt)] + state["messages"])
            content = strip_reasoning(reply.content or "")
        except Exception as exc:
            logger.warning("Synthesizer failed (%s); returning the lanes' own output.", exc)
            return {"messages": [AIMessage(
                content=_fallback_answer(state, exc), name="final")]}

        asked_retry, content = _split_retry(content)
        if asked_retry and state.get("retries", 0) < 1:
            return {"retries": state.get("retries", 0) + 1,
                    "messages": [AIMessage(content=f"[quality gate] {asked_retry}",
                                           name="synthesizer")]}
        return {"messages": [AIMessage(content=content, name="final")]}

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
