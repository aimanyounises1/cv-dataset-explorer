"""Tests for the orchestration graph and the block protocol.

These run without Ollama and without a browser: the graph takes an injected model
and injected specialists precisely so its own behaviour can be tested at the
speed of a unit test. What is asserted here is the set of claims the architecture
makes and that are otherwise easy to believe without evidence:

* fan-out is genuinely concurrent (lane execution windows overlap in wall clock),
* one lane failing does not fail the turn,
* a lane that hangs is cut off rather than holding the request open,
* adding a specialist requires no edit to the graph, and
* a malformed render block is rejected at the boundary, not in the browser.
"""
import json
import threading
import time

import pytest

# Guard first, imports second. langchain_core and langgraph arrive together
# with requirements-agent.txt, so an environment without the optional agent
# stack has to SKIP this module -- and the guard was unreachable below the
# imports it protects: the first langchain_core import raised ModuleNotFound
# during collection and interrupted the whole suite.
pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from pydantic import Field, ValidationError  # noqa: E402

from app.agent import blocks, registry  # noqa: E402
from app.agent.graph import (  # noqa: E402
    _split_retry,
    _ungrounded_figures,
    _unsupported_premises,
    build_graph,
    normalise_routes,
    route_schema,
)
from app.agent.report_md import report_to_markdown  # noqa: E402

# --------------------------------------------------------------- test doubles

class StubModel(BaseChatModel):
    """Answers the orchestrator with fixed JSON, then acts as the synthesizer.

    A real `BaseChatModel` subclass rather than a duck type, because
    `create_agent` type-checks its model argument — and the extensibility test
    has to compile the *real* registry, whose specialists are genuine
    tool-calling agents.

    The graph calls the model twice per turn with different system prompts; which
    call this is can be told from the prompt, which is more robust than counting
    invocations when a retry can add a third.
    """
    routes: list[str] = Field(default_factory=list)
    synth: str = "Final answer."
    fail_on: set[str] = Field(default_factory=set)
    calls: list[str] = Field(default_factory=list)

    def __init__(self, routes=None, *, synth="Final answer.", fail_on=None, **kw):
        super().__init__(routes=list(routes or []), synth=synth,
                         fail_on=set(fail_on or ()), calls=[], **kw)

    @property
    def _llm_type(self) -> str:
        return "stub"

    def bind_tools(self, tools, **kwargs):
        """A no-op: the specialists under test are prebuilt lanes, so the stub is
        only ever asked to route and to synthesize, never to call a tool."""
        return self

    def with_structured_output(self, schema, **kwargs):
        """Stand in for Ollama's schema-constrained decoding.

        Production routing is `model.with_structured_output(Route)`, so the stub
        has to answer that call too. It returns the scripted routes as a
        validated instance of the real schema — which means a stub that scripts
        a lane the registry does not have fails here exactly as the live model
        would, rather than sneaking past into the graph.
        """
        stub = self

        class _Router:
            def invoke(self, messages, config=None, **kw):
                stub.calls.append("orchestrate")
                if "orchestrate" in stub.fail_on:
                    raise RuntimeError("stub model refused to orchestrate")
                r = list(stub.routes) or ["retrieval"]
                return schema(route=r[0],
                              also=r[1] if len(r) > 1 else None,
                              brief="go")

        return _Router()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = str(getattr(messages[0], "content", ""))
        role = "synthesize" if "quality gate" in system else "orchestrate"
        self.calls.append(role)
        if role in self.fail_on:
            raise RuntimeError(f"stub model refused to {role}")
        if role == "orchestrate":
            text = json.dumps({"routes": self.routes, "brief": "go"})
        else:
            text = self.synth
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class RecordingLane:
    """A specialist stand-in that records when it started and finished."""

    def __init__(self, name, delay=0.0, boom=None, hang=False):
        self.name = name
        self.delay = delay
        self.boom = boom
        self.hang = hang
        self.window = None

    def invoke(self, state, config=None):
        start = time.monotonic()
        if self.hang:
            # Longer than any lane timeout under test. The graph must not wait
            # for this; the thread is abandoned when the timeout fires.
            time.sleep(30)
        time.sleep(self.delay)
        self.window = (start, time.monotonic())
        if self.boom:
            raise self.boom
        return {"messages": list(state["messages"]) +
                            [AIMessage(content=f"{self.name} did the work",
                                       name=self.name)]}


def make_specialist(lane, cost="cheap"):
    return registry.Specialist(name=lane.name, summary=f"{lane.name} lane",
                               prompt="p", tools=[], cost=cost, agent=lane)


def run(model, lanes, message="do the thing"):
    graph = build_graph(model=model,
                        specialists=[make_specialist(x) for x in lanes])
    return graph.invoke(
        {"messages": [HumanMessage(message)], "routes": [], "retries": 0,
         "lanes_ok": [], "lanes_failed": []},
        config={"recursion_limit": 40})


# ------------------------------------------------------------- route selection

@pytest.mark.parametrize("routes,expected", [
    (["visualization"], ["visualization"]),
    (["insights"], ["insights"]),
    (["insights", "visualization"], ["insights", "visualization"]),
    # More lanes than the cap allows are truncated, not honoured.
    (["insights", "visualization", "retrieval"], ["insights", "visualization"]),
    # "direct" means no specialist, so it cannot ride along with one.
    (["direct", "insights"], ["insights"]),
    (["direct"], ["direct"]),
    # A name outside the registry cannot reach a lane.
    (["nonsense"], ["retrieval"]),
    ([], ["retrieval"]),
])
def test_normalise_routes(routes, expected):
    assert normalise_routes(routes)[0] == expected


def test_the_router_schema_only_admits_lanes_that_exist():
    """The names are an enum in the schema the model is constrained to, so an
    invented specialist is a validation error rather than a silent bad route."""
    Route = route_schema()
    assert Route(route="insights", brief="x").route == "insights"
    assert Route(route="insights", also="visualization").also == "visualization"
    # One lane is the default shape: a second is opt-in, not a list to fill.
    assert Route(route="insights").also is None
    with pytest.raises(ValidationError):
        Route(route="nonsense", brief="x")
    with pytest.raises(ValidationError):
        Route(route="insights", also="nonsense")
    with pytest.raises(ValidationError):       # a lane must be named
        Route(brief="x")


def test_routing_survives_a_router_that_fails():
    """Structured output can still fail — a refusing model, a validation error,
    an Ollama hiccup. The turn must continue on the cheap read-only lane."""
    lanes = [RecordingLane("retrieval")]
    out = run(StubModel(["insights"], fail_on={"orchestrate"}), lanes)
    assert lanes[0].window is not None, "retrieval did not run after a router failure"
    assert any(getattr(m, "name", "") == "final" for m in out["messages"])


def test_expensive_lane_never_rides_along():
    """An expensive lane runs alone: a question about captions must not boot a
    browser as a side effect of the orchestrator hedging."""
    routes, _ = normalise_routes(["qa", "insights"])
    assert routes == ["qa"]


def test_duplicate_routes_collapse():
    routes, _ = normalise_routes(["insights", "insights"])
    assert routes == ["insights"]


# ------------------------------------------------------------------- fan-out

def test_two_lanes_run_concurrently():
    """The parallelism claim, measured rather than asserted.

    Each lane sleeps 0.4s. Run sequentially the turn takes >= 0.8s and the two
    execution windows do not overlap; run in parallel it takes ~0.4s and they do.
    Overlap is the real assertion — a duration threshold alone would pass on a
    fast machine that still ran them one after another.
    """
    a = RecordingLane("insights", delay=0.4)
    b = RecordingLane("visualization", delay=0.4)
    started = time.monotonic()
    result = run(StubModel(["insights", "visualization"]), [a, b])
    elapsed = time.monotonic() - started

    assert sorted(result["lanes_ok"]) == ["insights", "visualization"]
    assert a.window and b.window, "both lanes must have run"
    overlap = min(a.window[1], b.window[1]) - max(a.window[0], b.window[0])
    assert overlap > 0.2, f"lanes did not overlap (overlap={overlap:.3f}s)"
    assert elapsed < 0.75, f"turn took {elapsed:.2f}s — lanes appear serialized"


def test_single_lane_is_the_ordinary_case():
    a = RecordingLane("retrieval", delay=0.01)
    result = run(StubModel(["retrieval"]), [a])
    assert result["lanes_ok"] == ["retrieval"]
    assert result["lanes_failed"] == []


def test_direct_route_skips_every_lane():
    a = RecordingLane("retrieval")
    result = run(StubModel(["direct"]), [a])
    assert a.window is None, "no specialist should have run"
    assert result["lanes_ok"] == []
    assert result["messages"][-1].content == "Final answer."


# --------------------------------------------------------- failure isolation

def test_one_failing_lane_does_not_fail_the_turn():
    good = RecordingLane("insights", delay=0.05)
    bad = RecordingLane("visualization", boom=ValueError("tool exploded"))
    result = run(StubModel(["insights", "visualization"]), [good, bad])

    assert result["lanes_ok"] == ["insights"]
    assert result["lanes_failed"] == ["visualization"]
    # The turn still produced an answer.
    assert result["messages"][-1].name == "final"
    # And the failure is visible in the transcript rather than swallowed.
    assert any("visualization failed" in str(m.content) for m in result["messages"])


def test_synthesizer_is_told_which_lanes_failed():
    """The prompt must name the failed lanes; otherwise the answer covers half
    the request while reading as though it covered all of it."""
    seen = {}

    class Watcher(StubModel):
        def invoke(self, messages):
            system = str(getattr(messages[0], "content", ""))
            if "quality gate" in system:
                seen["prompt"] = system
            return super().invoke(messages)

    bad = RecordingLane("visualization", boom=RuntimeError("nope"))
    run(Watcher(["visualization"]), [bad])
    assert "visualization" in seen.get("prompt", "")
    assert "failed" in seen.get("prompt", "").lower()


def test_hanging_lane_is_cut_off(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "AGENT_LANE_TIMEOUT", 0.3)
    hung = RecordingLane("retrieval", hang=True)
    started = time.monotonic()
    result = run(StubModel(["retrieval"]), [hung])
    elapsed = time.monotonic() - started

    assert result["lanes_failed"] == ["retrieval"]
    assert elapsed < 5, f"the turn waited {elapsed:.1f}s on a hung lane"
    assert any("ran out of time" in str(m.content) for m in result["messages"])


def test_orchestrator_failure_falls_back_to_retrieval():
    a = RecordingLane("retrieval", delay=0.01)
    result = run(StubModel(["insights"], fail_on={"orchestrate"}), [a])
    assert result["lanes_ok"] == ["retrieval"]


def test_synthesizer_failure_still_answers():
    """If the quality gate is down, the specialist's own answer is handed over
    with the gap stated — losing completed work to a failed review step would be
    the worse outcome."""
    a = RecordingLane("retrieval", delay=0.01)
    result = run(StubModel(["retrieval"], fail_on={"synthesize"}), [a])
    final = result["messages"][-1]
    assert final.name == "final"
    assert "retrieval did the work" in final.content
    assert "unverified" in final.content


# -------------------------------------------------------------- extensibility

def test_registering_a_specialist_needs_no_graph_edit():
    """The extension point, asserted directly: register a specialist and the
    compiled graph gains a node and the routing menu gains a line, with no other
    change anywhere."""
    before_nodes = set(build_graph(model=StubModel(["retrieval"])).get_graph().nodes)
    before_menu = registry.routing_menu()

    spec = registry.Specialist(name="weather", summary="the weather, obviously",
                               prompt="p", tools=[],
                               agent=RecordingLane("weather"))
    registry.register(spec)
    try:
        after_nodes = set(build_graph(model=StubModel(["retrieval"])).get_graph().nodes)
        assert after_nodes - before_nodes == {"weather"}
        assert "weather" in registry.routing_menu()
        assert registry.routing_menu() != before_menu
        # And it is immediately routable.
        assert normalise_routes(["weather"])[0] == ["weather"]
    finally:
        registry.unregister("weather")

    assert "weather" not in registry.routing_menu()


def test_registry_names_are_unique_and_described():
    names = [s.name for s in registry.SPECIALISTS]
    assert len(names) == len(set(names))
    for s in registry.SPECIALISTS:
        assert s.summary and s.prompt, f"{s.name} is missing a summary or prompt"


def test_duplicate_registration_is_refused():
    with pytest.raises(ValueError):
        registry.register(registry.SPECIALISTS[0])


def test_lanes_are_thread_safe_under_concurrent_turns():
    """Two turns at once must not cross-contaminate lane bookkeeping."""
    results = {}

    def go(tag):
        results[tag] = run(StubModel(["insights"]),
                           [RecordingLane("insights", delay=0.1)])

    threads = [threading.Thread(target=go, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert len(results) == 3
    for r in results.values():
        assert r["lanes_ok"] == ["insights"]
        assert r["lanes_failed"] == []


# ------------------------------------------------------------------- blocks

def test_every_declared_kind_has_a_model():
    from pydantic import TypeAdapter

    from app.agent.blocks import Block

    adapter = TypeAdapter(Block)
    schema = adapter.json_schema()
    assert schema, "the block union must produce a schema"
    assert set(blocks.BLOCK_KINDS) == {
        m.model_fields["kind"].default
        for m in (blocks.BarBlock, blocks.LineBlock, blocks.PieBlock,
                  blocks.HistogramBlock, blocks.TableBlock, blocks.StatBlock,
                  blocks.FlowBlock, blocks.ImagesBlock, blocks.ReportBlock,
                  blocks.QABlock, blocks.TagProposalBlock)}


def test_block_requires_a_source():
    """A chart with no stated provenance is indistinguishable from one the model
    invented, so `source` is required rather than optional."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="source"):
        blocks.BarBlock(title="untraceable", series=[])


def test_cap_points_folds_the_tail_rather_than_dropping_it():
    points = [{"label": f"c{i}", "value": float(i)} for i in range(100)]
    kept, dropped = blocks.cap_points(points)
    assert len(kept) == blocks.MAX_POINTS
    assert dropped == 100 - blocks.MAX_POINTS + 1
    # The folded mass is preserved, not discarded.
    assert sum(p["value"] for p in kept) == pytest.approx(sum(range(100)))
    assert "other" in kept[-1]["label"]


def test_capping_is_disclosed_in_the_note():
    block = blocks.bar("many", "SELECT", [{"label": f"c{i}", "value": i}
                                          for i in range(60)])
    assert block.note and "other" in block.note


def test_flow_rejects_an_edge_to_a_missing_node():
    with pytest.raises(ValueError):
        blocks.flow("bad", "src", [{"id": "a", "label": "A"}],
                    [{"src": "a", "dst": "ghost"}], [["a"]])


def test_table_truncation_is_disclosed():
    rows = [{"a": i} for i in range(blocks.MAX_TABLE_ROWS + 50)]
    block = blocks.table("big", "SELECT", [{"key": "a", "label": "A"}], rows)
    assert len(block.rows) == blocks.MAX_TABLE_ROWS
    assert "rows" in (block.note or "")


def test_images_block_reports_the_full_total():
    ids = list(range(100))
    block = blocks.images("many", "SELECT", ids)
    assert len(block.sample_ids) == blocks.MAX_IMAGES
    assert block.total == 100


# ---------------------------------------------------------------- markdown

def test_report_markdown_covers_every_block_kind():
    """A report must not lose a section because a block kind has no Markdown
    renderer — the download is the artifact people keep."""
    every = [
        blocks.bar("B", "s", [{"label": "x", "value": 1}]),
        blocks.line("L", "s", [{"name": "n", "points": [{"x": 1, "y": 2}]}]),
        blocks.pie("P", "s", [{"label": "x", "value": 1}]),
        blocks.histogram("H", "s", [{"lo": 0, "hi": 1, "count": 3}], marker=0.5,
                         marker_label="cut"),
        blocks.table("T", "s", [{"key": "a", "label": "A"}], [{"a": 1}]),
        blocks.stat("S", "s", [{"label": "n", "value": "1"}]),
        blocks.flow("F", "s", [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                    [{"src": "a", "dst": "b", "label": "e"}], [["a"], ["b"]]),
        blocks.images("I", "s", [1, 2]),
    ]
    report = blocks.ReportBlock(
        title="All kinds", source="test",
        sections=[blocks.ReportSection(heading="Everything", blocks=every)])
    md = report_to_markdown(report.model_dump(mode="json"))
    assert "no Markdown rendering" not in md
    for b in every:
        assert b.title in md
        assert b.source in md


# -------------------------------------------------- reasoning-model artifacts

@pytest.mark.parametrize("raw,expected", [
    ("<think>secret plan</think>\n\nThe answer is 42.", "The answer is 42."),
    # The case that actually reached the screen: an orphan closing tag, because
    # the opening tag arrived as a separate chunk.
    ("No further action needed.\n</think>\n\nThe report covers X.",
     "The report covers X."),
    ("a<think>mid</think>b", "ab"),
    ("<think>only thinking</think>", ""),
    ("plain answer", "plain answer"),
    ("", ""),
    # An unmatched *opening* tag is left alone: cutting everything after it would
    # discard the entire answer.
    ("<think>unclosed and then the answer",
     "<think>unclosed and then the answer"),
])
def test_reasoning_is_stripped_from_user_facing_text(raw, expected):
    from app.agent.graph import strip_reasoning

    assert strip_reasoning(raw) == expected


def test_synthesized_answer_has_no_reasoning_markers():
    a = RecordingLane("retrieval", delay=0.01)
    model = StubModel(["retrieval"],
                      synth="<think>hmm, let me see</think>\nThe dataset has 8,000 images.")
    result = run(model, [a])
    assert result["messages"][-1].content == "The dataset has 8,000 images."


def test_chart_payload_precomputes_shares():
    """The model must never have to derive a percentage: given raw counts it
    reported "train 60%" under a chart correctly showing 75%. Tested on the
    payload builder rather than through a tool, so it needs no dataset."""
    from app.agent.viz_tools import _payload

    block = blocks.pie("Samples per split", "COUNT(*) grouped by split",
                       [{"label": "train", "value": 6000},
                        {"label": "test", "value": 1000},
                        {"label": "validation", "value": 1000}])
    payload = json.loads(_payload(block, summary="s"))
    assert payload["figures"] == ["train: 6,000 (75.0%)", "test: 1,000 (12.5%)",
                                 "validation: 1,000 (12.5%)"]
    assert "do not compute percentages" in payload["note_to_agent"]


def test_single_series_bars_also_ship_figures():
    from app.agent.viz_tools import _payload

    block = blocks.bar("Hard tail", "COUNT(*) where axis >= 8",
                       [{"label": "Difficulty", "value": 1600},
                        {"label": "Clutter", "value": 400}])
    payload = json.loads(_payload(block))
    assert payload["figures"] == ["Difficulty: 1,600 (80.0%)", "Clutter: 400 (20.0%)"]


def test_payload_omits_figures_when_there_is_nothing_to_share():
    """A flow diagram has no values, so it must not gain a bogus figures list."""
    from app.agent.viz_tools import _payload

    block = blocks.flow("Wiring", "read from the registry",
                        [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                        [{"src": "a", "dst": "b"}], [["a"], ["b"]])
    payload = json.loads(_payload(block))
    assert "figures" not in payload
    assert "note_to_agent" not in payload


def test_markdown_escapes_pipes_in_captions():
    block = blocks.table("T", "s", [{"key": "c", "label": "caption"}],
                         [{"c": "a dog | a cat"}])
    md = report_to_markdown(block.model_dump(mode="json"))
    assert r"a dog \| a cat" in md


def test_tag_samples_proposes_and_never_writes():
    """The assistant's one mutating tool is now an approval gate: it returns a
    tag_proposal block for existing ids and leaves sample_tags untouched."""
    import json as _json

    from app import db as _db
    from app.agent.tools import tag_samples

    conn = _db.connect()
    _db.init_db(conn)
    cur = conn.execute(
        "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
        "VALUES ('flickr8k', 'proposal_probe.jpg', 'train', 10, 10, 1)")
    sid = cur.lastrowid
    conn.commit()
    try:
        # An id that does not exist voids the proposal rather than being
        # quietly dropped: the remainder would be a shorter, cleaner-looking
        # fiction built from images nobody selected.
        refused = _json.loads(tag_samples.func(
            sample_ids=[sid, 999_999_999], tag="Edge-Case ",
            reason="probe reason"))
        assert "blocks" not in refused
        assert 999_999_999 in refused["missing"]

        out = _json.loads(tag_samples.func(
            sample_ids=[sid], tag="Edge-Case ", reason="probe reason"))
        assert out["proposed"] is True and out["candidates"] == 1
        block = out["blocks"][0]
        assert block["kind"] == "tag_proposal" and block["tag"] == "edge-case"
        assert block["sample_ids"] == [sid]
        assert "approve" in out["next"]
        rows = conn.execute(
            "SELECT COUNT(*) FROM sample_tags WHERE sample_id = ?", (sid,)).fetchone()[0]
        assert rows == 0, "a proposal must write nothing"
        # the block survives the union validation the chat boundary applies
        from app.api.chat import _validated_blocks
        assert _validated_blocks([block])[0]["kind"] == "tag_proposal"
    finally:
        conn.execute("DELETE FROM samples WHERE id = ?", (sid,))
        conn.commit()
        conn.close()


def test_system_diagram_names_the_active_provider(monkeypatch):
    """The architecture diagram must credit whichever provider is actually
    ranking — a hard-coded model name lies the moment the provider flips."""
    import json as _json

    from app.agent.viz_tools import system_diagram
    from app.ml import providers

    def fake_state(active):
        return providers.ProviderState(
            preferred=active, active=active,
            model_id=providers.provider_model_id(active), dim=8,
            index_ready=True, fallback_reason=None)

    for active, expected in (("siglip2", "SigLIP 2"), ("qwen3_vl", "Qwen3-VL")):
        monkeypatch.setattr(providers, "resolve", lambda a=active: fake_state(a))
        out = _json.loads(system_diagram.func())
        labels = " ".join(n["label"] for b in out.get("blocks", [])
                          for n in b.get("nodes", []))
        assert expected in labels, f"diagram must name {expected} when active"
        wrong = "SigLIP" if active == "qwen3_vl" else "Qwen"
        engine_labels = " ".join(
            n["label"] for b in out.get("blocks", [])
            for n in b.get("nodes", []) if n.get("group") == "engine")
        assert wrong not in engine_labels, (
            f"engine nodes must not credit {wrong} while {active} is active")


def test_graph_streams_real_node_updates():
    """The live trace is LangGraph's own update stream — node names arrive in
    execution order, and the final values snapshot carries the full state the
    blocking endpoint would have returned."""
    a = RecordingLane("insights", delay=0.05)
    graph = build_graph(model=StubModel(["insights"]),
                        specialists=[make_specialist(a)])
    seen_nodes, final_state = [], None
    for mode, chunk in graph.stream(
            {"messages": [HumanMessage("stream probe")], "routes": [],
             "retries": 0, "lanes_ok": [], "lanes_failed": []},
            config={"recursion_limit": 40},
            stream_mode=["updates", "values"]):
        if mode == "values":
            final_state = chunk
        else:
            seen_nodes += list(chunk)
    assert seen_nodes[0] == "orchestrate"
    assert "insights" in seen_nodes
    assert "synthesize" in seen_nodes
    assert final_state is not None and final_state["lanes_ok"] == ["insights"]


def test_inspect_album_reports_measured_analysis():
    """The assistant's album tool resolves by name or id and returns the same
    measured analysis the UI shows — counted signals, never invented."""
    import json as _json

    from app import db as _db
    from app.agent.tools import inspect_album

    conn = _db.connect()
    _db.init_db(conn)
    cur = conn.execute(
        "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
        "VALUES ('flickr8k', 'agent_alb.jpg', 'train', 10, 10, 1)")
    sid = cur.lastrowid
    a = conn.execute("INSERT INTO albums(name, origin, position, created_at, "
                     "updated_at) VALUES ('agent-probe-album', 'manual', 0, "
                     "'2026-07-28', '2026-07-28')")
    aid = a.lastrowid
    conn.execute("INSERT INTO album_items(album_id, sample_id, position, "
                 "added_at) VALUES (?, ?, 0, '2026-07-28')", (aid, sid))
    conn.commit()
    try:
        out = _json.loads(inspect_album.func(album="agent-probe-album"))
        assert out["album_id"] == aid and out["count"] == 1
        assert "measured" in out and out["sample_ids"] == [sid]
        by_id = _json.loads(inspect_album.func(album=str(aid)))
        assert by_id["name"] == "agent-probe-album"
        miss = _json.loads(inspect_album.func(album="no-such-album"))
        assert "existing_albums" in miss
    finally:
        conn.execute("DELETE FROM album_items WHERE album_id = ?", (aid,))
        conn.execute("DELETE FROM albums WHERE id = ?", (aid,))
        conn.execute("DELETE FROM samples WHERE id = ?", (sid,))
        conn.commit()
        conn.close()


def test_a_proposal_lists_every_id_it_proposes_and_writes_nothing():
    """The approval UI renders one togglable member per proposed id, so the
    block has to carry them all — a preview would let an approval tag samples
    the reviewer never saw. Ids that do not exist are reported separately
    rather than padding the list, and proposing writes no tag at all: the
    mutation belongs to the human click that follows."""
    import json as _json

    from app import db as _db
    from app.agent.tools import tag_samples

    conn = _db.connect()
    _db.init_db(conn)
    made = []
    for i in range(3):
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, "
            "filesize) VALUES ('flickr8k', ?, 'train', 10, 10, 1)",
            (f"propose_{i}.jpg",))
        made.append(cur.lastrowid)
    conn.commit()
    ghost = 10_000_000 + made[0]
    try:
        # A list carrying one invented id is refused whole.
        with_ghost = _json.loads(tag_samples.func(
            tag="Night-Probe", sample_ids=[*made, ghost],
            reason="three real frames and one that does not exist"))
        assert "blocks" not in with_ghost
        assert with_ghost["missing"] == [ghost]

        out = _json.loads(tag_samples.func(
            tag="Night-Probe", sample_ids=made,
            reason="  three real frames  "))
        block = out["blocks"][0]
        assert block["kind"] == "tag_proposal"
        # Every real id survives, in order, so the grid can render each one.
        assert block["sample_ids"] == made
        assert block["tag"] == "night-probe"          # normalized for the write
        assert block["reason"] == "three real frames"

        # The proposal is a proposal: no tag row, no membership, nothing.
        assert conn.execute(
            "SELECT COUNT(*) c FROM tags WHERE name = 'night-probe'"
        ).fetchone()["c"] == 0

        # An entirely invented proposal is refused rather than half-built.
        bad = _json.loads(tag_samples.func(tag="ghosts", sample_ids=[ghost],
                                            reason="none of these exist"))
        assert "error" in bad and "blocks" not in bad

        # The block is bounded, so "render every member" stays a finite promise.
        # Built from REAL rows, because an invented id now voids the proposal.
        many = []
        for i in range(205):
            c = conn.execute(
                "INSERT INTO samples(dataset, filename, split, width, height, "
                "filesize) VALUES ('flickr8k', ?, 'train', 10, 10, 1)",
                (f"cap_{i}.jpg",))
            many.append(c.lastrowid)
        conn.commit()
        try:
            wide = _json.loads(tag_samples.func(
                tag="wide", sample_ids=many, reason="more ids than the cap"))
            assert len(wide["blocks"][0]["sample_ids"]) == 200
        finally:
            qm = ",".join("?" * len(many))
            conn.execute(f"DELETE FROM samples WHERE id IN ({qm})", many)
            conn.commit()
    finally:
        for sid in made:
            conn.execute("DELETE FROM samples WHERE id = ?", (sid,))
        conn.execute("DELETE FROM tags WHERE name IN ('night-probe','wide')")
        conn.commit()
        conn.close()


@pytest.mark.parametrize("reply,expect_retry,expect_answer", [
    # The classic shape the handler already caught.
    ("RETRY: ask for the split sizes instead", "ask for the split sizes instead", ""),
    # The shape that leaked: a full answer, then the control token as its own
    # paragraph. The user was shown the agent's instruction to itself.
    ("The test split holds 1,000 images.\n\nRETRY: check whether agreement "
     "metrics exist", "check whether agreement metrics exist",
     "The test split holds 1,000 images."),
    # Decorated by a model that likes markdown.
    ("Answer.\n\n**RETRY:** try the other lane", "try the other lane", "Answer."),
    # No token: the answer passes through untouched.
    ("The corpus holds 8,000 images.", "", "The corpus holds 8,000 images."),
])
def test_the_quality_gate_token_never_reaches_the_reader(reply, expect_retry, expect_answer):
    """`RETRY:` is the synthesizer's private word for "send this back".

    It was recognised only at position 0, so a model that answered and *then*
    appended the token both skipped the retry and printed its own control
    instruction into the chat, in the same voice as the answer.
    """
    retry, answer = _split_retry(reply)
    assert retry == expect_retry
    assert answer == expect_answer
    assert "RETRY:" not in answer


def test_ungrounded_figures_are_found_by_arithmetic_not_by_trust():
    """A number in the question that no tool returned is named, not assumed.

    The live failure: "Summarize the 30% accuracy improvement the hubness
    correction produced" — a figure that exists nowhere in this repo — was
    explained as fact in three runs of three, once with invented sample ids and
    a curation proposal attached. A general "premises are not evidence" rule did
    not hold, so the check computes the answer and hands the synthesizer the
    specific figure to refuse.
    """
    from langchain_core.messages import AIMessage as _AI
    from langchain_core.messages import HumanMessage as _HM

    # The tools said nothing about 30%.
    convo = [_HM("Summarize the 30% accuracy improvement the hubness correction produced."),
             _AI("dataset_overview: 8000 samples, 40000 captions")]
    assert _ungrounded_figures(convo) == ["30%"]

    # A figure the tools DID return is grounded and must not be flagged.
    grounded = [_HM("Why do 1,000 images sit in the test split?"),
                _AI('{"splits": {"test": 1000, "train": 6000}}')]
    assert _ungrounded_figures(grounded) == []

    # Only the latest question is judged, and a question with no figures is quiet.
    assert _ungrounded_figures([_HM("Which captions look wrong?"), _AI("x")]) == []


def test_a_figure_is_grounded_as_a_number_not_as_a_substring():
    """The check must not be satisfied by digits inside another number.

    This is the bug that made the first version useless: "30" appears inside
    sample id 4301 and inside 0.308, so every tool result "grounded" the 30%
    premise and the warning never fired on the one question it exists for.
    """
    from langchain_core.messages import AIMessage as _AI
    from langchain_core.messages import HumanMessage as _HM

    q = _HM("Summarize the 30% accuracy improvement the correction produced.")
    noise = _AI('{"sample_ids": [4301, 1305], "scores": [0.308, 0.130]}')
    assert _ungrounded_figures([q, noise]) == ["30%"]

    # A real 30 in the evidence does ground it.
    real = _AI('{"splits": {"test": 30}}')
    assert _ungrounded_figures([q, real]) == []


def test_a_fabricated_event_is_refused_even_with_no_number_in_it():
    """The figure check is arithmetic over digits, so a premise carrying no
    digits walked past it: "Recap the duplicate cluster you removed last week"
    was adopted in 3 of 4 runs, two of which invented "8,500 → 8,000" and
    declared the corpus clean. A claim about what this assistant DID can never
    be grounded — no tool reports past actions — so it is always refusable."""
    from langchain_core.messages import HumanMessage as _HM

    assert _unsupported_premises(
        [_HM("Recap the duplicate cluster you removed last week.")]) == ["you removed"]
    assert _unsupported_premises(
        [_HM("Summarize the indoor bias you measured earlier.")]) == ["you measured"]
    # An ordinary question makes no such claim.
    assert _unsupported_premises([_HM("Which captions look wrong?")]) == []
    assert _unsupported_premises([_HM("Can you find night scenes?")]) == []


def test_the_orchestrator_note_is_not_evidence_for_itself():
    """The warning quotes the figure it warns about. Counting the orchestrator's
    own note as tool output made the second pass conclude 30% was grounded —
    the warning silencing the warning."""
    from langchain_core.messages import AIMessage as _AI
    from langchain_core.messages import HumanMessage as _HM

    convo = [_HM("Summarize the 30% improvement the correction produced."),
             _AI("[orchestrator → insights] NOTE: 30% appear in the question and "
                 "in no tool result.", name="orchestrator"),
             _AI('{"total_samples": 8000}')]
    assert _ungrounded_figures(convo) == ["30%"]
