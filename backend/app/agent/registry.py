"""The specialist registry: the one place a new agent is declared.

`build_graph` derives everything from `SPECIALISTS` — the graph nodes, the
orchestrator's routing menu, the parallel fan-out, and the synthesizer's view of
which lanes ran. So adding an agent is appending one `Specialist` here, and
nothing that already works has to be edited to accommodate it.

That property is easy to claim and easy to lose, so `tests/test_agent_graph.py`
asserts it directly: it registers a throwaway specialist and checks the compiled
graph gained a node and the routing prompt gained a line, with no other change.

The `cost` field exists because a menu of equals invites a model to pick
everything. Fan-out is capped and expensive lanes are excluded from speculative
selection, which is the difference between "the orchestrator may consult two
specialists" and "every question boots a browser".
"""
from dataclasses import dataclass, field
from typing import Any, Literal

from .tools import (
    attribute_coverage,
    dataset_overview,
    find_similar,
    get_sample_details,
    rare_slice_examples,
    search_images,
    suspect_captions,
    tag_samples,
)
from .viz_tools import VIZ_TOOLS, plot_distribution

Cost = Literal["cheap", "expensive"]


@dataclass(frozen=True)
class Specialist:
    """One specialist agent.

    name    — graph node name and the token the orchestrator emits.
    summary — the routing menu line. This is the only description the
              orchestrator sees, so it must say what the specialist is *for*,
              phrased as the kind of request that should reach it.
    prompt  — the specialist's own system prompt.
    tools   — LangChain tools it may call.
    cost    — "expensive" lanes are never chosen speculatively alongside others
              and are excluded from parallel fan-out.
    """
    name: str
    summary: str
    prompt: str
    tools: list[Any] = field(default_factory=list)
    cost: Cost = "cheap"
    # A prebuilt runnable to use instead of wrapping `tools` in a ReAct loop.
    # Two reasons it exists: a specialist whose work is deterministic (the QA
    # sweep, say) does not need a model in the loop at all, and tests need to
    # exercise the graph's fan-out and failure handling without an 8B model
    # deciding how long that takes.
    agent: Any = None


RETRIEVAL = Specialist(
    name="retrieval",
    summary="finding or showing images: search, similar images, inspecting or "
            "tagging specific samples",
    prompt="""You are the retrieval specialist for a CV dataset platform.
Use your tools to find, inspect, compare, or tag dataset samples. Prefer hybrid
search. When the user wants to *see* images, call show_images with the ids you
found so they render inline. Keep answers concise; mention sample ids — the UI
renders the images, so never fabricate image markdown or URLs.""",
    tools=[search_images, find_similar, get_sample_details, tag_samples],
)

INSIGHTS = Specialist(
    name="insights",
    summary="dataset statistics, attribute coverage, rare or long-tail slices, "
            "caption quality and annotation errors, stated in numbers",
    prompt="""You are the dataset-insights specialist for a CV dataset platform.
Use your tools to report statistics, attribute coverage, rare slices, and
caption-quality findings. Ground every number in a tool result — never estimate.
If a distribution is the answer, prefer a chart tool over describing it in prose.
Keep answers concise and analytical.""",
    # Its own five, plus one chart tool. Handing insights the whole
    # visualization kit measurably degrades tool choice on an 8B model — eleven
    # options is past where it picks reliably — and the orchestrator can fan out
    # to the visualization lane in parallel when a request needs both.
    tools=[dataset_overview, attribute_coverage, rare_slice_examples,
           suspect_captions, get_sample_details, plot_distribution],
)

VISUALIZATION = Specialist(
    name="visualization",
    summary="anything to plot, chart, diagram or compare visually; a written "
            "report the user wants generated; and how this platform is built "
            "(its architecture, components, or how a request flows through it)",
    prompt="""You are the visualization specialist for a CV dataset platform.
Your job is to put a *chart* in front of the user, not a paragraph about one.

- plot_distribution(dimension) for how anything is distributed.
- compare_slices(group) to rank slices side by side.
- plot_retrieval_benchmark() for measured search accuracy.
- system_diagram() for how the platform is built.
- build_dataset_report(...) when the user asks for a report, audit or summary.
- show_images(ids) to display specific pictures.

Never invent numbers: the tools return the data and the chart together. After a
tool call, write one or two sentences naming what the chart shows and the single
most useful thing to notice in it. Do not describe the chart's shape — the user
can see it.""",
    tools=VIZ_TOOLS,
)


def _qa_tools() -> list[Any]:
    """Imported lazily: the QA lane depends on Playwright, which is a developer
    tool and deliberately absent from requirements.txt. Importing it eagerly
    would make the whole agent stack unavailable on a machine that only wants to
    ask questions about the dataset."""
    try:
        from .qa_tools import QA_TOOLS
    except ImportError:                                   # pragma: no cover
        return []
    return QA_TOOLS


QA = Specialist(
    name="qa",
    # Narrow on purpose, with the exclusion stated. Without it, "how does this
    # platform work?" routed here and booted a browser — the model matched on
    # "this platform" and ignored that the question was about design, not health.
    summary="whether this application is currently WORKING: run its tests, check "
            "for broken screens, report pass/fail status. Only when the user asks "
            "about health, tests, QA or breakage — NOT for how the platform is "
            "built, what it can do, or anything about the dataset",
    prompt="""You are the QA specialist for this application. You do not answer
questions about the dataset — you answer questions about whether the software
works.

Call run_app_qa to drive a real browser over every workflow and produce a status
report with screenshots. It takes minutes; if a recent run exists, read it
instead of starting another. Report the pass/fail tally plainly, name any failed
workflow, and never describe a run you did not observe in a tool result.""",
    tools=_qa_tools(),
    cost="expensive",
)


SPECIALISTS: list[Specialist] = [RETRIEVAL, INSIGHTS, VISUALIZATION, QA]

# Two lanes is the useful maximum: a request needing three specialists is
# usually one badly-phrased request, and every extra lane is another full model
# round-trip against a single local GPU.
MAX_PARALLEL_LANES = 2


def by_name() -> dict[str, Specialist]:
    return {s.name: s for s in SPECIALISTS}


def routing_menu() -> str:
    """The specialist list as the orchestrator sees it."""
    lines = [f'- "{s.name}": {s.summary}'
             f'{" (expensive — only when clearly asked for)" if s.cost == "expensive" else ""}'
             for s in SPECIALISTS]
    lines.append('- "direct": greetings, or questions about what you can do — '
                 'no data access needed')
    return "\n".join(lines)
def register(spec: Specialist, *, at: int | None = None) -> None:
    """Add a specialist. Exposed so tests can prove the extension point works
    without editing this file, and so a deployment can add a lane at runtime."""
    if spec.name in by_name():
        raise ValueError(f"specialist '{spec.name}' is already registered")
    if at is None:
        SPECIALISTS.append(spec)
    else:
        SPECIALISTS.insert(at, spec)


def unregister(name: str) -> None:
    global SPECIALISTS
    SPECIALISTS = [s for s in SPECIALISTS if s.name != name]
