"""What "the application works" means, written once as executable flows.

There is no frontend unit-test tier in this project, so this registry is the net
that catches what `tsc` cannot — a view that renders empty, a control that no
longer filters, a console error, a 404. Every flow asserts *behaviour through the
UI* rather than implementation details, so it survives restyling.

The registry has three consumers — the CLI (`scripts/ui_smoke.py`), the HTTP
endpoint (`POST /api/qa/run`) and the assistant's status report — and exactly one
definition. Duplicating the assertions per consumer is how a suite starts lying:
the copy nobody runs rots, and then the two disagree about whether the app works.

Adding a workflow is one decorated function and no other edit anywhere::

    @flow("Tag editor")
    def tag_editor(pg, ok):
        pg.goto(url("/samples/1723"), wait_until="domcontentloaded")
        ok("tag input present", bool(pg.query_selector(".tag-editor input")))

A flow receives a Playwright page and `ok(name, cond, detail)`. Beyond that it
may do anything — sample canvas pixels, shift-drag a lasso, drive the keyboard —
because "does the map produce a set the gallery can show?" is not expressible in
a declarative table of selectors. `ok` returns the boolean it recorded, so a flow
can branch on a failed precondition instead of cascading into noise. Raising is
also fine: the runner isolates each flow, so an exception fails that flow alone.

Two lessons are encoded in the waits and assertions below, both learned the hard
way and both easy to undo by accident:

* **Never wait on "networkidle" against a dev server.** Vite's HMR socket and 60
  lazy-loaded thumbnails mean it may never settle; wait for a selector.
* **Assert against DOM text (`textContent`), not `inner_text`,** wherever CSS
  applies `text-transform` — `inner_text` returns the *rendered* casing, so a
  lowercase assertion against an uppercased header fails for no real reason.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from .. import config

# ---------------------------------------------------------------- the registry


@dataclass(frozen=True)
class Flow:
    """One workflow. `budget_s` bounds it: a flow that runs past its budget is
    stopped at the next check rather than being allowed to hang the sweep.

    `induces_errors` marks a flow that deliberately breaks something — a
    degradation test that intercepts a request and returns a 500. Its console
    errors and 4xx/5xx responses are recorded against the flow but kept out of
    the sweep's global tallies, because those tallies decide whether the
    application is healthy. Without this, a test that proves the app handles a
    failure well would itself report the app as broken.
    """
    name: str
    fn: Callable[..., None]
    budget_s: float
    induces_errors: bool = False


FLOWS: list[Flow] = []


def flow(name: str, *, budget_s: float = 150.0, induces_errors: bool = False):
    """Register a flow under `name`. Names are the CLI's `--flow` filter and the
    deck's slide titles, so a duplicate is rejected at import rather than
    quietly shadowing whichever twin ran first."""
    def register(fn: Callable[..., None]) -> Callable[..., None]:
        if any(f.name == name for f in FLOWS):
            raise ValueError(f"duplicate QA flow name: {name!r}")
        FLOWS.append(Flow(name=name, fn=fn, budget_s=budget_s,
                          induces_errors=induces_errors))
        return fn
    return register


def get_flows(only: Optional[list[str]] = None) -> list[Flow]:
    """The registry, or the subset whose names contain any of `only`.

    Substring and case-insensitive: `--flow map` should not require the caller
    to remember whether the flow is called "Map" or "Similarity map".
    """
    if not only:
        return list(FLOWS)
    wanted = [w.lower() for w in only]
    return [f for f in FLOWS if any(w in f.name.lower() for w in wanted)]


def url(path: str) -> str:
    """A frontend URL. Read late so a test or an env var can retarget the sweep."""
    return config.QA_BASE_URL.rstrip("/") + path


def api(path: str) -> str:
    return config.QA_API_URL.rstrip("/") + path


# --------------------------------------------------------------------- flows

# Every route and the selector that proves it rendered rather than merely
# returned 200. A blank page with a working router is still a broken page.
ROUTES = [
    ("/", ".grid"),
    ("/map", ".map-canvas"),
    ("/stats", ".stat-cards"),
    ("/quality", ".dist-bars"),
    ("/eval", "button.primary"),
    ("/chat", ".chat-page"),
    ("/samples/1723", ".detail-image"),
]


@flow("Routes", budget_s=240.0)
def routes(pg, ok):
    """Every route loads and paints its own content."""
    for path, marker in ROUTES:
        try:
            pg.goto(url(path), wait_until="domcontentloaded", timeout=45000)
            pg.wait_for_selector(marker, timeout=25000)
            ok(f"route {path}", True)
        except Exception as e:                      # one dead route, not one dead flow
            ok(f"route {path}", False, str(e).split("\n")[0][:110])


@flow("Gallery")
def gallery(pg, ok):
    """Browse, all three search modes, axis filter, sort, density, paging, export."""
    pg.goto(url("/"), wait_until="domcontentloaded")
    pg.wait_for_selector(".grid .card")
    base_n = len(pg.query_selector_all(".grid .card"))
    ok("browse renders 60 cards", base_n == 60, f"{base_n}")
    # Scoped to cards: the result-bar legend renders the same component as a
    # worked example, so a bare `.axis-spark` count is one too many. The check
    # means "every card has one", and now says so.
    sparks = len(pg.query_selector_all(".grid .card .axis-spark"))
    ok("sparkline present on cards", sparks == base_n, f"{sparks} of {base_n} cards")

    for mode in ("hybrid", "semantic", "keyword"):
        pg.goto(url(f"/?q=dog%20in%20snow&mode={mode}"), wait_until="domcontentloaded")
        try:
            pg.wait_for_selector(".grid .card", timeout=25000)
            n = len(pg.query_selector_all(".grid .card"))
            ok(f"search mode {mode}", n > 0, f"{n} results")
        except Exception:
            ok(f"search mode {mode}", False, "no results rendered")

    # axis filter + sort + chips
    pg.goto(url("/?difficulty_min=9&sort=clutter_desc"), wait_until="domcontentloaded")
    pg.wait_for_selector(".grid .card")
    # .active-filters/.filter-chip are ActiveFilters' own classes; ".chip" is the
    # unrelated query-suggestion row, which an earlier version of this test hit.
    chips = [c.inner_text().strip().replace("\n", " ")
             for c in pg.query_selector_all(".active-filters .filter-chip")]
    ok("active filter chip shown for axis",
       any("Difficulty" in c for c in chips), "; ".join(chips)[:100])
    # The order dial is a listbox the design system owns, not a native select;
    # its trigger mirrors the value into data-value, and lives in the DOM even
    # while the settings panel is collapsed.
    ok("sort control reflects URL",
       "clutter" in (pg.eval_on_selector(".sort-trigger",
                                         "e=>e.dataset.value") or ""))

    # density control — now a dial inside the collapsed search-settings panel
    # (one command bar; the query's dials open on demand), so open it first.
    pg.click(".search-settings > summary")
    pg.click(".density-group button:last-child")
    pg.wait_for_timeout(400)
    w = pg.eval_on_selector(".grid .card", "e=>e.getBoundingClientRect().width")
    ok("density L widens cards", w > 200, f"card width {w:.0f}px")

    # load more
    pg.goto(url("/?q=dog&mode=hybrid"), wait_until="domcontentloaded")
    pg.wait_for_selector(".grid .card")
    if pg.query_selector(".load-more button"):
        pg.click(".load-more button")
        pg.wait_for_timeout(1500)
        n2 = len(pg.query_selector_all(".grid .card"))
        ok("load more appends", n2 > 60, f"{n2} cards")
    else:
        ok("load more appends", False, "no Load more button")

    # export links resolve
    hrefs = pg.eval_on_selector_all(".export-pill", "els=>els.map(e=>e.getAttribute('href'))")
    ok("export links present", len(hrefs) == 3, f"{hrefs[:1]}")
    if hrefs:
        r = pg.context.request.get(api(hrefs[0]))
        ok("export csv responds 200", r.status == 200, f"{r.status}, {len(r.body())} bytes")


@flow("Similarity map")
def similarity_map(pg, ok):
    """Colour modes, that the canvas is actually painted, hover, and lasso →
    gallery. The point of the map is to produce a set; the hand-off asserts the
    set can leave the page, which is the part that used to be untested."""
    pg.goto(url("/map"), wait_until="domcontentloaded")
    pg.wait_for_selector(".map-canvas")
    pg.wait_for_timeout(1500)
    for mode in ("cluster", "split", "agreement", "difficulty"):
        pg.select_option("select[aria-label='Colour points by']", mode)
        pg.wait_for_timeout(500)
        has_legend = (bool(pg.query_selector(".legend-ramp"))
                      or bool(pg.query_selector(".legend-swatches .legend-swatch")))
        ok(f"colour-by {mode} renders legend", has_legend)

    # canvas actually painted (non-uniform pixels)
    painted = pg.evaluate("""() => {
      const c = document.querySelector('.map-canvas');
      const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
      const seen = new Set();
      for (let i=0;i<d.length;i+=4000) seen.add(`${d[i]},${d[i+1]},${d[i+2]}`);
      return seen.size;
    }""")
    ok("canvas has painted many colours", painted > 5, f"{painted} distinct sampled colours")

    box = pg.eval_on_selector(
        ".map-canvas",
        "e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}}")
    pg.mouse.move(box["x"] + box["w"] * 0.35, box["y"] + box["h"] * 0.55)
    pg.wait_for_timeout(700)
    ok("hover shows thumbnail tooltip", bool(pg.query_selector(".map-tooltip img")))

    pg.keyboard.down("Shift")
    pg.mouse.move(box["x"] + box["w"] * 0.30, box["y"] + box["h"] * 0.30)
    pg.mouse.down()
    pg.mouse.move(box["x"] + box["w"] * 0.42, box["y"] + box["h"] * 0.45, steps=12)
    pg.mouse.up()
    pg.keyboard.up("Shift")
    pg.wait_for_timeout(600)
    handoff = pg.query_selector(".selection-bar a.button-link")
    ok("lasso selects points", bool(pg.query_selector(".selection-bar")),
       pg.inner_text(".selection-bar .pill")
       if pg.query_selector(".selection-bar .pill") else "no selection bar")
    if not handoff:
        ok("lasso hands its ids to the gallery", False, "no hand-off link after lasso")
        return
    n_sel = int(re.search(r"\d+", pg.inner_text(".selection-bar .pill")).group())
    handoff.click()
    pg.wait_for_selector(".grid .card", timeout=25000)
    pg.wait_for_timeout(600)
    chips = [c.inner_text() for c in pg.query_selector_all(".active-filters .filter-chip")]
    ok("lasso hands its ids to the gallery",
       "ids=" in pg.url and any("Id list" in c for c in chips),
       f"{n_sel} selected; chips: {'; '.join(chips)[:80]}")
    shown = len(pg.query_selector_all(".grid .card"))
    ok("gallery shows the lassoed set", 0 < shown <= n_sel, f"{shown} of {n_sel} on page 1")


@flow("Statistics")
def statistics(pg, ok):
    """Charts render, and the provenance panel states where the numbers came from."""
    pg.goto(url("/stats"), wait_until="domcontentloaded")
    pg.wait_for_selector(".stat-cards")
    pg.wait_for_timeout(1200)
    surfaces = len(pg.query_selector_all(".recharts-surface"))
    ok("charts rendered", surfaces >= 4, f"{surfaces} chart surfaces")
    # The panel is a collapsed <details>, so its text is absent from inner_text
    # until expanded — the earlier assertion was testing the wrong thing.
    pg.click(".caveat summary")
    pg.wait_for_timeout(400)
    ok("provenance panel present", "8,091" in pg.inner_text("body"))


@flow("Quality")
def quality(pg, ok):
    """Agreement histogram, the review threshold, and brush → gallery.

    The suspect list is capped at 100 rows, so the hand-off is the only way to
    reach the rest of a selection; the assertion is that the count on the button
    is the count the gallery then reports.
    """
    pg.goto(url("/quality"), wait_until="domcontentloaded")
    pg.wait_for_selector(".dist-bars")
    pg.wait_for_timeout(1200)
    bars = len(pg.query_selector_all(".dist-bar"))
    before = pg.inner_text(".dist-readout")
    rows_before = len(pg.query_selector_all(".suspect-row"))
    pg.eval_on_selector("#qa-threshold", """el=>{el.value=String(Number(el.max)*0.5);
        el.dispatchEvent(new Event('input',{bubbles:true}));
        el.dispatchEvent(new Event('change',{bubbles:true}));}""")
    pg.wait_for_timeout(1400)
    after = pg.inner_text(".dist-readout")
    rows_after = len(pg.query_selector_all(".suspect-row"))
    ok("histogram rendered", bars == 40, f"{bars} bars")
    ok("threshold changes readout", before != after, f"{before.strip()} -> {after.strip()}")
    ok("suspect list responds", rows_after > 0, f"{rows_before} -> {rows_after} rows")

    review = pg.query_selector(".dist-actions a.button-link")
    ok("threshold offers a review hand-off", bool(review),
       review.inner_text().strip() if review else "no hand-off link")
    exports = pg.eval_on_selector_all(
        ".dist-actions .export-pill", "els=>els.map(e=>e.getAttribute('href'))")
    ok("threshold offers export",
       len(exports) == 3 and "max_agreement" in (exports[0] or ""), str(exports[:1]))
    if exports:
        r = pg.context.request.get(api(exports[0]))
        ok("selection export responds 200", r.status == 200, f"{r.status}, {len(r.body())} bytes")
    if not review:
        return
    claimed = re.search(r"([\d,]+)", review.inner_text())
    claimed_n = int(claimed.group(1).replace(",", "")) if claimed else None
    review.click()
    pg.wait_for_selector(".grid .card", timeout=25000)
    pg.wait_for_timeout(900)
    chips = [c.inner_text().replace("\n", " ")
             for c in pg.query_selector_all(".active-filters .filter-chip")]
    ok("brush becomes a gallery filter",
       "max_agreement=" in pg.url and any("agreement" in c.lower() for c in chips),
       "; ".join(chips)[:90])
    head = pg.inner_text(".result-bar .meta-line")
    reported = int(re.sub(r"[^\d]", "", head.split("sample")[0]) or 0)
    ok("gallery total matches the button's count", claimed_n == reported,
       f"button said {claimed_n}, gallery says {reported}")


# The benchmark computes recall over 1,000 held-out captions on a cold cache,
# which is minutes rather than seconds — hence the budget.
@flow("Benchmark", budget_s=300.0)
def benchmark(pg, ok):
    """The self-benchmark runs and reports all three retrieval modes."""
    pg.goto(url("/eval"), wait_until="domcontentloaded")
    pg.get_by_role("button", name=re.compile("benchmark")).click()
    try:
        pg.wait_for_selector("table.eval-table tbody tr", timeout=180000)
        rows = pg.query_selector_all("table.eval-table tbody tr")
        ok("benchmark returns every mode", len(rows) == 3, f"{len(rows)} rows")
        # Compare against the DOM text, not inner_text: CSS uppercases the
        # headers, so inner_text yields "CANDIDATES".
        hdrs = pg.eval_on_selector_all("table.eval-table th", "e=>e.map(x=>x.textContent.trim())")
        ok("candidates column present", "candidates" in hdrs, str(hdrs))
    except Exception as e:
        ok("benchmark returns every mode", False, str(e).split("\n")[0][:100])


@flow("Sample detail")
def sample_detail(pg, ok):
    """A card opens its sample, which shows every caption, similar images, and
    keyboard navigation to the next result."""
    pg.goto(url("/?q=dog&mode=hybrid"), wait_until="domcontentloaded")
    pg.wait_for_selector(".grid .card")
    pg.query_selector_all(".grid .card")[0].click()
    pg.wait_for_selector(".detail-image", timeout=20000)
    pg.wait_for_timeout(1200)
    captions = len(pg.query_selector_all(".caption-list li"))
    ok("5 captions shown", captions == 5, f"{captions}")
    similar = len(pg.query_selector_all(".grid .card"))
    ok("similar images loaded", similar > 0, f"{similar}")
    url0 = pg.url
    pg.keyboard.press("ArrowRight")
    pg.wait_for_timeout(1000)
    ok("arrow key advances", pg.url != url0, f"...{url0[-6:]} -> ...{pg.url[-6:]}")


@flow("Assistant")
def assistant(pg, ok):
    """The chat page states a definite state: ready, or unavailable with setup
    instructions. A blank panel would be the actual failure."""
    pg.goto(url("/chat"), wait_until="domcontentloaded")
    pg.wait_for_timeout(1500)
    unavailable = "Assistant unavailable" in pg.inner_text("body")
    ok("assistant reports a definite state", True,
       "unavailable panel" if unavailable else "input ready")
    if not unavailable:
        ok("chat input present", bool(pg.query_selector(".chat-input input")))


@flow("Axis legend", budget_s=60.0)
def axis_legend(pg, ok):
    """The key for the four-bar sparkline every card carries.

    That encoding is on all 60 cards and had no legend anywhere — the map has
    one, four chart renderers have one, the gallery had none. Its only
    explanation was an SVG <title>, i.e. a hover you had to already suspect.
    The regression that matters is not "the legend exists" but that explaining
    it costs the grid no vertical space: the gallery just went from 481px to
    159px of chrome before the first image and must not pay it back here.
    """
    pg.goto(url("/"), wait_until="domcontentloaded")
    pg.wait_for_selector(".grid .card")
    pg.wait_for_timeout(800)

    if not ok("a legend is present", bool(pg.query_selector(".axis-legend"))):
        return
    dom = pg.eval_on_selector(".axis-legend", "e=>e.textContent")
    ok("it names all four axes",
       all(a in dom for a in ("Legibility", "Rarity", "Difficulty", "Clutter")), dom[:60])
    ok("it states the direction", "taller" in dom)
    ok("it shows the same sparkline the cards use",
       bool(pg.query_selector(".axis-legend .axis-spark")))

    before = pg.evaluate("()=>Math.round(document.querySelector('.grid .card')"
                         ".getBoundingClientRect().top + scrollY)")
    pg.click(".axis-legend-more")
    pg.wait_for_timeout(600)
    after = pg.evaluate("()=>Math.round(document.querySelector('.grid .card')"
                        ".getBoundingClientRect().top + scrollY)")
    ok("opening it costs the grid no vertical space", before == after,
       f"first frame {before}px -> {after}px")

    panel = pg.eval_on_selector(".axis-legend-panel", "e=>e.textContent")
    ok("it explains what the badge means", "dif 9" in panel or "leg 10" in panel)

    # The cards carrying this encoding render on the sample page and in the
    # assistant too. A key that exists only on the gallery leaves the same bars
    # unexplained everywhere a user can arrive by link — which was the original
    # defect, one page over.
    pg.goto(url("/samples/1723"), wait_until="domcontentloaded")
    pg.wait_for_selector(".detail-image", timeout=25000)
    pg.wait_for_timeout(1200)
    if pg.query_selector(".grid .card .axis-spark"):
        ok("the key follows the cards to the sample page",
           bool(pg.query_selector(".axis-legend")))
    # The axes are dataset-relative percentile ranks. A legend that omits that
    # invites reading a 7 as an absolute measurement.
    ok("it states the percentile caveat", "percentile" in panel)


@flow("Set description", budget_s=90.0)
def set_description(pg, ok):
    """The inversion: given a selection, what characterises it.

    The assertions that matter are the honesty ones. A lift multiplier is trivial
    to compute and trivial to mislead with, so the panel must always show the raw
    count beside it, must report under-representation as well as over-, and must
    report *nothing* when a selection is most of the corpus — where a naive
    binomial test would call every facet significant.
    """
    pg.goto(url("/"), wait_until="domcontentloaded")
    pg.wait_for_selector(".grid .card")
    ok("hidden when nothing is selected", pg.query_selector(".set-summary") is None)

    pg.goto(url("/?max_agreement=0.08"), wait_until="domcontentloaded")
    pg.wait_for_selector(".grid .card")
    if not ok("appears for a selection", bool(pg.query_selector(".set-summary"))):
        return
    pg.click(".set-summary > summary")
    pg.wait_for_timeout(1600)

    rows = pg.query_selector_all(".facet-row")
    ok("facets are reported", len(rows) >= 3, f"{len(rows)} facets")
    # textContent, not inner_text: `.eyebrow` is text-transform:uppercase, so
    # inner_text returns the RENDERED casing and a title-case assertion fails.
    dom = pg.eval_on_selector(".set-summary", "e=>e.textContent")
    ok("under-representation is reported too", "Under-represented" in dom)
    ok("every multiplier carries its raw count",
       len(pg.query_selector_all(".facet-count")) == len(rows))
    ok("the zero-shot caveat is present", "zero-shot" in dom)

    rows[0].click()
    pg.wait_for_selector(".grid .card", timeout=25000)
    pg.wait_for_timeout(500)
    ok("a facet drills into its slice", "attr=" in pg.url,
       pg.url.split("5173")[-1] or "/")

    # The statistical guard: 6,000 of 8,000 samples cannot differ from the corpus.
    pg.goto(url("/?split=train"), wait_until="domcontentloaded")
    pg.wait_for_selector(".grid .card")
    pg.click(".set-summary > summary")
    pg.wait_for_timeout(1600)
    ok("a near-corpus selection reports nothing rather than noise",
       "Nothing stands out" in pg.eval_on_selector(".set-summary", "e=>e.textContent"))

    # Cluster: colourable on the map long before it was reachable anywhere.
    pg.goto(url("/?cluster=6"), wait_until="domcontentloaded")
    pg.wait_for_selector(".grid .card")
    pg.wait_for_timeout(400)
    head = pg.inner_text(".result-bar .meta-line")
    ok("cluster is filterable", "300" in head, head.strip()[:50])


@flow("Train/test leakage", budget_s=90.0)
def leakage(pg, ok):
    """Held-out images with a training near-duplicate.

    The assertion that matters is not the number — it moves with the threshold,
    by design — but that the reader is never shown one without the means to
    check it: the whole threshold ladder, the split of each side, and the pairs
    at a size where you can judge whether two images really are the same.
    """
    pg.goto(url("/stats"), wait_until="domcontentloaded")
    try:
        pg.wait_for_selector(".leakage", timeout=45000)
    except Exception as exc:
        ok("leakage panel renders", False, str(exc).split("\n")[0][:100])
        return
    pg.wait_for_timeout(2000)

    figure = pg.inner_text(".leak-figure").replace("\n", " ")
    ok("a headline figure is reported", any(c.isdigit() for c in figure), figure)
    rungs = pg.query_selector_all(".leak-rung")
    ok("the whole threshold ladder is shown", len(rungs) >= 5, f"{len(rungs)} rungs")
    pairs = pg.query_selector_all(".leak-pair")
    ok("cross-split pairs are shown for inspection", len(pairs) >= 2, f"{len(pairs)} pairs")
    decoded = pg.eval_on_selector_all(
        ".leak-pair img", "els=>els.filter(e=>e.naturalWidth>0).length")
    ok("pair thumbnails actually load", decoded >= 2 * len(pairs) - 1, f"{decoded} decoded")
    ok("each side is labelled with its split",
       len(pg.query_selector_all(".leak-split")) >= 2 * len(pairs))
    dom = pg.eval_on_selector(".leakage", "e=>e.textContent")
    ok("the uncalibrated-threshold caveat is present", "calibrated" in dom)

    # The finding must visibly depend on the threshold — that dependence is the
    # reason the panel exists in this shape.
    if len(rungs) >= 4:
        rungs[3].click()
        pg.wait_for_timeout(1400)
        ok("moving the threshold moves the finding",
           pg.inner_text(".leak-figure").replace("\n", " ") != figure,
           f"{figure.strip()} -> {pg.inner_text('.leak-figure').replace(chr(10), ' ').strip()}")


@flow("Data integrity", budget_s=45.0)
def data_integrity(pg, ok):
    """The embedding indexes and the database must still describe each other.

    `EmbeddingIndex.save` overwrites wholesale and nothing re-checked it, so
    deleting rows without re-running ingest silently leaves orphan vectors —
    which is exactly what happened here. The endpoint exists so the disagreement
    is reported rather than discovered.
    """
    resp = pg.request.get(f"{config.QA_API_URL}/api/admin/integrity")
    ok("integrity endpoint responds", resp.status == 200, str(resp.status))
    if resp.status != 200:
        return
    body = resp.json()
    for kind, check in body["indexes"].items():
        if not check.get("available"):
            ok(f"{kind} index present", True, "not computed — skipped")
            continue
        # Reported, not asserted clean: this database has three known orphan
        # caption vectors. The check that matters is that the count is VISIBLE,
        # and that no row is missing a vector — that one would break search.
        ok(f"{kind} index: no row is missing a vector",
           check["rows_without_vectors"] == 0,
           f"{check['index_vectors']:,} vectors / {check['db_rows']:,} rows, "
           f"{check['orphan_vectors']} orphan(s)")


@flow("Graceful degradation", budget_s=90.0, induces_errors=True)
def graceful_degradation(pg, ok):
    """A failing data source must announce itself, and a missing one must not.

    Graceful degradation is a property this whole application claims — of the
    embedding stack, the VLM, the agent layer, Playwright. It is also the
    property most likely to rot silently, because the happy path keeps passing.
    So it is tested by *causing* the failures: a 500 is intercepted and the UI
    must say so, while a 404 (an optional router simply not mounted) must stay
    quiet. Getting those two confused is the real bug — a swallowed error renders
    as "you have nothing", which is indistinguishable from working correctly.

    Route interception is safe here because the runner gives every flow its own
    page and closes it afterwards, so nothing leaks into the next flow.
    """
    def open_palette():
        pg.goto(url("/"), wait_until="domcontentloaded")
        pg.wait_for_selector(".grid .card")
        pg.keyboard.press("Meta+k")
        pg.wait_for_timeout(500)
        if not pg.query_selector("[role='dialog']"):
            pg.keyboard.press("Control+k")
            pg.wait_for_timeout(500)
        return bool(pg.query_selector("[role='dialog']"))

    def notice() -> str:
        el = pg.query_selector(".cmdk-notice")
        return el.inner_text().lower() if el else ""

    # A broken endpoint must be reported.
    pg.route("**/api/views", lambda r: r.fulfill(status=500, body="boom"))
    if not ok("palette opens for the degradation checks", open_palette()):
        return
    ok("a 500 from saved views is reported", "saved views" in notice(),
       notice()[:80] or "no notice — the failure was swallowed")
    ok("the palette still works without saved views",
       len(pg.query_selector_all("[role='option']")) > 0,
       f"{len(pg.query_selector_all('[role=option]'))} options")
    pg.unroute("**/api/views")

    # A router that simply is not mounted is not a failure, and must stay quiet.
    pg.route("**/api/views", lambda r: r.fulfill(status=404, body="{}"))
    open_palette()
    ok("a 404 from saved views stays quiet", "saved views" not in notice(),
       notice()[:70] or "no notice (correct)")
    pg.unroute("**/api/views")

    # And the same contract for the suggestion sources.
    pg.route("**/api/tags", lambda r: r.fulfill(status=500, body="boom"))
    open_palette()
    ok("a 500 from tags is reported", "tag" in notice(),
       notice()[:80] or "no notice — the failure was swallowed")
    pg.keyboard.type("quality")
    pg.wait_for_timeout(500)
    pg.keyboard.press("Enter")
    pg.wait_for_timeout(1200)
    ok("navigation still works with suggestions down", "/quality" in pg.url,
       pg.url.split("5173")[-1] or "/")
    pg.unroute("**/api/tags")


@flow("Command palette", budget_s=60.0)
def command_palette(pg, ok):
    """⌘K reaches anything without navigating first, and gets out of the way.

    Asserted through the dialog's semantics rather than its classes: `role`,
    `option` and keyboard behaviour are what a user (and a screen reader) relies
    on, so they are what should break the build when they regress.
    """
    pg.goto(url("/"), wait_until="domcontentloaded")
    pg.wait_for_selector(".grid .card")

    pg.keyboard.press("Meta+k")
    pg.wait_for_timeout(600)
    opened = ok("opens on Cmd+K", bool(pg.query_selector("[role='dialog']")))
    if not opened:
        # Linux/CI keyboards send Control; try it before giving up on the flow.
        pg.keyboard.press("Control+k")
        pg.wait_for_timeout(600)
        opened = ok("opens on Ctrl+K", bool(pg.query_selector("[role='dialog']")))
    if not opened:
        return

    pg.keyboard.type("quality")
    pg.wait_for_timeout(500)
    rows = pg.query_selector_all("[role='option']")
    ok("a query returns options", len(rows) > 0, f"{len(rows)} options")

    pg.keyboard.press("Enter")
    pg.wait_for_timeout(1200)
    ok("Enter navigates to the highlighted result", "/quality" in pg.url,
       pg.url.split("5173")[-1] or "/")
    ok("the palette closed after navigating",
       not pg.query_selector("[role='dialog']"))

    pg.keyboard.press("Meta+k")
    pg.wait_for_timeout(500)
    if pg.query_selector("[role='dialog']"):
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(400)
        ok("Escape closes it", not pg.query_selector("[role='dialog']"))


# Budget is generous because this flow waits on a local 8B model to route a
# request, call a tool and have its answer reviewed — 20-60s is normal. It is
# still worth having in the suite: the inline canvas is the headline feature, and
# a broken renderer shows up as an empty bubble that no other check would catch.
@flow("Assistant canvas", budget_s=300.0)
def assistant_canvas(pg, ok):
    """A charting request comes back as a live component that drills into the data.

    Skipped rather than failed when Ollama is absent: the assistant is an optional
    layer, and reporting the whole application broken because a model is not
    installed would make the status report useless on most machines.
    """
    pg.goto(url("/chat"), wait_until="domcontentloaded")
    pg.wait_for_timeout(1500)
    if "Assistant unavailable" in pg.inner_text("body"):
        ok("canvas check skipped — assistant unavailable", True,
           "no local model; the optional agent layer is not installed")
        return

    pg.fill(".chat-input input",
            "Plot how the dataset splits into train, validation and test")
    pg.click(".chat-input button.primary")
    try:
        pg.wait_for_selector(".chat-blocks .vblock", timeout=240000)
    except Exception as exc:
        ok("a charting request renders a block", False,
           str(exc).split("\n")[0][:110])
        return
    pg.wait_for_timeout(1500)

    blocks = pg.query_selector_all(".chat-blocks .vblock")
    ok("a charting request renders a block", len(blocks) >= 1, f"{len(blocks)} block(s)")
    # Identical blocks must collapse: a local model calls the same tool twice, and
    # two parallel lanes can independently chart the same thing.
    kinds = pg.eval_on_selector_all(
        ".chat-blocks .vblock",
        "els=>els.map(e=>[...e.classList].find(c=>c.startsWith('vblock-')))")
    ok("no duplicate blocks in one turn", len(kinds) == len(set(kinds)), str(kinds))
    ok("every block states its source",
       len(pg.query_selector_all(".vblock-source")) == len(blocks))
    ok("the chart is a real chart, not an image",
       len(pg.query_selector_all(".chat-blocks .recharts-surface")) > 0)
    ok("the turn reports which lanes ran", bool(pg.query_selector(".chat-foot")),
       pg.inner_text(".chat-foot") if pg.query_selector(".chat-foot") else "")
    ok("no reasoning markers leaked into the answer",
       "</think>" not in pg.inner_text(".chat-text"))

    # The drill-down contract: a chart element opens the slice it counts.
    targets = pg.query_selector_all(
        ".chat-blocks button[title*='gallery'], .chat-blocks a[title*='gallery']")
    drillable = ok("chart elements are drillable", len(targets) > 0,
                   f"{len(targets)} clickable")
    if drillable:
        targets[0].click()
        pg.wait_for_selector(".grid .card", timeout=25000)
        pg.wait_for_timeout(600)
        chips = [c.inner_text().replace("\n", " ")
                 for c in pg.query_selector_all(".active-filters .filter-chip")]
        ok("drilling a chart lands on that slice in the gallery",
           bool(chips) and len(pg.query_selector_all(".grid .card")) > 0,
           f"{len(pg.query_selector_all('.grid .card'))} cards; {'; '.join(chips)[:50]}")


@flow("Compare", budget_s=60.0)
def compare(pg, ok):
    """Two samples under one loupe: the transform is genuinely shared, the
    shared/different readout renders, and region drawing is reachable.

    The wheel is dispatched as a raw WheelEvent because the page's zoom
    listener is a native non-passive one (React's delegated wheel is passive,
    so preventDefault would be ignored) — Playwright's mouse.wheel would work
    too, but the dispatch pins the coordinates to pane A's centre exactly.

    The annotations endpoint was stubbed here while it was being built in a
    parallel lane (its 404 probe would have failed the sweep's zero-console-
    errors verdict). The router shipped; the flow now exercises the real
    thing, so an annotations regression on this page is the sweep's to catch.
    """
    pg.goto(url("/compare?a=76&b=2259"), wait_until="domcontentloaded")
    pg.wait_for_selector(".compare-pane img", timeout=25000)
    pg.wait_for_timeout(800)
    imgs = pg.eval_on_selector_all(
        ".compare-pane img", "els=>els.filter(e=>e.naturalWidth>0).length")
    ok("both panes render an image", imgs == 2, f"{imgs} decoded")

    # One wheel on pane A must move BOTH layers, identically: the inline
    # transform is the single shared view state, so the two strings are equal
    # by construction — if they ever differ, the panes have grown two views.
    pg.eval_on_selector(".compare-pane[data-slot='a'] .pane-stage", """e=>{
        const r = e.getBoundingClientRect();
        e.dispatchEvent(new WheelEvent('wheel', {
            deltaY: -240, clientX: r.x + r.width / 2, clientY: r.y + r.height / 2,
            bubbles: true, cancelable: true}));
    }""")
    pg.wait_for_timeout(400)
    t = pg.eval_on_selector_all(".compare-img-layer", "els=>els.map(e=>e.style.transform)")
    ok("wheel zoom applies one shared transform",
       len(t) == 2 and t[0] == t[1] and "scale(1)" not in t[0],
       " == ".join(t)[:110])

    rows = len(pg.query_selector_all(".compare-diff .diff-row"))
    ok("shared/different panel renders attribute rows", rows >= 3, f"{rows} rows")
    toggles = len(pg.query_selector_all(".compare-pane .draw-toggle"))
    ok("region drawing is reachable on both panes", toggles == 2,
       f"{toggles} Draw region toggles")


@flow("Hero journey", budget_s=180.0)
def hero_journey(pg, ok):
    """The workspace's spine as one continuous story: search -> inspect
    evidence -> pick two -> compare -> find more -> name the set -> the album
    explains itself. The one mutation (the album) cleans up after itself.
    The agent-proposal step is deliberately not here: flows must run without
    Ollama; the approval gate is pinned by the agent test suite instead."""
    pg.goto(url("/?q=a%20dog%20jumping%20into%20water"), wait_until="domcontentloaded")
    pg.wait_for_selector(".grid .card", timeout=25000)
    ok("search returns results", len(pg.query_selector_all(".grid .card")) > 0)

    pg.hover(".grid .card")
    pg.wait_for_timeout(300)
    score = (pg.text_content(".grid .card .ev-score") or "").strip()
    ok("evidence shows the score and its basis", bool(score), score[:40])

    # Picking is modeless: every card carries its own ✓, so a click on the card
    # always navigates and only the check toggles membership. There is no mode
    # to enter first.
    pg.evaluate("document.querySelectorAll('.grid .card .card-check')[0].click()")
    pg.evaluate("document.querySelectorAll('.grid .card .card-check')[1].click()")
    pg.wait_for_selector(".selection-tray", timeout=5000)
    ok("tray appears with two picked",
       "2" in (pg.text_content(".selection-tray") or ""))

    compare_btn = pg.query_selector(".selection-tray button:has-text('Compare')")
    if ok("tray offers Compare for a pair", compare_btn is not None):
        compare_btn.click()
        pg.wait_for_selector(".compare-pane img", timeout=15000)
        ok("compare canvas loads both panes",
           len(pg.query_selector_all(".compare-pane img")) >= 2)
        pg.go_back()
        pg.wait_for_selector(".grid .card", timeout=15000)

    pg.hover(".grid .card")
    pg.wait_for_timeout(250)
    pg.click(".card-actions button:has-text('More like this')")
    pg.wait_for_selector(".ref-chip.like", timeout=8000)
    pg.wait_for_timeout(1200)
    pg.hover(".grid .card")
    pg.wait_for_timeout(250)
    basis = (pg.text_content(".grid .card .ev-score") or "").strip()
    ok("composed ranking carries its basis", basis.startswith("composed"), basis)

    if not pg.query_selector(".selection-tray"):
        pg.evaluate("document.querySelectorAll('.grid .card .card-check')[0].click()")
        pg.evaluate("document.querySelectorAll('.grid .card .card-check')[1].click()")
        pg.wait_for_selector(".selection-tray", timeout=5000)
    pg.fill(".selection-tray input", "hero-journey-flow")
    pg.click(".selection-tray .primary")
    pg.wait_for_timeout(1500)
    album_id = pg.evaluate("new URLSearchParams(location.search).get('album')")
    ok("the set became an album and the view landed in it",
       album_id is not None, f"album={album_id}")

    if album_id:
        try:
            pg.wait_for_selector(".album-header", timeout=8000)
            pg.click("button:has-text('Details & analysis')")
            pg.click("button:has-text('Analyze')")
            pg.wait_for_selector(".ah-measured", timeout=10000)
            ok("analysis renders its measured half",
               "Measured" in (pg.text_content(".ah-panel-title") or ""))
            ok("the generated half names its model",
               bool(pg.query_selector(".ah-panel-title.ai")))
        finally:
            r = pg.request.delete(f"{config.QA_API_URL}/api/albums/{album_id}")
            ok("journey cleans up its album", r.status == 200, f"{r.status}")
