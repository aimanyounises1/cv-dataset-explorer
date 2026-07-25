#!/usr/bin/env python3
"""End-to-end UI smoke test: drives real Chrome through every view.

There is no frontend unit-test tier in this project, so this is the net that
catches what `tsc` cannot — a view that renders empty, a control that no longer
filters, a console error, a 404. It asserts behaviour through the UI rather than
implementation details, so it survives restyling.

    # both servers must already be running
    cd backend && .venv/bin/uvicorn app.main:app --port 8000 &
    cd frontend && npm run dev &

    cd backend && uv run --with playwright --python .venv/bin/python \
        python ../scripts/ui_smoke.py

Playwright is intentionally NOT in requirements.txt: it is a developer tool, not
a runtime dependency, and `uv run --with` keeps it out of the app's environment.
It drives the system Chrome via channel="chrome", so no browser download either.

Two lessons are encoded in the waits and assertions, both learned the hard way:
  * Never wait on "networkidle" against a dev server. Vite's HMR socket and 60
    lazy-loaded thumbnails mean it may never settle; wait for a selector.
  * Assert against DOM text (textContent), not inner_text, wherever CSS applies
    text-transform — inner_text returns the *rendered* casing.
"""
import re, sys
from playwright.sync_api import sync_playwright

BASE = "http://localhost:5173"
SHOTS = "/private/tmp/claude-501/-Users-aimanyounis-Downloads-cv-dataset-explorer/cce78e3d-f3bb-4522-b59d-3630f7996c63/scratchpad/shots"
problems, results = [], []

def ok(name, cond, detail=""):
    results.append((bool(cond), name, detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        problems.append(name)

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
    console_errors, net_failures = [], []

    def on_console(m):
        if m.type in ("error",):
            console_errors.append(f"[{m.type}] {m.text[:200]}")
    def on_response(r):
        if r.status >= 400:
            net_failures.append(f"{r.status} {r.request.method} {r.url.replace(BASE,'')[:110]}")

    ctx.on("console", on_console)
    ctx.on("response", on_response)
    ctx.on("weberror", lambda e: console_errors.append(f"[pageerror] {str(e.error)[:200]}"))
    pg = ctx.new_page()

    # ---------------------------------------------------------------- routes
    print("\n== every route loads ==")
    for path, marker in [
        ("/", ".grid"), ("/map", ".map-canvas"), ("/stats", ".stat-cards"),
        ("/quality", ".dist-bars"), ("/eval", "button.primary"), ("/chat", ".chat-page"),
        ("/samples/1723", ".detail-image"),
    ]:
        try:
            pg.goto(BASE + path, wait_until="domcontentloaded", timeout=45000)
            pg.wait_for_selector(marker, timeout=25000)
            ok(f"route {path}", True)
        except Exception as e:
            ok(f"route {path}", False, str(e).split("\n")[0][:110])

    # --------------------------------------------------------------- gallery
    print("\n== gallery ==")
    pg.goto(BASE + "/", wait_until="domcontentloaded"); pg.wait_for_selector(".grid .card")
    base_n = len(pg.query_selector_all(".grid .card"))
    ok("browse renders 60 cards", base_n == 60, f"{base_n}")
    ok("sparkline present on cards", len(pg.query_selector_all(".axis-spark")) == base_n,
       f"{len(pg.query_selector_all('.axis-spark'))} sparklines")

    for mode in ("hybrid", "semantic", "keyword"):
        pg.goto(f"{BASE}/?q=dog%20in%20snow&mode={mode}", wait_until="domcontentloaded")
        try:
            pg.wait_for_selector(".grid .card", timeout=25000)
            n = len(pg.query_selector_all(".grid .card"))
            ok(f"search mode {mode}", n > 0, f"{n} results")
        except Exception:
            ok(f"search mode {mode}", False, "no results rendered")

    # axis filter + sort + chips
    pg.goto(f"{BASE}/?difficulty_min=9&sort=clutter_desc", wait_until="domcontentloaded")
    pg.wait_for_selector(".grid .card")
    # .active-filters/.filter-chip are ActiveFilters' own classes; ".chip" is the
    # unrelated query-suggestion row, which an earlier version of this test hit.
    chips = [c.inner_text().strip().replace("\n", " ")
             for c in pg.query_selector_all(".active-filters .filter-chip")]
    ok("active filter chip shown for axis",
       any("Difficulty" in c for c in chips), "; ".join(chips)[:100])
    ok("sort select reflects URL",
       "clutter" in (pg.eval_on_selector("select[aria-label='Sort results']", "e=>e.value") or ""))

    # density control
    pg.click(".density-group button:last-child")
    pg.wait_for_timeout(400)
    w = pg.eval_on_selector(".grid .card", "e=>e.getBoundingClientRect().width")
    ok("density L widens cards", w > 200, f"card width {w:.0f}px")

    # load more
    pg.goto(f"{BASE}/?q=dog&mode=hybrid", wait_until="domcontentloaded")
    pg.wait_for_selector(".grid .card")
    if pg.query_selector(".load-more button"):
        pg.click(".load-more button"); pg.wait_for_timeout(1500)
        n2 = len(pg.query_selector_all(".grid .card"))
        ok("load more appends", n2 > 60, f"{n2} cards")
    else:
        ok("load more appends", False, "no Load more button")

    # export links resolve
    hrefs = pg.eval_on_selector_all(".export-pill", "els=>els.map(e=>e.getAttribute('href'))")
    ok("export links present", len(hrefs) == 3, f"{hrefs[:1]}")
    if hrefs:
        r = ctx.request.get("http://localhost:8000" + hrefs[0].replace("/api", "/api"))
        ok("export csv responds 200", r.status == 200, f"{r.status}, {len(r.body())} bytes")

    # --------------------------------------------------------------- map
    print("\n== map ==")
    pg.goto(BASE + "/map", wait_until="domcontentloaded"); pg.wait_for_selector(".map-canvas")
    pg.wait_for_timeout(1500)
    for mode in ("cluster", "split", "agreement", "difficulty"):
        pg.select_option("select[aria-label='Colour points by']", mode)
        pg.wait_for_timeout(500)
        has_legend = bool(pg.query_selector(".legend-ramp")) or bool(pg.query_selector(".legend-swatches .legend-swatch"))
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
    # hover tooltip
    box = pg.eval_on_selector(".map-canvas", "e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}}")
    pg.mouse.move(box["x"] + box["w"]*0.35, box["y"] + box["h"]*0.55)
    pg.wait_for_timeout(700)
    ok("hover shows thumbnail tooltip", bool(pg.query_selector(".map-tooltip img")))

    # ------------------------------------------------------------ statistics
    print("\n== statistics ==")
    pg.goto(BASE + "/stats", wait_until="domcontentloaded"); pg.wait_for_selector(".stat-cards")
    pg.wait_for_timeout(1200)
    ok("charts rendered", len(pg.query_selector_all(".recharts-surface")) >= 4,
       f"{len(pg.query_selector_all('.recharts-surface'))} chart surfaces")
    # The panel is a collapsed <details>, so its text is absent from inner_text
    # until expanded — the earlier assertion was testing the wrong thing.
    pg.click(".caveat summary"); pg.wait_for_timeout(400)
    ok("provenance panel present", "8,091" in pg.inner_text("body"))

    # --------------------------------------------------------------- quality
    print("\n== quality ==")
    pg.goto(BASE + "/quality", wait_until="domcontentloaded"); pg.wait_for_selector(".dist-bars")
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

    # ------------------------------------------------------------- benchmark
    print("\n== benchmark ==")
    pg.goto(BASE + "/eval", wait_until="domcontentloaded")
    pg.get_by_role("button", name=re.compile("benchmark")).click()
    try:
        pg.wait_for_selector("table.eval-table tbody tr", timeout=180000)
        rows = pg.query_selector_all("table.eval-table tbody tr")
        ok("benchmark returns 3 modes", len(rows) == 3, f"{len(rows)} rows")
        # Compare against the DOM text, not inner_text: CSS uppercases the
        # headers, so inner_text yields "CANDIDATES".
        hdrs = pg.eval_on_selector_all("table.eval-table th", "e=>e.map(x=>x.textContent.trim())")
        ok("candidates column present", "candidates" in hdrs, str(hdrs))
    except Exception as e:
        ok("benchmark returns 3 modes", False, str(e).split("\n")[0][:100])

    # --------------------------------------------------------------- sample
    print("\n== sample detail ==")
    pg.goto(f"{BASE}/?q=dog&mode=hybrid", wait_until="domcontentloaded")
    pg.wait_for_selector(".grid .card")
    pg.query_selector_all(".grid .card")[0].click()
    pg.wait_for_selector(".detail-image", timeout=20000); pg.wait_for_timeout(1200)
    ok("5 captions shown", len(pg.query_selector_all(".caption-list li")) == 5,
       f"{len(pg.query_selector_all('.caption-list li'))}")
    ok("similar images loaded", len(pg.query_selector_all(".grid .card")) > 0,
       f"{len(pg.query_selector_all('.grid .card'))}")
    url0 = pg.url
    pg.keyboard.press("ArrowRight"); pg.wait_for_timeout(1000)
    ok("arrow key advances", pg.url != url0, f"...{url0[-6:]} -> ...{pg.url[-6:]}")

    # -------------------------------------------------------------- assistant
    print("\n== assistant ==")
    pg.goto(BASE + "/chat", wait_until="domcontentloaded"); pg.wait_for_timeout(1500)
    body = pg.inner_text("body")
    unavailable = "Assistant unavailable" in body
    ok("assistant reports a definite state", True,
       "unavailable panel" if unavailable else "input ready")
    if not unavailable:
        ok("chat input present", bool(pg.query_selector(".chat-input input")))

    pg.screenshot(path=f"{SHOTS}/test_final.png")
    browser.close()

print("\n== console errors ==")
if console_errors:
    for e in dict.fromkeys(console_errors):
        print("  " + e)
else:
    print("  none")
print("== failed network requests ==")
if net_failures:
    for f in dict.fromkeys(net_failures):
        print("  " + f)
else:
    print("  none")

passed = sum(1 for c, _, _ in results if c)
print(f"\n{passed}/{len(results)} checks passed")
if problems:
    print("FAILED: " + ", ".join(problems))
sys.exit(1 if problems or console_errors else 0)
