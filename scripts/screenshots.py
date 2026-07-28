"""Regenerate the figures in `docs/screenshots/`, used by the README.

Committed rather than ad-hoc so the figures can be reproduced instead of trusted.
Each entry below names the exact URL it came from, which means a reviewer can
check any claim the README makes by opening the same address.

    # with the backend on :8000 and the dev server on :5173
    cd backend && .venv/bin/python ../scripts/screenshots.py

Add `--headed` to watch it drive the browser. Uses the system Chrome via
Playwright's `channel="chrome"`, so no browser download is needed.

Every shot is taken at a fixed 1600x1000 viewport at 2x, because a figure whose
layout depends on the machine that captured it cannot be compared against a later
one. Screenshots are deliberately *not* full-page: the point of most of these is
what a reviewer sees without scrolling.
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PWTimeout

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
BASE = "http://localhost:5173"
API = "http://localhost:8000"
VIEWPORT = {"width": 1600, "height": 1000}


# Captured at 2x for legible text, then downscaled to this width. The 2x plates
# came to 20 MB for eleven figures, which is a real cost on every clone of a repo
# whose entire point is being small enough to run locally — and the README never
# displays them wider than this anyway. Resampled with LANCZOS so the 10px
# monospace in the evidence strips survives the reduction.
FIGURE_WIDTH = 1600


def _shrink(path: Path) -> tuple[int, int]:
    """Downscale a captured plate in place. Returns (before, after) in bytes."""
    from PIL import Image

    before = path.stat().st_size
    with Image.open(path) as im:
        if im.width > FIGURE_WIDTH:
            h = round(im.height * FIGURE_WIDTH / im.width)
            im = im.resize((FIGURE_WIDTH, h), Image.LANCZOS)
        im.save(path, optimize=True)
    return before, path.stat().st_size


def _tags() -> dict[str, int]:
    with urllib.request.urlopen(f"{API}/api/tags", timeout=15) as r:
        return {t["name"]: t["count"] for t in json.load(r)}


def _drop_tag(name: str) -> None:
    """Remove a tag from every sample carrying it.

    There is no bulk delete — `DELETE /api/samples/{id}/tags/{name}` is per
    sample — so this asks which samples hold it and unpicks them one at a time.
    """
    url = f"{API}/api/samples?tag={urllib.parse.quote(name)}&page_size=200"
    with urllib.request.urlopen(url, timeout=30) as r:
        ids = [s["id"] for s in json.load(r)["items"]]
    for sid in ids:
        req = urllib.request.Request(
            f"{API}/api/samples/{sid}/tags/{urllib.parse.quote(name)}",
            method="DELETE")
        try:
            urllib.request.urlopen(req, timeout=15).close()
        except Exception as exc:                          # noqa: BLE001
            print(f"    could not untag {sid}: {exc}")
    print(f"    reverted the assistant's `{name}` tag on {len(ids)} samples")


def _open_describe(page: Page) -> None:
    page.locator("details.set-summary > summary").click()
    page.wait_for_timeout(1800)


def _run_benchmark(page: Page) -> None:
    """Click through and wait for the table.

    Unclicked, the page is an explanation and a button — it documents the
    protocol and shows none of the result, which is the one thing a reviewer
    wants to see. The run encodes 1,000 captions through SigLIP and scores three
    modes, so it takes minutes, not seconds.
    """
    page.get_by_role("button", name="Run benchmark").click()
    page.wait_for_selector("table.eval-table", timeout=600_000)
    page.wait_for_timeout(800)


def _open_palette(page: Page) -> None:
    page.keyboard.press("Meta+k")
    page.wait_for_timeout(600)
    page.keyboard.type("night", delay=45)
    page.wait_for_timeout(900)


def _ask_assistant(page: Page) -> None:
    """Ask a real question and wait for the real answer.

    Addressed by aria-label, not `input[type=text]` — the element declares no
    `type`, so the attribute selector matches nothing even though the browser
    treats it as a text input.

    A screenshot of an empty composer would show the chrome and none of the
    point, which is that the agent answers with interactive blocks rather than
    prose. This needs Ollama on :11434; without it the page renders its degraded
    message, which is a legitimate figure too — so a timeout here is not fatal.

    The turn is *not* read-only. Asked for the worst-agreement images, the agent
    routes through `insights -> suspect_captions`, `retrieval -> search_images`
    and then `retrieval -> tag_samples` — it writes a `low-agreement` tag as a
    convenience, which is good agent behaviour and bad behaviour for a script a
    reviewer might run. So the tags are diffed around the turn and anything new
    is reverted; regenerating a figure must not leave the dataset different.
    """
    before = _tags()
    box = page.get_by_label("Ask the dataset assistant")
    box.click()
    # Phrased to ask for the *set*, not for an explanation. Asked "which images",
    # a local 8B model tends to enumerate all twelve in prose and push the image
    # block off the fold — and on one run it volunteered that the scores come
    # from "BLEU/ROUGE-like metrics", which they do not (they are SigLIP cosines).
    # Asking it to show them keeps the answer a block, which is the point of the
    # figure. The turn is still model-dependent: re-run if the figure is poor.
    box.type("Show me the 12 images with the worst caption agreement.", delay=12)
    page.keyboard.press("Enter")
    try:
        # The turn is done when the typing indicator goes away. A local 8B model
        # with tool calls takes 10-60s, well past any default timeout.
        page.wait_for_selector(".loading-dots", state="detached", timeout=120_000)
    except PWTimeout:
        print("    (assistant still thinking — capturing as-is)")
    page.wait_for_timeout(1200)
    # The pane auto-scrolls to the newest content, which puts the question
    # off-screen and leaves a figure of an answer to nothing. Pull the user turn
    # back to the top so the exchange reads as an exchange.
    page.locator(".chat-turn.user").last.scroll_into_view_if_needed()
    page.evaluate("() => { const s = document.querySelector('.chat-scroll');"
                  " if (s) s.scrollTop -= 24; }")
    page.wait_for_timeout(500)
    for name in set(_tags()) - set(before):
        _drop_tag(name)


def _show_floor(page: Page) -> None:
    """Bring the similar grid into frame: section head at the top of the
    viewport, so the above-floor cards, the divider and the greyed tail all
    land inside the fixed 1600x1000 capture."""
    page.wait_for_selector(".sim-divider")
    page.eval_on_selector(".similar-head",
                          "el => el.scrollIntoView({block: 'start'})")
    page.wait_for_timeout(1200)          # lazy thumbnails below the fold


# (filename, url, settle_ms, extra_action)
SHOTS = [
    ("1-gallery.png", "/?q=a+crowded+street+at+night&mode=hybrid", 3000, None),
    ("2-describe.png",
     "/?attr=time_of_day%3Anight&attr=setting%3Aindoor", 2200, _open_describe),
    ("4-axes.png", "/?difficulty_min=8&legibility_min=8", 2600, None),
    ("5-map.png", "/map", 6000, None),
    ("6-quality.png", "/quality", 4000, None),
    ("7-stats.png", "/stats", 4000, None),
    ("8-eval.png", "/eval", 1500, _run_benchmark),
    ("9-sample.png", "/samples/1865", 3500, None),
    ("10-palette.png", "/", 1800, _open_palette),
    ("11-assistant.png", "/chat", 2500, _ask_assistant),
    # A search click-through arrives carrying its provenance in the URL, so the
    # banner in this figure is exactly what a pasted link reproduces.
    ("12-provenance.png",
     "/samples/6065?src=search&q=a+crowded+street+at+night&mode=hybrid&rank=1",
     3500, None),
    # A sample whose neighbours straddle the similarity floor: real class above
    # the divider, greyed context below it. The grid sits below the inspector,
    # so the shot scrolls to it — a figure about the floor must show the floor.
    ("13-floor.png", "/samples/76", 3500, _show_floor),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headed", action="store_true", help="show the browser")
    ap.add_argument("--only", help="substring filter on the output filename")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    shots = [s for s in SHOTS if not args.only or args.only in s[0]]
    failures: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=not args.headed)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)

        # A 4xx/5xx behind a screenshot is invisible in the image but makes the
        # figure a lie, so the run reports one rather than saving quietly.
        bad: list[str] = []
        page.on("response", lambda r: bad.append(f"{r.status} {r.url}")
                if r.status >= 400 and "/api/" in r.url else None)

        for name, url, settle, action in shots:
            bad.clear()
            print(f"  {name:18s} {url}")
            try:
                page.goto(f"{BASE}{url}", wait_until="networkidle")
                page.wait_for_timeout(settle)
                if action is not None:
                    action(page)
                page.screenshot(path=str(OUT / name))
                was, now = _shrink(OUT / name)
                print(f"    {was // 1024} kB -> {now // 1024} kB")
            except Exception as exc:                      # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print(f"    FAILED: {exc}")
                continue
            if bad:
                failures.append(f"{name}: API errors {bad[:3]}")
                print(f"    API ERRORS: {bad[:3]}")

        browser.close()

    if failures:
        print("\n  Figures that are not trustworthy:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print(f"\n  {len(shots)} figures written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
