"""Re-measure the zero-shot detector's headline numbers on this machine.

The per-image latency is asserted in three places — app/ml/detect.py,
GET /api/detect/status (which puts it on screen) and the README — so it needs
the same re-runnable harness every other headline number in this repo has.
Writing it caught the first version of that claim: 256 ms was the fastest
single image, not the warm median, which is ~330 ms.
This is that harness, and it is a client of the shipping code path: it calls
`detect_ml.get_detector().detect(...)`, never a private copy of it.

Measured, in one fresh subprocess so the load is genuinely cold:

  - cold load: weights off disk onto the device, first `get_detector()`,
  - warm per-image latency: median/p95 over N real corpus images, after a
    discarded warm-up call (the first forward compiles the Metal graph),
  - resident memory: peak RSS, plus MPS driver memory where available,
  - boxes found per image, so a fast run that detects nothing is visible as
    such rather than passing as a good number.

Weights are never downloaded. If they are not cached the script says so and
names the fetch command, exactly as the request path does.

Output: a markdown table on stdout and a JSON artifact under
backend/data/cache/bench_detector.json.

    cd backend && .venv/bin/python ../scripts/bench_detector.py [--images 20]
"""
import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"

DEFAULT_QUERIES = "a person. an animal. a vehicle. an object."


def _peak_rss_gb() -> float:
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2), 2)


def child(args) -> int:
    sys.path.insert(0, str(BACKEND))

    from PIL import Image

    from app import config, db
    from app.ml import detect as detect_ml

    out: dict = {"model": detect_ml.DETECT_MODEL, "queries": args.queries}

    ready, reason = detect_ml.detect_ready()
    if not ready:
        # Honest degradation, same contract as the request path: name the
        # reason and the enabling command, fetch nothing.
        out["unavailable"] = reason
        out["fetch_hint"] = detect_ml.FETCH_HINT
        print(json.dumps(out))
        return 0

    conn = db.connect()
    files = [r["filename"] for r in conn.execute(
        "SELECT filename FROM samples ORDER BY id LIMIT ?", (args.images + 1,))]
    conn.close()
    paths = [config.IMAGES_DIR / f for f in files if (config.IMAGES_DIR / f).exists()]
    if len(paths) < 2:
        out["unavailable"] = (
            f"fewer than 2 corpus images readable under {config.IMAGES_DIR} — "
            "run `python -m app.ingest`")
        print(json.dumps(out))
        return 0

    t0 = time.perf_counter()
    detector = detect_ml.get_detector()
    out["cold_load_s"] = round(time.perf_counter() - t0, 2)
    if detector is None:
        out["unavailable"] = "weights are cached but the detector failed to load — see server log"
        print(json.dumps(out))
        return 0
    out["device"] = str(detector.device)

    images = [Image.open(p).convert("RGB") for p in paths]
    # The first forward compiles the graph; it is a load cost, not a warm cost.
    warm = time.perf_counter()
    detector.detect(images[0], args.queries)
    out["first_call_s"] = round(time.perf_counter() - warm, 2)

    lat, counts = [], []
    for im in images[1:]:
        t = time.perf_counter()
        boxes = detector.detect(im, args.queries)
        lat.append((time.perf_counter() - t) * 1000)
        counts.append(len(boxes))
    out["images"] = len(lat)
    out["ms_p50"] = round(statistics.median(lat), 1)
    out["ms_p95"] = round(
        statistics.quantiles(lat, n=20)[18] if len(lat) >= 20 else max(lat), 1)
    out["ms_min"] = round(min(lat), 1)
    out["ms_max"] = round(max(lat), 1)
    out["boxes_per_image"] = round(statistics.mean(counts), 1)
    out["images_with_no_box"] = sum(1 for c in counts if c == 0)
    out["peak_rss_gb"] = _peak_rss_gb()
    try:
        import torch
        if torch.backends.mps.is_available():
            out["mps_driver_gb"] = round(
                torch.mps.driver_allocated_memory() / 1024 ** 3, 2)
    except Exception:
        pass
    print(json.dumps(out))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=20,
                    help="corpus images timed after the warm-up call")
    ap.add_argument("--queries", default=DEFAULT_QUERIES,
                    help="period-separated phrases, the model's query format")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return child(args)

    print("— measuring the detector in a fresh process …", file=sys.stderr)
    proc = subprocess.run(
        [sys.executable, __file__, "--child", "--images", str(args.images),
         "--queries", args.queries],
        capture_output=True, text=True, cwd=BACKEND)
    if proc.returncode != 0 or not proc.stdout.strip():
        print(proc.stderr.strip()[-800:], file=sys.stderr)
        print("\nthe measuring subprocess failed — no numbers to report")
        return 1
    r = json.loads(proc.stdout.strip().splitlines()[-1])

    artifact = BACKEND / "data" / "cache" / "bench_detector.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(r, indent=1))

    print(f"\nmodel: {r['model']}\nqueries: {r['queries']!r}")
    if "unavailable" in r:
        # A missing number is a fact, not a blank — and never a download.
        print(f"\nNOT MEASURED: {r['unavailable']}")
        hint = r.get("fetch_hint")
        if hint and hint not in r["unavailable"]:
            print(f"fetch the weights with:\n  {hint}")
        print(f"\nartifact: {artifact}")
        return 0

    print(f"device: {r['device']}   images timed: {r['images']}\n")
    print("| cold load s | first call s | warm ms p50 | p95 | min | max "
          "| boxes/img | empty | peak RSS GB | MPS GB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print(f"| {r['cold_load_s']} | {r['first_call_s']} | {r['ms_p50']} "
          f"| {r['ms_p95']} | {r['ms_min']} | {r['ms_max']} "
          f"| {r['boxes_per_image']} | {r['images_with_no_box']} "
          f"| {r['peak_rss_gb']} | {r.get('mps_driver_gb', '—')} |")
    print(f"\nartifact: {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
