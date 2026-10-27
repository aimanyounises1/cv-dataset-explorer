"""Measure the retrieval providers against each other on this machine.

For every provider that resolves (siglip2, qwen3_vl), a fresh subprocess with
CVDE_EMBED_PROVIDER pinned runs:

  - model load time,
  - single-query text-encode latency (median/p95 over 100 real captions),
  - single-image encode latency (median over 20 corpus images),
  - the repository's own retrieval benchmark (R@1/5/10, text->image, the
    protocol in app/api/eval.py — this script is a client of it, never a fork),
  - peak RSS and, where available, MPS driver memory,
  - index size on disk and the ingest encode timings the manifest recorded.

Output: a markdown table on stdout and a JSON artifact (choose a reviewable
location with --output; the default is backend/data/cache/bench_providers.json).
The complete retrieval response is retained, including its protocol and inputs.
Providers that do not resolve are
reported with their named reason — a missing number is a fact, not a blank.

    cd backend && .venv/bin/python ../scripts/bench_providers.py [--sample-size 1000]
"""
import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
DEFAULT_ARTIFACT = BACKEND / "data" / "cache" / "bench_providers.json"
ARTIFACT_VERSION = 1
DEFAULT_SAMPLE_SIZE = 1000
MIN_SAMPLE_SIZE = 50
MAX_SAMPLE_SIZE = 5000
TEXT_QUERY_COUNT = 100
IMAGE_QUERY_COUNT = 20
PERCENTILE_INTERVALS = 20


def child(provider: str, sample_size: int) -> int:
    os.environ["CVDE_EMBED_PROVIDER"] = provider
    sys.path.insert(0, str(BACKEND))
    os.chdir(BACKEND)

    from app import config, db
    from app.ml import providers

    out: dict = {"provider": provider}
    t0 = time.perf_counter()
    runtime = providers.get_retrieval_bundle()
    out["load_s"] = round(time.perf_counter() - t0, 2)
    if runtime is None or runtime.provider != provider:
        state = providers.resolve()
        out["unavailable"] = (
            state.reasons.get(provider)
            or state.fallback_reason
            or f"resolved {getattr(runtime, 'provider', None)!r}, not {provider!r}"
        )
        print(json.dumps(out))
        return 0
    encoder = runtime.encoder
    out["model_id"] = runtime.model_id
    out["dim"] = runtime.manifest["dim"]

    conn = db.connect()
    captions = [r["text"] for r in conn.execute(
        "SELECT text FROM captions ORDER BY id LIMIT ?", (TEXT_QUERY_COUNT,))]
    files = [r["filename"] for r in conn.execute(
        "SELECT filename FROM samples ORDER BY id LIMIT ?", (IMAGE_QUERY_COUNT,))]
    conn.close()
    if len(captions) < 2 or not files:
        out["unavailable"] = "Latency measurement requires two captions and one image."
        print(json.dumps(out))
        return 0

    encoder.encode_texts(captions[:1])  # warm-up
    lat = []
    for text in captions:
        t = time.perf_counter()
        encoder.encode_texts([text])
        lat.append((time.perf_counter() - t) * 1000)
    out["text_ms_p50"] = round(statistics.median(lat), 1)
    out["text_ms_p95"] = round(
        statistics.quantiles(lat, n=PERCENTILE_INTERVALS)[-1], 1)
    out["text_latency_queries"] = len(lat)

    from PIL import Image
    imgs = [Image.open(config.IMAGES_DIR / f).convert("RGB") for f in files]
    encoder.encode_images(imgs[:1])  # warm-up
    ilat = []
    for im in imgs:
        t = time.perf_counter()
        encoder.encode_images([im])
        ilat.append((time.perf_counter() - t) * 1000)
    out["image_ms_p50"] = round(statistics.median(ilat), 1)
    out["image_latency_queries"] = len(ilat)

    # The repository's own benchmark, through the real route.
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as client:
        response = client.get("/api/eval/retrieval", params={"sample_size": sample_size})
        response.raise_for_status()
        r = response.json()
    out["retrieval"] = r
    if r.get("available"):
        out["recall"] = {row["mode"]: {"r1": row["recall_at"]["1"],
                                       "r5": row["recall_at"]["5"],
                                       "r10": row["recall_at"]["10"]}
                         for row in r.get("results", [])
                         if row.get("recall_at")}
        out["benchmark_sample"] = r.get("sample_size")
    else:
        out["recall_unavailable"] = r.get("message")

    emb_dir = runtime.emb_dir
    out["index_mb"] = round(sum(
        f.stat().st_size for f in emb_dir.glob("*.npy")) / 1e6, 1)
    manifest = providers.read_manifest(emb_dir)
    if manifest:
        out["ingest_image_encode_s"] = manifest.get("image_encode_seconds")
        out["ingest_caption_encode_s"] = manifest.get("caption_encode_seconds")
        out["sim_floor_p10"] = manifest.get("sim_floor_p10")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform in {"darwin", "linux"}:
        # getrusage reports bytes on macOS and KiB on Linux.
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        out["peak_rss_gb"] = round(rss_bytes / 1024 ** 3, 2)
    else:
        out["peak_rss_unavailable"] = "RSS units are not defined for this platform."
    try:
        import torch
        if torch.backends.mps.is_available():
            out["mps_driver_gb"] = round(
                torch.mps.driver_allocated_memory() / 1024 ** 3, 2)
    except (ImportError, RuntimeError) as exc:
        out["mps_memory_unavailable"] = str(exc)
    print(json.dumps(out))
    return 0


def write_artifact(path: Path, results: list[dict], sample_size: int) -> None:
    """Export complete provider results with the source revision and run context.

    Filesystem errors propagate so a failed export cannot look successful.
    This does not claim that synthetic tests or a stored-vector baseline are
    full-data model measurements: each retrieval response names its own mode.
    """
    if not MIN_SAMPLE_SIZE <= sample_size <= MAX_SAMPLE_SIZE:
        raise ValueError("sample_size is outside the retrieval endpoint's supported range")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=BACKEND,
        capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=BACKEND,
        capture_output=True, text=True, check=False,
    )
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "python_version": sys.version,
        "platform": sys.platform,
        "sample_size": sample_size,
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    ap.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT,
                    help="JSON destination; choose a path outside generated data to share results")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not MIN_SAMPLE_SIZE <= args.sample_size <= MAX_SAMPLE_SIZE:
        ap.error(f"--sample-size must be between {MIN_SAMPLE_SIZE} and {MAX_SAMPLE_SIZE}")
    if args.child:
        return child(args.child, args.sample_size)

    results = []
    for provider in ("siglip2", "qwen3_vl"):
        print(f"— measuring {provider} …", file=sys.stderr)
        proc = subprocess.run(
            [sys.executable, __file__, "--child", provider,
             "--sample-size", str(args.sample_size)],
            capture_output=True, text=True, cwd=BACKEND, check=False)
        if proc.returncode != 0:
            results.append({"provider": provider,
                            "error": proc.stderr.strip()[-400:]})
            continue
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    write_artifact(args.output, results, args.sample_size)

    print("\n| provider | dim | load s | text ms p50/p95 | image ms p50 "
          "| semantic R@1/5/10 | hybrid R@1/5/10 | index MB | ingest s (img+cap) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if "unavailable" in r or "error" in r:
            print(f"| {r['provider']} | — | — | — | — | "
                  f"{r.get('unavailable') or r.get('error')} | — | — | — |")
            continue

        def rk(mode, row=r):
            m = row.get("recall", {}).get(mode)
            return (f"{m['r1']:.1%}/{m['r5']:.1%}/{m['r10']:.1%}" if m
                    else row.get("recall_unavailable", "n/a"))
        ing = "—"
        if r.get("ingest_image_encode_s") is not None:
            ing = f"{r['ingest_image_encode_s']:.0f}+{r.get('ingest_caption_encode_s') or 0:.0f}"
        print(f"| {r['provider']} | {r['dim']} | {r['load_s']} "
              f"| {r['text_ms_p50']}/{r['text_ms_p95']} | {r['image_ms_p50']} "
              f"| {rk('semantic')} | {rk('hybrid')} | {r['index_mb']} | {ing} |")
    print(f"\nartifact: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
