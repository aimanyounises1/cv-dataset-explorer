"""Measure SAM 2.1 promptable segmentation on this machine, then decide.

Written to answer one question with numbers instead of adjectives: can real
promptable segmentation ship on top of the existing region/detector flow?
It is the same harness shape as `scripts/bench_detector.py` — one fresh
subprocess so the load is genuinely cold, warm p50/p95 over real corpus
images, peak RSS, and honest degradation that names the fetch command instead
of downloading anything.

Four things are measured, because three of them can each independently
disqualify the model and the fourth decides whether it is worth shipping at
all:

  1. Dependency cost. SAM2 must come from the *pinned* transformers, or it is
     a new dependency and therefore out — this repository adds none.
     The script prints the transformers version it used and the model class.
  2. Speed and memory: cold load, warm ms per mask for a BOX prompt and for a
     POINT prompt, peak RSS, MPS driver bytes.
  3. Correctness on real data, not a demo truck: masks are prompted with the
     shipping detector's own boxes (`app.ml.detect`) on corpus images, and
     overlay PNGs are written so a human can look at them. `iou_score` is the
     model's self-reported quality; `mask_frac_of_box` is the share of the
     prompt rectangle the mask actually keeps.
  4. THE PRODUCT NUMBER — `rect_vs_masked_cos`, `top10_overlap` and
     `on_target`. Retrieval here consumes rectangular crops (POST
     /api/search/by-region embeds `img.crop(box)`). So the only thing a mask
     can buy a *user of this tool* is a different ranking. This measures
     exactly that: embed the rectangle crop, embed the same crop with
     everything outside the mask erased, and compare the cosine between the
     two query vectors and the overlap of the top-10 they retrieve from the
     real index.

     A different ranking is not automatically a better one, and saying "the
     mask changed the results" as if that were an improvement is the exact
     unmeasured causal claim this repo forbids. So `on_target` scores both
     rankings against a caption-word proxy: for a region the detector called
     "an animal", how many of the top 10 have a caption mentioning an animal?
     It is coarse — Flickr8k captions are free text and the detector's four
     phrases are broad — and it is reported as a proxy, not as accuracy. Its
     job is only to answer "better, worse, or a wash", and a wash means the
     mask is decoration.

Stability is checked the way a demo would break it: N interleaved box/point
calls with SigLIP image encodes in between, because both modules share one
Metal device and `ml/embedder.py` holds an inference lock precisely because
concurrent forwards there segfault. RSS is sampled across the loop so slow
growth shows up as a number rather than as a crash on stage.

Weights are never downloaded — the child runs with HF_HUB_OFFLINE=1, so a
missing checkpoint fails as "not cached" instead of quietly fetching 149 MB.

Output: a markdown table on stdout, overlay PNGs under --overlays, and a JSON
artifact under backend/data/cache/bench_sam2.json.

    cd backend && .venv/bin/python ../scripts/bench_sam2.py [--images 20]
"""
import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"

SAM2_MODEL = "facebook/sam2.1-hiera-tiny"
FETCH_HINT = ("python -c \"from huggingface_hub import snapshot_download; "
              "snapshot_download('{model}')\"")
DEFAULT_QUERIES = "a person. an animal. a vehicle. an object."

# Caption words that count as "on target" for each detector phrase. Deliberately
# generous and deliberately incomplete: this is a directional proxy for whether
# a masked query retrieves the same KIND of thing, not a recall metric. "an
# object" has no honest word list, so regions labelled that way are excluded
# from the proxy rather than scored against a guess.
ON_TARGET_WORDS = {
    "a person": {"man", "men", "woman", "women", "boy", "boys", "girl", "girls",
                 "person", "people", "child", "children", "kid", "kids", "guy",
                 "lady", "player", "toddler", "baby"},
    "an animal": {"dog", "dogs", "puppy", "puppies", "cat", "cats", "horse",
                  "horses", "bird", "birds", "animal", "animals", "sheep", "cow"},
    "a vehicle": {"car", "cars", "truck", "trucks", "bike", "bikes", "bicycle",
                  "bicycles", "motorcycle", "bus", "boat", "train", "vehicle"},
}


def _peak_rss_gb() -> float:
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2), 2)


def _rss_gb() -> float:
    """Current (not peak) RSS, for growth across the stability loop."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True).stdout.strip()
        return round(int(out) / 1024 ** 2, 3)          # ps reports KB
    except Exception:                                  # noqa: BLE001
        return 0.0


def _weights_cached(model: str) -> bool:
    from huggingface_hub.constants import HF_HUB_CACHE

    return (Path(HF_HUB_CACHE) /
            f"models--{model.replace('/', '--')}").exists()


def sam2_ready(model: str) -> tuple[bool, str | None]:
    """Cheap probe, same contract as `detect.detect_ready`: no model load."""
    try:
        import torch  # noqa: F401
        import transformers
    except ImportError:
        return False, "torch/transformers not installed — the base requirements provide them"
    if not hasattr(transformers, "Sam2Model"):
        # The disqualifying case: SAM2 would need a dependency bump.
        return False, (f"transformers {transformers.__version__} has no Sam2Model — "
                       "SAM2 would require a dependency change, which this repo forbids")
    if not _weights_cached(model):
        return False, (f"SAM2 weights not downloaded — run: "
                       f"{FETCH_HINT.format(model=model)}")
    return True, None


def _overlay(image, mask, box, point, path: Path) -> None:
    """Mask in sage over the image, prompt box outlined, prompt point dotted."""
    from PIL import Image, ImageDraw

    base = image.convert("RGBA")
    tint = Image.new("RGBA", base.size, (33, 109, 80, 110))     # sage #216d50
    alpha = Image.fromarray((mask * 255).astype("uint8"), mode="L")
    base = Image.composite(Image.alpha_composite(base, tint), base, alpha)
    d = ImageDraw.Draw(base)
    d.rectangle(box, outline=(22, 76, 57, 255), width=3)
    if point is not None:
        x, y = point
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(245, 247, 244, 255),
                  outline=(22, 76, 57, 255), width=2)
    base.convert("RGB").save(path)


def _masked_crop(image, mask, box):
    """The crop with everything outside the mask erased to neutral grey.

    Grey, not black or white: SigLIP is not scale-free and a hard black field
    is itself a strong image feature. This is the most favourable honest
    rendering of "embed only the object".
    """
    from PIL import Image

    crop = image.crop(box)
    m = Image.fromarray((mask * 255).astype("uint8"), mode="L").crop(box)
    flat = Image.new("RGB", crop.size, (128, 128, 128))
    return Image.composite(crop, flat, m)


def child(args) -> int:                                   # noqa: C901
    # Guarantee: this process cannot reach the network for weights.
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(BACKEND))

    import numpy as np
    from PIL import Image

    from app import config, db

    out: dict = {"model": args.model}

    ready, reason = sam2_ready(args.model)
    if not ready:
        out["unavailable"] = reason
        out["fetch_hint"] = FETCH_HINT.format(model=args.model)
        print(json.dumps(out))
        return 0

    import torch
    import transformers
    from transformers import Sam2Model, Sam2Processor

    out["transformers"] = transformers.__version__
    out["torch"] = torch.__version__

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

    from app.ml.embedder import _pick_device
    device = _pick_device()

    t0 = time.perf_counter()
    processor = Sam2Processor.from_pretrained(args.model)
    # `output_loading_info` is the honesty check: facebook/sam2.1-hiera-tiny
    # ships a `sam2_video` config, so transformers warns about the class
    # mismatch. Silently randomly-initialised weights would look like a
    # working model and segment garbage, so the missing-key set is reported.
    model, info = Sam2Model.from_pretrained(
        args.model, output_loading_info=True)
    model = model.to(device).eval()
    out["cold_load_s"] = round(time.perf_counter() - t0, 2)
    out["device"] = device
    out["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    out["missing_keys"] = len(info.get("missing_keys") or ())
    out["unexpected_keys"] = len(info.get("unexpected_keys") or ())
    out["mismatched_keys"] = len(info.get("mismatched_keys") or ())

    images = [Image.open(p).convert("RGB") for p in paths]

    def segment(image, box=None, point=None):
        """One mask, the way a request path would ask for it."""
        kw = {}
        if box is not None:
            kw["input_boxes"] = [[list(box)]]
        if point is not None:
            kw["input_points"] = [[[list(point)]]]
            kw["input_labels"] = [[[1]]]
        inputs = processor(images=image, return_tensors="pt", **kw).to(device)
        with torch.no_grad():
            res = model(**inputs, multimask_output=False)
        masks = processor.post_process_masks(
            res.pred_masks.cpu(), inputs["original_sizes"])[0]
        return (masks[0, 0].numpy().astype(bool),
                float(res.iou_scores.detach().float().cpu().reshape(-1)[0]))

    # Prompts come from the SHIPPING detector where it is available, so this
    # measures the real "detector box -> mask" flow rather than a lucky box.
    if args.isolate:
        # SAM2 alone in the process: the only honest way to attribute RSS and
        # MPS bytes to it rather than to a co-resident detector and embedder.
        detector, det_reason = None, "--isolate"
    else:
        from app.ml import detect as detect_ml
        det_ok, det_reason = detect_ml.detect_ready()
        detector = detect_ml.get_detector() if det_ok else None
    out["prompt_source"] = ("grounding-dino boxes" if detector is not None
                            else f"centred fallback box ({det_reason})")

    def prompt_for(image):
        W, H = image.size
        if detector is not None:
            boxes = detector.detect(image, args.queries)
            if boxes:
                b = boxes[0]
                return (b["x"] * W, b["y"] * H,
                        (b["x"] + b["w"]) * W, (b["y"] + b["h"]) * H), b["label"]
        return (W * 0.2, H * 0.2, W * 0.8, H * 0.8), "(fallback centre box)"

    prompts = [prompt_for(im) for im in images]

    # The first forward compiles the Metal graph; that is a load cost.
    warm = time.perf_counter()
    segment(images[0], box=prompts[0][0])
    out["first_call_s"] = round(time.perf_counter() - warm, 2)

    box_ms, point_ms, ious, frac_of_box, mask_empty = [], [], [], [], 0
    for im, (box, _label) in zip(images[1:], prompts[1:], strict=True):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

        t = time.perf_counter()
        mask, iou = segment(im, box=box)
        box_ms.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        segment(im, point=(cx, cy))
        point_ms.append((time.perf_counter() - t) * 1000)

        ious.append(iou)
        area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
        frac_of_box.append(float(mask.sum()) / area)
        if not mask.any():
            mask_empty += 1

    def stats(v, key):
        return {f"{key}_p50": round(statistics.median(v), 1),
                f"{key}_p95": round(statistics.quantiles(v, n=20)[18]
                                    if len(v) >= 20 else max(v), 1),
                f"{key}_min": round(min(v), 1), f"{key}_max": round(max(v), 1)}

    out["images"] = len(box_ms)
    out.update(stats(box_ms, "box_ms"))
    out.update(stats(point_ms, "point_ms"))
    out["iou_mean"] = round(statistics.mean(ious), 3)
    out["iou_min"] = round(min(ious), 3)
    out["mask_frac_of_box_mean"] = round(statistics.mean(frac_of_box), 3)
    out["masks_empty"] = mask_empty

    # ---- The product question, in numbers -------------------------------
    # Does a mask change what retrieval returns? Same embedder, same index,
    # same crop geometry as POST /api/search/by-region.
    from app.ml.providers import get_retrieval_bundle

    runtime = None if args.isolate else get_retrieval_bundle()
    index = runtime.image_index if runtime is not None else None
    embedder = runtime.encoder if runtime is not None else None
    if args.isolate:
        out["retrieval_note"] = "--isolate: SAM2 measured alone, ranking effect skipped"
    elif index is None or embedder is None:
        out["retrieval_note"] = ("embeddings/index unavailable — run "
                                 "`python -m app.ingest`; ranking effect NOT measured")
    else:
        from app.api.deps import first_captions

        def on_target(ids, words):
            """Share of a top-k whose first caption mentions the right kind."""
            conn2 = db.connect()
            caps = first_captions(conn2, list(ids))
            conn2.close()
            hits = sum(1 for i in ids
                       if words & set((caps.get(i) or "").lower()
                                      .replace(".", " ").replace(",", " ").split()))
            return hits / max(len(ids), 1)

        cosines, overlaps, rect_ot, mask_ot = [], [], [], []
        for im, (box, label) in zip(images[1:args.retrieval + 1],
                                    prompts[1:args.retrieval + 1], strict=True):
            mask, _iou = segment(im, box=box)
            if not mask.any():
                continue
            ibox = (int(box[0]), int(box[1]),
                    max(int(box[2]), int(box[0]) + 8),
                    max(int(box[3]), int(box[1]) + 8))
            rect = im.crop(ibox)
            masked = _masked_crop(im, mask, ibox)
            vr, vm = embedder.encode_images([rect, masked])
            vr = vr / (np.linalg.norm(vr) + 1e-12)
            vm = vm / (np.linalg.norm(vm) + 1e-12)
            cosines.append(float(vr @ vm))
            tr = [i for i, _ in index.search(vr, top_k=10)]
            tm = [i for i, _ in index.search(vm, top_k=10)]
            overlaps.append(len(set(tr) & set(tm)) / 10.0)
            words = ON_TARGET_WORDS.get(label)
            if words:                       # "an object" has no honest word list
                rect_ot.append(on_target(tr, words))
                mask_ot.append(on_target(tm, words))
        if cosines:
            out["retrieval_pairs"] = len(cosines)
            out["rect_vs_masked_cos_mean"] = round(statistics.mean(cosines), 4)
            out["rect_vs_masked_cos_min"] = round(min(cosines), 4)
            out["top10_overlap_mean"] = round(statistics.mean(overlaps), 3)
            out["top10_overlap_min"] = round(min(overlaps), 3)
            out["top10_identical_share"] = round(
                sum(1 for o in overlaps if o == 1.0) / len(overlaps), 3)
        if rect_ot:
            out["on_target_pairs"] = len(rect_ot)
            out["on_target_rect"] = round(statistics.mean(rect_ot), 3)
            out["on_target_masked"] = round(statistics.mean(mask_ot), 3)
            out["on_target_delta"] = round(
                statistics.mean(mask_ot) - statistics.mean(rect_ot), 3)
            out["on_target_masked_wins"] = sum(
                1 for m, rc in zip(mask_ot, rect_ot, strict=True) if m > rc)
            out["on_target_rect_wins"] = sum(
                1 for m, rc in zip(mask_ot, rect_ot, strict=True) if rc > m)

    # ---- Stability -------------------------------------------------------
    # Interleaved box/point calls with SigLIP encodes between them: both
    # modules on one Metal device is exactly the demo-time condition.
    rss_track = [_rss_gb()]
    crashes = None
    try:
        for i in range(args.stability):
            im = images[1 + (i % max(len(images) - 1, 1))]
            box, _l = prompts[1 + (i % max(len(prompts) - 1, 1))]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            segment(im, box=box) if i % 2 == 0 else segment(im, point=(cx, cy))
            if embedder is not None and i % 3 == 0:
                embedder.encode_images([im])
            if i % 5 == 0:
                rss_track.append(_rss_gb())
    except Exception as exc:                                   # noqa: BLE001
        crashes = f"{type(exc).__name__}: {exc}"
    rss_track.append(_rss_gb())
    out["stability_calls"] = args.stability
    out["stability_error"] = crashes
    out["rss_start_gb"], out["rss_end_gb"] = rss_track[0], rss_track[-1]
    out["rss_growth_gb"] = round(rss_track[-1] - rss_track[0], 3)
    out["peak_rss_gb"] = _peak_rss_gb()
    if torch.backends.mps.is_available():
        out["mps_driver_gb"] = round(
            torch.mps.driver_allocated_memory() / 1024 ** 3, 2)

    # ---- Overlays a human can look at ------------------------------------
    saved = []
    odir = Path(args.overlays)
    odir.mkdir(parents=True, exist_ok=True)
    for im, p, (box, label) in list(zip(images, paths, prompts, strict=True))[:args.overlay_count]:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        mask, iou = segment(im, box=box)
        f = odir / f"{p.stem}_box.png"
        _overlay(im, mask, box, None, f)
        pmask, piou = segment(im, point=(cx, cy))
        pf = odir / f"{p.stem}_point.png"
        _overlay(im, pmask, box, (cx, cy), pf)
        inter = float((mask & pmask).sum())
        union = float((mask | pmask).sum()) or 1.0
        saved.append({"file": p.name, "label": label,
                      "box_overlay": str(f), "point_overlay": str(pf),
                      "box_iou_score": round(iou, 3),
                      "point_iou_score": round(piou, 3),
                      "box_vs_point_iou": round(inter / union, 3)})
    out["overlays"] = saved

    print(json.dumps(out))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=20,
                    help="corpus images timed after the warm-up call")
    ap.add_argument("--model", default=SAM2_MODEL,
                    help="SAM2 checkpoint (tiny by default; small is the next size up)")
    ap.add_argument("--queries", default=DEFAULT_QUERIES,
                    help="detector phrases that produce the box prompts")
    ap.add_argument("--stability", type=int, default=30,
                    help="interleaved box/point/embed calls in the stability loop")
    ap.add_argument("--retrieval", type=int, default=12,
                    help="regions compared as rectangle-crop vs masked-crop queries")
    ap.add_argument("--overlays", default="/tmp/sam2_overlays",
                    help="directory for overlay PNGs")
    ap.add_argument("--overlay-count", type=int, default=4)
    ap.add_argument("--isolate", action="store_true",
                    help="load SAM2 alone (no detector, no embedder) so RSS and "
                         "MPS bytes belong to it and nothing else")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return child(args)

    print("— measuring SAM2 in a fresh process …", file=sys.stderr)
    proc = subprocess.run(
        [sys.executable, __file__, "--child",
         "--images", str(args.images), "--model", args.model,
         "--queries", args.queries, "--stability", str(args.stability),
         "--retrieval", str(args.retrieval), "--overlays", args.overlays,
         "--overlay-count", str(args.overlay_count)]
        + (["--isolate"] if args.isolate else []),
        capture_output=True, text=True, cwd=BACKEND)
    if proc.returncode != 0 or not proc.stdout.strip():
        print(proc.stderr.strip()[-1500:], file=sys.stderr)
        print(f"\nthe measuring subprocess failed (exit {proc.returncode}) "
              "— no numbers to report")
        return 1
    r = json.loads(proc.stdout.strip().splitlines()[-1])

    artifact = BACKEND / "data" / "cache" / "bench_sam2.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(r, indent=1))

    print(f"\nmodel: {r['model']}")
    if "unavailable" in r:
        # A missing number is a fact, not a blank — and never a download.
        print(f"\nNOT MEASURED: {r['unavailable']}")
        hint = r.get("fetch_hint")
        if hint and hint not in r["unavailable"]:
            print(f"fetch the weights with:\n  {hint}")
        print(f"\nartifact: {artifact}")
        return 0

    print(f"transformers {r['transformers']} (pinned, no new dependency) · "
          f"torch {r['torch']} · device {r['device']} · {r['params_m']} M params")
    print(f"weights: {r['missing_keys']} missing / {r['unexpected_keys']} unexpected"
          f" / {r['mismatched_keys']} mismatched keys")
    print(f"prompts: {r['prompt_source']}   images timed: {r['images']}\n")

    print("| cold load s | first call s | box ms p50 | p95 | point ms p50 | p95 "
          "| peak RSS GB | MPS GB |")
    print("|---|---|---|---|---|---|---|---|")
    print(f"| {r['cold_load_s']} | {r['first_call_s']} | {r['box_ms_p50']} "
          f"| {r['box_ms_p95']} | {r['point_ms_p50']} | {r['point_ms_p95']} "
          f"| {r['peak_rss_gb']} | {r.get('mps_driver_gb', '—')} |")

    print("\n| self-reported IoU mean | min | mask frac of box | empty masks |")
    print("|---|---|---|---|")
    print(f"| {r['iou_mean']} | {r['iou_min']} | {r['mask_frac_of_box_mean']} "
          f"| {r['masks_empty']} |")

    print(f"\nstability: {r['stability_calls']} interleaved calls · "
          f"error={r['stability_error'] or 'none'} · "
          f"RSS {r['rss_start_gb']} → {r['rss_end_gb']} GB "
          f"(growth {r['rss_growth_gb']} GB)")

    print("\nwhat a mask changes for RETRIEVAL (rectangle crop vs masked crop):")
    if "retrieval_pairs" not in r:
        print(f"  {r.get('retrieval_note', 'not measured')}")
    else:
        print(f"  pairs: {r['retrieval_pairs']}   "
              f"query-vector cosine: mean {r['rect_vs_masked_cos_mean']}, "
              f"min {r['rect_vs_masked_cos_min']}")
        print(f"  top-10 overlap: mean {r['top10_overlap_mean']}, "
              f"min {r['top10_overlap_min']}, "
              f"identical for {r['top10_identical_share'] * 100:.0f}% of regions")
    if "on_target_pairs" in r:
        print(f"\ndoes the change HELP? caption-word proxy over "
              f"{r['on_target_pairs']} regions (coarse — a direction, not accuracy):")
        print(f"  on-target share of top-10: rectangle {r['on_target_rect']} vs "
              f"masked {r['on_target_masked']}  (delta {r['on_target_delta']:+})")
        print(f"  masked better on {r['on_target_masked_wins']} regions, "
              f"rectangle better on {r['on_target_rect_wins']}")

    if r.get("overlays"):
        print("\noverlays:")
        for o in r["overlays"]:
            print(f"  {o['file']}  label={o['label']!r}  "
                  f"box IoU-score {o['box_iou_score']}, point {o['point_iou_score']}, "
                  f"box-vs-point agreement {o['box_vs_point_iou']}")
            print(f"    {o['box_overlay']}\n    {o['point_overlay']}")

    print(f"\nartifact: {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
