"""Optional VLM enrichment via a local Ollama server.

    python -m app.enrich [--model qwen2.5vl:7b] [--limit N]

Tags every image with a local vision-language model (objects, scene,
conditions). Tags become filterable facets and boost keyword search recall
beyond what the 5 human captions mention. Entirely optional: the app is fully
functional without this pass.
"""
import argparse
import base64
import json
import logging
import sys

import httpx
from tqdm import tqdm

from . import config, db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("enrich")

PROMPT = (
    "You are labeling an image for a computer vision dataset catalog. "
    "Return STRICT JSON: {\"tags\": [..]} with 5-10 short lowercase tags covering "
    "main objects, scene type, activity, and notable conditions "
    "(e.g. \"dog\", \"beach\", \"running\", \"low light\", \"motion blur\"). No prose."
)


def check_ollama(client: httpx.Client) -> bool:
    try:
        client.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        return True
    except Exception:
        return False


def tag_image(client: httpx.Client, model: str, image_path) -> list[str]:
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    resp = client.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
            "format": "json",
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    tags = json.loads(content).get("tags", [])
    return [str(t).strip().lower() for t in tags if str(t).strip()][:10]


def main() -> int:
    parser = argparse.ArgumentParser(description="VLM-tag samples via local Ollama.")
    parser.add_argument("--model", default=config.VLM_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    conn = db.connect()
    rows = conn.execute(
        "SELECT id, filename FROM samples WHERE id NOT IN (SELECT DISTINCT sample_id FROM vlm_tags) "
        "ORDER BY id" + (f" LIMIT {int(args.limit)}" if args.limit else "")
    ).fetchall()
    if not rows:
        logger.info("All samples already enriched.")
        return 0

    with httpx.Client() as client:
        if not check_ollama(client):
            logger.error(
                "Ollama is not reachable at %s. Install from https://ollama.com, then: "
                "`ollama pull %s`. The app works fine without enrichment.",
                config.OLLAMA_URL, args.model,
            )
            return 1

        failures = 0
        for row in tqdm(rows, desc=f"Enriching with {args.model}"):
            try:
                # Thumbnails are plenty for tagging and much faster to encode.
                tags = tag_image(client, args.model, config.THUMBS_DIR / row["filename"])
                conn.executemany(
                    "INSERT OR IGNORE INTO vlm_tags(sample_id, tag) VALUES (?,?)",
                    [(row["id"], t) for t in tags],
                )
                conn.commit()
            except Exception as exc:
                failures += 1
                logger.warning("Failed on %s: %s", row["filename"], exc)
                if failures > 20:
                    logger.error("Too many failures; aborting.")
                    return 1
    logger.info("Enrichment complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
