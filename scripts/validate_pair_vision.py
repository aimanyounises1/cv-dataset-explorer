#!/usr/bin/env python3
"""Re-run the measured ordered-frame vision contract on frozen local inputs.

This is an acceptance probe, not a benchmark and not a semantic golden test.
It binds the advertised pair capability to two immutable source images, the
configured model digest, the configured Ollama runtime version, the adapter
protocol, the Pydantic schema, and observable output criteria that do not
hard-code the objects in these frames.

Run from the repository root:

    backend/.venv/bin/python scripts/validate_pair_vision.py
    backend/.venv/bin/python scripts/validate_pair_vision.py \
      --write backend/data/reports/pair-vision-validation.json
To qualify a newly pulled artifact or upgraded runtime before changing the
production pins:

    backend/.venv/bin/python scripts/validate_pair_vision.py --model qwen3.5:9b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app import config, db
from app.main import app
from fastapi.testclient import TestClient

FROZEN_PAIR = {
    "a": {
        "sample_id": 76,
        "filename": "train_000075.jpg",
        "sha256": "665ebb7cc4f664097607f079b93e912771a49c21eeabbbec554d09d1cbf5c2ee",
    },
    "b": {
        "sample_id": 2259,
        "filename": "train_002258.jpg",
        "sha256": "82d0916091994f0650d5e1c3e595020e2433028fe0a111a99506fa1c8d895415",
    },
}


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def _frozen_sources() -> dict[str, dict]:
    observed: dict[str, dict] = {}
    with db.get_db() as conn:
        for side, expected in FROZEN_PAIR.items():
            row = conn.execute(
                "SELECT filename, split FROM samples WHERE id = ?",
                (expected["sample_id"],),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    f"Frozen sample {expected['sample_id']} ({side}) is missing."
                )
            if row["filename"] != expected["filename"]:
                raise RuntimeError(
                    f"Frozen sample {expected['sample_id']} now resolves to "
                    f"{row['filename']!r}, expected {expected['filename']!r}."
                )
            image_path = config.IMAGES_DIR / row["filename"]
            image_bytes = image_path.read_bytes()
            digest = hashlib.sha256(image_bytes).hexdigest()
            if digest != expected["sha256"]:
                raise RuntimeError(
                    f"Frozen input {row['filename']} digest changed: {digest}."
                )
            observed[side] = {
                **expected,
                "split": row["split"],
                "byte_length": len(image_bytes),
            }
    return observed


def _observed_candidate(model: str) -> dict[str, str]:
    base = config.OLLAMA_URL.rstrip("/")
    with urllib.request.urlopen(f"{base}/api/tags", timeout=10) as response:
        tags = json.load(response)
    with urllib.request.urlopen(f"{base}/api/version", timeout=10) as response:
        version_body = json.load(response)

    models = tags.get("models") if isinstance(tags, dict) else None
    if not isinstance(models, list):
        raise TypeError("Ollama returned no model list.")
    artifact = next(
        (
            item
            for item in models
            if isinstance(item, dict) and item.get("name") == model
        ),
        None,
    )
    if artifact is None:
        raise RuntimeError(f"Ollama model {model!r} is not installed.")
    digest = artifact.get("digest")
    capabilities = artifact.get("capabilities")
    runtime_version = (
        version_body.get("version") if isinstance(version_body, dict) else None
    )
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"Ollama returned no immutable digest for {model!r}.")
    if not isinstance(capabilities, list) or "vision" not in capabilities:
        raise RuntimeError(f"Ollama model {model!r} does not advertise vision.")
    if not isinstance(runtime_version, str) or not runtime_version.strip():
        raise RuntimeError("Ollama returned no runtime version.")
    return {
        "model": model,
        "model_digest": digest,
        "runtime_version": runtime_version.strip(),
    }


def _acceptance(
    response: dict,
    candidate: dict[str, str],
) -> dict[str, bool]:
    proposal = response["proposal"]
    differences = proposal["differences"]
    concrete_differences = all(
        _normalized(item["image_a"]) not in {"frame a", "image a"}
        and _normalized(item["image_b"]) not in {"frame b", "image b"}
        and _normalized(item["image_a"]) != _normalized(item["image_b"])
        for item in differences
    )
    return {
        "typed_pair_proposal": proposal["kind"] == "pair_comparison",
        "ordered_frozen_sources": (
            response["image_a"]["sample_id"] == FROZEN_PAIR["a"]["sample_id"]
            and response["image_a"]["image_sha256"] == FROZEN_PAIR["a"]["sha256"]
            and response["image_b"]["sample_id"] == FROZEN_PAIR["b"]["sample_id"]
            and response["image_b"]["image_sha256"] == FROZEN_PAIR["b"]["sha256"]
        ),
        "two_distinct_source_digests": (
            response["image_a"]["image_sha256"] != response["image_b"]["image_sha256"]
        ),
        "at_least_one_concrete_difference": bool(differences) and concrete_differences,
        "grounding_terms_proposed_for_both_frames": bool(proposal["grounding_terms_a"])
        and bool(proposal["grounding_terms_b"]),
        "exact_model_artifact": (
            response["model"] == candidate["model"]
            and response["model_digest"] == candidate["model_digest"]
        ),
        "validated_runtime": (
            response["runtime_version"] == candidate["runtime_version"]
        ),
        "ordered_adapter_protocol": (
            response["adapter_id"] == "ollama_sequential_frames"
            and response["protocol"] == "sequential_frames_v1"
        ),
    }


def run(model: str) -> dict:
    sources = _frozen_sources()
    candidate = _observed_candidate(model)
    configured_before = {
        "model": config.VISION_PAIR_MODEL,
        "model_digest": config.VISION_PAIR_MODEL_DIGEST,
        "runtime_version": config.VISION_PAIR_RUNTIME_VERSION,
    }
    original_allowlist = config.VISION_MODELS
    config.VISION_MODELS = tuple(dict.fromkeys((*original_allowlist, model)))
    config.VISION_PAIR_MODEL = candidate["model"]
    config.VISION_PAIR_MODEL_DIGEST = candidate["model_digest"]
    config.VISION_PAIR_RUNTIME_VERSION = candidate["runtime_version"]
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/vision/compare",
                json={
                    "a_sample_id": FROZEN_PAIR["a"]["sample_id"],
                    "b_sample_id": FROZEN_PAIR["b"]["sample_id"],
                },
            )
    finally:
        config.VISION_MODELS = original_allowlist
        config.VISION_PAIR_MODEL = configured_before["model"]
        config.VISION_PAIR_MODEL_DIGEST = configured_before["model_digest"]
        config.VISION_PAIR_RUNTIME_VERSION = configured_before["runtime_version"]
    if response.status_code != 200:
        raise RuntimeError(
            f"Pair endpoint returned {response.status_code}: {response.text}"
        )
    payload = response.json()
    criteria = _acceptance(payload, candidate)
    failed = [name for name, passed in criteria.items() if not passed]
    if failed:
        raise RuntimeError(
            "Pair capability failed acceptance criteria: " + ", ".join(failed)
        )
    return {
        "validation": "ordered_two_frame_semantic_difference_v1",
        "status": "passed",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_sources": sources,
        "configured_before_probe": configured_before,
        "validated_candidate": candidate,
        "suggested_config": {
            "CVDE_VISION_PAIR_MODEL": candidate["model"],
            "CVDE_VISION_PAIR_MODEL_DIGEST": candidate["model_digest"],
            "CVDE_VISION_PAIR_RUNTIME_VERSION": candidate["runtime_version"],
        },
        "provider": payload["provider"],
        "runtime": payload["runtime"],
        "runtime_version": payload["runtime_version"],
        "model": payload["model"],
        "model_digest": payload["model_digest"],
        "adapter_id": payload["adapter_id"],
        "adapter_version": payload["adapter_version"],
        "protocol": payload["protocol"],
        "prompt_version": payload["prompt_version"],
        "schema_version": payload["schema_version"],
        "request_sha256": payload["request_sha256"],
        "proposal_id": payload["proposal_id"],
        "latency_ms": payload["latency_ms"],
        "criteria": criteria,
        "proposal": payload["proposal"],
        "note": payload["note"],
    }


def _evidence_destination(value: Path) -> Path:
    destination = value if value.is_absolute() else ROOT / value
    destination = destination.resolve()
    data_root = config.DATA_DIR.resolve()
    if not destination.is_relative_to(data_root):
        raise ValueError(
            f"--write must stay below the gitignored data directory {data_root}."
        )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=config.VISION_PAIR_MODEL,
        help=(
            "Installed Ollama alias to qualify. The probe discovers its live "
            "digest and runtime before temporarily binding the in-process adapter."
        ),
    )
    parser.add_argument(
        "--write",
        type=Path,
        help=(
            "Write the full JSON evidence record below backend/data/. Generated "
            "runtime evidence is local and must not be committed."
        ),
    )
    args = parser.parse_args()
    try:
        destination = (
            _evidence_destination(args.write) if args.write is not None else None
        )
        record = run(args.model)
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered)
        print(f"PASS: wrote {destination}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
