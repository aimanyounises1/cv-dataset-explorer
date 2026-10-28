"""Benchmark exports preserve the evidence instead of only formatted recall."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_export_preserves_complete_retrieval_response(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from scripts.bench_providers import ARTIFACT_VERSION, write_artifact

    # Synthetic payload: this checks preservation, not model quality.
    retrieval = {
        "available": True,
        "sample_size": 4,
        "results": [{"mode": "semantic", "recall_at": {"1": 0.25}, "mrr": 0.5208}],
        "metadata": {"protocol_version": 11, "query_caption_ids": [4, 2, 1, 3]},
    }
    results = [{"provider": "synthetic", "retrieval": retrieval}]
    target = tmp_path / "share" / "benchmark.json"
    write_artifact(target, results, sample_size=50)
    artifact = json.loads(target.read_text())
    assert artifact["artifact_version"] == ARTIFACT_VERSION
    assert artifact["results"][0]["retrieval"] == retrieval
    assert artifact["source_revision"]
    assert isinstance(artifact["source_worktree_dirty"], bool)
    assert artifact["created_at"].endswith("+00:00")
    assert artifact["python_version"]


def test_export_rejects_an_unsupported_sample_size(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from scripts.bench_providers import write_artifact

    with pytest.raises(ValueError, match="sample_size"):
        write_artifact(tmp_path / "benchmark.json", [], sample_size=0)


def test_parent_renders_unavailable_latency_without_hiding_recall(monkeypatch, tmp_path, capsys):
    """Missing latency has its own state and does not become provider failure."""
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from scripts import bench_providers

    result = {
        "provider": "synthetic", "dim": 8, "load_s": 0.0, "index_mb": 0.0,
        "latency_status": "unavailable", "text_ms_p50": None,
        "text_ms_p95": None, "image_ms_p50": None,
        "retrieval": {"available": True, "sample_size": 1},
        "recall": {"semantic": {"r1": 1.0, "r5": 1.0, "r10": 1.0}},
    }
    monkeypatch.setattr(bench_providers.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr=""))
    written = []
    monkeypatch.setattr(bench_providers, "write_artifact",
                        lambda path, results, sample_size: written.extend(results))
    monkeypatch.setattr(sys, "argv", ["bench_providers.py", "--output", str(tmp_path / "result.json")])

    assert bench_providers.main() == 0
    output = capsys.readouterr().out
    assert "| — | — | 100.0%/100.0%/100.0% |" in output
    assert "None" not in output
    assert len(written) == 2
    assert all(row["retrieval"]["sample_size"] == 1 for row in written)
