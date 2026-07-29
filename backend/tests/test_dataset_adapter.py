"""Dataset-source provenance tests."""
import sys
from types import SimpleNamespace

from app.datasets.flickr8k import Flickr8kAdapter


def test_flickr8k_loads_the_reviewed_hub_commit(monkeypatch):
    called = {}

    def fake_load_dataset(repo, **kwargs):
        called.update(repo=repo, **kwargs)
        return {}

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    adapter = Flickr8kAdapter()
    assert list(adapter.iter_samples()) == []
    assert called == {
        "repo": adapter.hf_repo,
        "revision": adapter.hf_revision,
    }
