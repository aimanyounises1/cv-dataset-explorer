"""Dataset adapter registry.

The rest of the app never touches dataset-specific code: any dataset that can
yield (image, split, captions) samples can be plugged in by registering an
adapter here.
"""
from .base import DatasetAdapter, RawSample
from .flickr8k import Flickr8kAdapter

_REGISTRY: dict[str, type[DatasetAdapter]] = {
    Flickr8kAdapter.name: Flickr8kAdapter,
}


def get_adapter(name: str) -> DatasetAdapter:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(_REGISTRY)}")


def available_datasets() -> list[str]:
    return list(_REGISTRY)
