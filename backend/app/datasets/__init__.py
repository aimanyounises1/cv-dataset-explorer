"""Dataset adapter registry.

The rest of the app never touches dataset-specific code: any dataset that can
yield (image, split, captions) samples can be plugged in by registering an
adapter here.

The contract is ONE corpus per data directory (`CVDE_DATA_DIR`): every view,
search, statistic, and percentile-ranked axis describes the whole `samples`
table, and no route filters by dataset. Ingesting a second adapter into the
same directory would interleave two corpora in every surface. The `dataset`
column records provenance for exports and manifests — it is not a filter
axis; point a second dataset at its own data directory instead.
"""
from .base import DatasetAdapter
from .base import RawSample as RawSample
from .flickr8k import Flickr8kAdapter

_REGISTRY: dict[str, type[DatasetAdapter]] = {
    Flickr8kAdapter.name: Flickr8kAdapter,
}


def get_adapter(name: str) -> DatasetAdapter:
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(_REGISTRY)}") from exc
