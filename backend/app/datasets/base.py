"""Dataset adapter interface.

An adapter is responsible for producing a stream of samples; the ingestion
pipeline handles everything else (storage, thumbnails, indexing, embeddings).
"""
from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol, runtime_checkable

from PIL import Image


@dataclass
class RawSample:
    filename: str            # unique, stable identifier used as the image filename
    split: str               # e.g. train / validation / test
    image: Image.Image       # decoded PIL image
    captions: list[str] = field(default_factory=list)


@runtime_checkable
class DatasetAdapter(Protocol):
    name: str

    def iter_samples(self, limit: Optional[int] = None) -> Iterator[RawSample]:
        """Yield samples. `limit` caps the total count (useful for dev runs)."""
        ...
