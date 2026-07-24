"""Flickr8k adapter (https://huggingface.co/datasets/jxie/flickr8k).

8,091 images, 5 human captions each, with train/validation/test splits.
Caption columns are detected by prefix so minor schema variations don't break
ingestion.
"""
from typing import Iterator, Optional

from .base import RawSample


class Flickr8kAdapter:
    name = "flickr8k"
    hf_repo = "jxie/flickr8k"

    def iter_samples(self, limit: Optional[int] = None) -> Iterator[RawSample]:
        from datasets import load_dataset  # lazy: heavy import

        ds = load_dataset(self.hf_repo)
        count = 0
        for split in ds.keys():
            caption_cols = [c for c in ds[split].column_names if c.lower().startswith("caption")]
            for i, row in enumerate(ds[split]):
                if limit is not None and count >= limit:
                    return
                image = row["image"].convert("RGB")
                captions = [str(row[c]).strip() for c in caption_cols if row.get(c)]
                yield RawSample(
                    filename=f"{split}_{i:06d}.jpg",
                    split=split,
                    image=image,
                    captions=captions,
                )
                count += 1
