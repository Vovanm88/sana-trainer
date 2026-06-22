from __future__ import annotations

from pathlib import Path
import hashlib

import pyarrow.dataset as pads
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset


class ParquetImageDataset(Dataset):
    def __init__(self, parquet: str, image_column: str = "image", caption_column: str = "caption"):
        table = pq.read_table(parquet, columns=[image_column, caption_column])
        self.images = table[image_column].to_pylist()
        self.captions = table[caption_column].to_pylist()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        path = Path(self.images[index])
        return {"id": str(path.resolve()), "image": Image.open(path).convert("RGB"), "caption": self.captions[index]}


def build_dataset(config: dict) -> Dataset:
    return ParquetImageDataset(**config)


class CommonCatalogDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        validation_fraction: float = 0.01,
        split_seed: int = 42,
        include_bad: bool = False,
        include_synthetic: bool = False,
        max_samples: int | None = None,
    ):
        if split not in ("train", "validation"):
            raise ValueError("CommonCatalog split must be train or validation")
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        self.root = Path(root)
        metadata = pads.dataset(self.root / "metadata", format="parquet")
        table = metadata.to_table(
            columns=["rel_path", "blip2_caption", "decode_ok", "is_bad", "is_synthetic"]
        )
        rows: list[tuple[str, str]] = []
        threshold = round(validation_fraction * 1_000_000)
        for row in table.to_pylist():
            caption = row["blip2_caption"]
            if not row["decode_ok"] or not isinstance(caption, str) or not caption.strip():
                continue
            if row["is_bad"] and not include_bad:
                continue
            if row["is_synthetic"] and not include_synthetic:
                continue
            rel_path = row["rel_path"]
            digest = hashlib.sha256(f"{split_seed}:{rel_path}".encode()).digest()
            is_validation = int.from_bytes(digest[:8], "big") % 1_000_000 < threshold
            if (split == "validation") != is_validation:
                continue
            rows.append((rel_path, caption.strip()))
            if max_samples is not None and len(rows) >= max_samples:
                break
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def get_metadata(self, index: int) -> dict:
        rel_path, caption = self.rows[index]
        path = self.root / rel_path
        return {"id": str(path.resolve()), "caption": caption}

    def __getitem__(self, index: int) -> dict:
        sample = self.get_metadata(index)
        sample["image"] = Image.open(sample["id"]).convert("RGB")
        return sample


def build_commoncatalog_dataset(config: dict) -> Dataset:
    return CommonCatalogDataset(**config)
