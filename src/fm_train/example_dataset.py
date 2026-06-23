from __future__ import annotations

from pathlib import Path
import hashlib
import warnings

import pyarrow.dataset as pads
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset


def _open_rgb_image(path: str | Path, draft_size: int | None = None) -> Image.Image:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            if draft_size is not None:
                image.draft("RGB", (draft_size, draft_size))
            return image.convert("RGB")


class ParquetImageDataset(Dataset):
    def __init__(self, parquet: str, image_column: str = "image", caption_column: str = "caption"):
        table = pq.read_table(parquet, columns=[image_column, caption_column])
        self.images = table[image_column].to_pylist()
        self.captions = table[caption_column].to_pylist()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        path = Path(self.images[index])
        return {"id": str(path.resolve()), "image": _open_rgb_image(path), "caption": self.captions[index]}


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
        sample["image"] = _open_rgb_image(sample["id"])
        return sample


def build_commoncatalog_dataset(config: dict) -> Dataset:
    return CommonCatalogDataset(**config)


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


class SupUpsDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        validation_fraction: float = 0.01,
        split_seed: int = 42,
        min_max_score: float = 0.6,
        include_bad_pool: bool = True,
        prefer: str = "long",
        long_caption_probability: float | None = None,
        caption_seed: int = 43,
        decode_size: int = 2048,
        max_samples: int | None = None,
    ):
        if split not in ("train", "validation"):
            raise ValueError("SupUps split must be train or validation")
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if prefer not in ("long", "short"):
            raise ValueError("prefer must be long or short")
        if long_caption_probability is not None and not 0 <= long_caption_probability <= 1:
            raise ValueError("long_caption_probability must be in [0, 1]")
        if not 0 <= min_max_score <= 1:
            raise ValueError("min_max_score must be in [0, 1]")
        if decode_size < 1024:
            raise ValueError("decode_size must be >= 1024")
        self.root = Path(root)
        self.prefer = prefer
        self.long_caption_probability = long_caption_probability
        self.caption_seed = caption_seed
        self.decode_size = decode_size

        metadata = pads.dataset(self.root / "metadata", format="parquet")
        table = metadata.to_table(
            columns=[
                "path", "cap_short", "cap_long", "photo_score", "general_score",
                "is_bad_pool",
            ]
        )
        rows: list[tuple[str, str | None, str | None, float, float, bool]] = []
        threshold = round(validation_fraction * 1_000_000)
        for row in table.to_pylist():
            is_bad = bool(row["is_bad_pool"])
            if is_bad and not include_bad_pool:
                continue
            photo_score = float(row["photo_score"])
            general_score = float(row["general_score"])
            if not is_bad and max(photo_score, general_score) < min_max_score:
                continue
            short = row["cap_short"].strip() if _nonempty_string(row["cap_short"]) else None
            long = row["cap_long"].strip() if _nonempty_string(row["cap_long"]) else None
            if short is None and long is None:
                continue
            path = row["path"]
            if not isinstance(path, str) or not path:
                continue
            digest = hashlib.sha256(f"{split_seed}:{path}".encode()).digest()
            is_validation = int.from_bytes(digest[:8], "big") % 1_000_000 < threshold
            if (split == "validation") != is_validation:
                continue
            rows.append((path, short, long, photo_score, general_score, is_bad))
            if max_samples is not None and len(rows) >= max_samples:
                break
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def get_metadata(self, index: int) -> dict:
        path, short, long, photo, general, is_bad = self.rows[index]
        if self.long_caption_probability is not None and short is not None and long is not None:
            digest = hashlib.sha256(f"{self.caption_seed}:{path}:caption".encode()).digest()
            use_long = int.from_bytes(digest[:8], "big") / 2**64 < self.long_caption_probability
            caption = long if use_long else short
        else:
            caption = (long if self.prefer == "long" else short) or short or long
        return {
            "id": str(Path(path).resolve()),
            "caption": caption,
            "caption_short": short,
            "caption_long": long,
            "photo_score": photo,
            "general_score": general,
            "is_bad_pool": is_bad,
        }

    def __getitem__(self, index: int) -> dict:
        sample = self.get_metadata(index)
        sample["image"] = _open_rgb_image(sample["id"], self.decode_size)
        return sample


def build_supups_dataset(config: dict) -> Dataset:
    return SupUpsDataset(**config)
