from __future__ import annotations

from pathlib import Path
import hashlib
import warnings

import pyarrow as pa
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


def _split_tags(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _validate_tag_keep_probabilities(
    probabilities: dict[str, float] | None,
) -> dict[str, float]:
    if probabilities is None:
        return {}
    normalized = {
        str(tag).strip().casefold(): _validate_probability(
            f"tag_keep_probabilities[{tag!r}]", probability
        )
        for tag, probability in probabilities.items()
    }
    if "" in normalized:
        raise ValueError("tag_keep_probabilities keys cannot be empty")
    return normalized


def _validate_tag_balance_classes(
    classes: dict[str, list[str]] | None,
) -> dict[str, set[str]]:
    if classes is None:
        return {}
    normalized = {}
    for class_name, tags in classes.items():
        name = str(class_name).strip()
        if not name:
            raise ValueError("tag_balance_classes names cannot be empty")
        if not isinstance(tags, list) or not tags:
            raise ValueError(f"tag_balance_classes[{name!r}] must be a non-empty list")
        normalized_tags = {str(tag).strip().casefold() for tag in tags}
        if "" in normalized_tags:
            raise ValueError(f"tag_balance_classes[{name!r}] contains an empty tag")
        normalized[name] = normalized_tags
    return normalized


_CAPTION_COLUMNS = {
    "very_short": ("cap_very_short", "very_short", "very short"),
    "short": ("cap_short",),
    "long": ("cap_long",),
}


def _caption_columns(schema_names: set[str]) -> dict[str, str]:
    columns = {}
    for variant, candidates in _CAPTION_COLUMNS.items():
        for column in candidates:
            if column in schema_names:
                columns[variant] = column
                break
    return columns


def _union_fragment_schema(dataset: pads.Dataset) -> pa.Schema:
    fields = {field.name: field for field in dataset.schema}
    for fragment in dataset.get_fragments():
        for field in fragment.physical_schema:
            fields.setdefault(field.name, field)
    return pa.schema(fields.values())


def _validate_caption_weights(caption_weights: dict[str, float] | None) -> dict[str, float] | None:
    if caption_weights is None:
        return None
    unknown = set(caption_weights) - set(_CAPTION_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown caption weight variants: {', '.join(sorted(unknown))}")
    weights = {variant: float(weight) for variant, weight in caption_weights.items()}
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("caption_weights must be non-negative")
    if sum(weights.values()) <= 0:
        raise ValueError("caption_weights must contain at least one positive weight")
    return weights


def _weighted_caption_choice(
    captions: dict[str, str | None],
    weights: dict[str, float],
    seed: int,
    path: str,
) -> str:
    available = [
        (variant, caption, weights.get(variant, 0.0))
        for variant, caption in captions.items()
        if caption is not None and weights.get(variant, 0.0) > 0
    ]
    if not available:
        available = [
            (variant, caption, 1.0)
            for variant, caption in captions.items()
            if caption is not None
        ]
    total = sum(weight for _, _, weight in available)
    digest = hashlib.sha256(f"{seed}:{path}:caption".encode()).digest()
    target = int.from_bytes(digest[:8], "big") / 2**64 * total
    cumulative = 0.0
    for _, caption, weight in available:
        cumulative += weight
        if target < cumulative:
            return caption
    return available[-1][1]


def _hash_unit(seed: int, path: str, salt: str) -> float:
    digest = hashlib.sha256(f"{seed}:{path}:{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


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
        caption_weights: dict[str, float] | None = None,
        caption_seed: int = 43,
        decode_size: int = 2048,
        max_samples: int | None = None,
        tag_keep_probabilities: dict[str, float] | None = None,
        untagged_keep_probability: float = 1.0,
        tag_filter_seed: int = 42,
        tag_balance_classes: dict[str, list[str]] | None = None,
        tag_balance_target: int | str | None = None,
        tag_balance_seed: int = 42,
        tag_balance_keep_unmatched: bool = True,
        tag_balance_rotate_each_epoch: bool = False,
    ):
        if split not in ("train", "validation"):
            raise ValueError("SupUps split must be train or validation")
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if prefer not in ("long", "short"):
            raise ValueError("prefer must be long or short")
        if long_caption_probability is not None and not 0 <= long_caption_probability <= 1:
            raise ValueError("long_caption_probability must be in [0, 1]")
        caption_weights = _validate_caption_weights(caption_weights)
        if not 0 <= min_max_score <= 1:
            raise ValueError("min_max_score must be in [0, 1]")
        if decode_size < 1024:
            raise ValueError("decode_size must be >= 1024")
        self.root = Path(root)
        self.prefer = prefer
        self.long_caption_probability = long_caption_probability
        self.caption_weights = caption_weights
        self.caption_seed = caption_seed
        self.decode_size = decode_size
        self.tag_keep_probabilities = _validate_tag_keep_probabilities(
            tag_keep_probabilities
        )
        self.untagged_keep_probability = _validate_probability(
            "untagged_keep_probability", untagged_keep_probability
        )
        self.tag_filter_seed = int(tag_filter_seed)
        self.tag_balance_classes = _validate_tag_balance_classes(tag_balance_classes)
        if isinstance(tag_balance_target, str) and tag_balance_target != "smallest":
            raise ValueError("tag_balance_target string value must be 'smallest'")
        if isinstance(tag_balance_target, int) and tag_balance_target < 1:
            raise ValueError("tag_balance_target integer must be positive")
        if self.tag_balance_classes and tag_balance_target is None:
            raise ValueError("tag_balance_target is required when tag_balance_classes is set")
        if tag_balance_target is not None and not self.tag_balance_classes:
            raise ValueError("tag_balance_classes is required when tag_balance_target is set")
        self.tag_balance_target = tag_balance_target
        self.tag_balance_seed = int(tag_balance_seed)
        self.tag_balance_keep_unmatched = bool(tag_balance_keep_unmatched)
        self.tag_balance_rotate_each_epoch = bool(tag_balance_rotate_each_epoch)
        self.balance_class_by_path: dict[str, str] = {}
        self._balance_buckets: dict[str, list[tuple]] = {}
        self._balance_unmatched: list[tuple] = []
        self._balance_target_count = 0
        self.epoch = 0

        metadata = pads.dataset(self.root / "metadata", format="parquet")
        schema = _union_fragment_schema(metadata)
        metadata = pads.dataset(self.root / "metadata", format="parquet", schema=schema)
        caption_columns = _caption_columns(set(schema.names))
        if not caption_columns:
            raise ValueError("SupUps metadata must contain at least one supported caption column")
        has_tags = "tags" in schema.names
        table = metadata.to_table(
            columns=[
                "path", *caption_columns.values(), "photo_score", "general_score",
                "is_bad_pool", *(["tags"] if has_tags else []),
            ]
        )
        raw_rows = table.to_pylist()
        observed_tags = {
            tag.casefold()
            for row in raw_rows
            for tag in _split_tags(row.get("tags"))
        }
        unknown_filter_tags = set(self.tag_keep_probabilities) - observed_tags
        if unknown_filter_tags:
            raise ValueError(
                "Unknown tag_keep_probabilities keys: "
                f"{', '.join(sorted(unknown_filter_tags))}; observed tags: "
                f"{', '.join(sorted(observed_tags))}"
            )
        balance_tags = (
            set().union(*self.tag_balance_classes.values())
            if self.tag_balance_classes
            else set()
        )
        unknown_balance_tags = balance_tags - observed_tags
        if unknown_balance_tags:
            raise ValueError(
                "Unknown tag_balance_classes tags: "
                f"{', '.join(sorted(unknown_balance_tags))}; observed tags: "
                f"{', '.join(sorted(observed_tags))}"
            )

        rows: list[tuple[str, dict[str, str | None], float, float, bool, list[str]]] = []
        threshold = round(validation_fraction * 1_000_000)
        for row in raw_rows:
            is_bad = bool(row["is_bad_pool"])
            if is_bad and not include_bad_pool:
                continue
            photo_score = float(row["photo_score"])
            general_score = float(row["general_score"])
            if not is_bad and max(photo_score, general_score) < min_max_score:
                continue
            captions = {
                variant: row[column].strip() if _nonempty_string(row[column]) else None
                for variant, column in caption_columns.items()
            }
            for variant in _CAPTION_COLUMNS:
                captions.setdefault(variant, None)
            if not any(captions.values()):
                continue
            path = row["path"]
            if not isinstance(path, str) or not path:
                continue
            tags = _split_tags(row.get("tags"))
            keep_probability = (
                min(
                    self.tag_keep_probabilities.get(tag.casefold(), 1.0)
                    for tag in set(tags)
                )
                if tags
                else self.untagged_keep_probability
            )
            if (
                keep_probability < 1.0
                and _hash_unit(self.tag_filter_seed, path, "tag-filter")
                >= keep_probability
            ):
                continue
            digest = hashlib.sha256(f"{split_seed}:{path}".encode()).digest()
            is_validation = int.from_bytes(digest[:8], "big") % 1_000_000 < threshold
            if (split == "validation") != is_validation:
                continue
            rows.append((path, captions, photo_score, general_score, is_bad, tags))
            if max_samples is not None and len(rows) >= max_samples:
                break
        if self.tag_balance_classes:
            self._balance_buckets = {
                class_name: [] for class_name in self.tag_balance_classes
            }
            for row in rows:
                row_tags = {tag.casefold() for tag in row[5]}
                class_name = next(
                    (
                        name
                        for name, class_tags in self.tag_balance_classes.items()
                        if row_tags & class_tags
                    ),
                    None,
                )
                if class_name is None:
                    self._balance_unmatched.append(row)
                else:
                    self._balance_buckets[class_name].append(row)
            empty = [
                name for name, bucket in self._balance_buckets.items() if not bucket
            ]
            if empty:
                raise ValueError(f"Tag balance classes are empty: {', '.join(empty)}")
            self._balance_target_count = (
                min(len(bucket) for bucket in self._balance_buckets.values())
                if self.tag_balance_target == "smallest"
                else int(self.tag_balance_target)
            )
            too_small = {
                name: len(bucket)
                for name, bucket in self._balance_buckets.items()
                if len(bucket) < self._balance_target_count
            }
            if too_small:
                details = ", ".join(f"{name}={count}" for name, count in too_small.items())
                raise ValueError(
                    f"tag_balance_target={self._balance_target_count} "
                    f"exceeds available class sizes: {details}"
                )
            self._set_balanced_epoch(0)
        else:
            self.rows = rows

    def _set_balanced_epoch(self, epoch: int) -> None:
        selected: list[tuple] = []
        balance_class_by_path = {}
        selection_epoch = epoch if self.tag_balance_rotate_each_epoch else 0
        for class_name, bucket in self._balance_buckets.items():
            ranked = sorted(
                bucket,
                key=lambda row: _hash_unit(
                    self.tag_balance_seed,
                    row[0],
                    f"tag-balance:{class_name}:epoch:{selection_epoch}",
                ),
            )
            for row in ranked[: self._balance_target_count]:
                selected.append(row)
                balance_class_by_path[row[0]] = class_name
        if self.tag_balance_keep_unmatched:
            selected.extend(self._balance_unmatched)
        # Stable path ordering decouples the subset choice from sampler shuffling.
        self.rows = sorted(selected, key=lambda row: row[0])
        self.balance_class_by_path = balance_class_by_path

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch
        if self.tag_balance_classes:
            self._set_balanced_epoch(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def get_metadata(self, index: int) -> dict:
        path, captions, photo, general, is_bad, tags = self.rows[index]
        short = captions["short"]
        long = captions["long"]
        very_short = captions["very_short"]
        if self.caption_weights is not None:
            caption = _weighted_caption_choice(captions, self.caption_weights, self.caption_seed, path)
        elif self.long_caption_probability is not None and short is not None and long is not None:
            digest = hashlib.sha256(f"{self.caption_seed}:{path}:caption".encode()).digest()
            use_long = int.from_bytes(digest[:8], "big") / 2**64 < self.long_caption_probability
            caption = long if use_long else short
        else:
            caption = (long if self.prefer == "long" else short) or very_short or short or long
        return {
            "id": str(Path(path).resolve()),
            "caption": caption,
            "caption_very_short": very_short,
            "caption_short": short,
            "caption_long": long,
            "tags": tags,
            "photo_score": photo,
            "general_score": general,
            "is_bad_pool": is_bad,
            "balance_class": self.balance_class_by_path.get(path),
        }

    def __getitem__(self, index: int) -> dict:
        sample = self.get_metadata(index)
        sample["image"] = _open_rgb_image(sample["id"], self.decode_size)
        return sample


def build_supups_dataset(config: dict) -> Dataset:
    return SupUpsDataset(**config)


class SupUpsDataset3Tier(SupUpsDataset):
    def __init__(
        self,
        *args,
        p_very_short: float = 0.5,
        p_short: float = 0.25,
        p_long: float = 0.175,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        weights = {
            "very_short": float(p_very_short),
            "short": float(p_short),
            "long": float(p_long),
        }
        self.tier_weights = _validate_caption_weights(weights)

    def get_metadata(self, index: int) -> dict:
        sample = super().get_metadata(index)
        captions = {
            "very_short": sample["caption_very_short"],
            "short": sample["caption_short"],
            "long": sample["caption_long"],
        }
        sample["caption"] = _weighted_caption_choice(
            captions, self.tier_weights, self.caption_seed, self.rows[index][0]
        )
        return sample


def build_supups_dataset_3tier(config: dict) -> Dataset:
    return SupUpsDataset3Tier(**config)


class SupUpsDataset4Tier(SupUpsDataset3Tier):
    def __init__(
        self,
        *args,
        p_no_tag: float = 0.5,
        p_tag_suffix: float = 0.25,
        p_tag_prefix: float = 0.25,
        p_tag_upper: float = 0.25,
        p_tag_lower: float = 0.25,
        p_tag_period: float = 0.5,
        tag_seed: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.p_no_tag = _validate_probability("p_no_tag", p_no_tag)
        self.p_tag_suffix = _validate_probability("p_tag_suffix", p_tag_suffix)
        self.p_tag_prefix = _validate_probability("p_tag_prefix", p_tag_prefix)
        if self.p_no_tag + self.p_tag_suffix + self.p_tag_prefix <= 0:
            raise ValueError("At least one tag placement probability must be positive")
        self.p_tag_upper = _validate_probability("p_tag_upper", p_tag_upper)
        self.p_tag_lower = _validate_probability("p_tag_lower", p_tag_lower)
        if self.p_tag_upper + self.p_tag_lower > 1:
            raise ValueError("p_tag_upper + p_tag_lower must be <= 1")
        self.p_tag_period = _validate_probability("p_tag_period", p_tag_period)
        self.tag_seed = self.caption_seed if tag_seed is None else int(tag_seed)

    def _tag_placement(self, path: str) -> str:
        total = self.p_no_tag + self.p_tag_suffix + self.p_tag_prefix
        target = _hash_unit(self.tag_seed, path, "tag-placement") * total
        if target < self.p_no_tag:
            return "none"
        if target < self.p_no_tag + self.p_tag_suffix:
            return "suffix"
        return "prefix"

    def _format_tag(self, tag: str, path: str) -> str:
        case_target = _hash_unit(self.tag_seed, path, f"tag-case:{tag}")
        if case_target < self.p_tag_upper:
            tag = tag.upper()
        elif case_target < self.p_tag_upper + self.p_tag_lower:
            tag = tag.lower()
        if _hash_unit(self.tag_seed, path, f"tag-period:{tag}") < self.p_tag_period:
            tag = tag.rstrip(".") + "."
        else:
            tag = tag.rstrip(".")
        return tag

    def get_metadata(self, index: int) -> dict:
        sample = super().get_metadata(index)
        path = self.rows[index][0]
        tags = sample["tags"]
        placement = self._tag_placement(path)
        sample["caption_tag"] = None
        sample["caption_tag_placement"] = placement
        if placement == "none" or not tags:
            return sample

        tag_index = int(_hash_unit(self.tag_seed, path, "tag-choice") * len(tags))
        tag = self._format_tag(tags[min(tag_index, len(tags) - 1)], path)
        sample["caption_tag"] = tag
        if placement == "prefix":
            sample["caption"] = f"{tag} {sample['caption']}"
        else:
            sample["caption"] = f"{sample['caption']} {tag}"
        return sample


def build_supups_dataset_4tier(config: dict) -> Dataset:
    return SupUpsDataset4Tier(**config)
