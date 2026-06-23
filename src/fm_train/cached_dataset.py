from __future__ import annotations

import time

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .cache import CacheCorruptionError, TensorCache, cache_fingerprint
from .config import Config
from .data import build_dataset, validate_sample


class CachedTrainingDataset(Dataset):
    def __init__(
        self, config: Config, factory_args: dict, wait_for_online: bool = False,
        discard_after_load: bool = False,
    ):
        self.config = config
        self.dataset = build_dataset(config.data.factory, factory_args)
        self.cache = TensorCache(config.cache.directory)
        self.wait_for_online = wait_for_online
        self.discard_after_load = discard_after_load

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int | tuple[int, int]) -> dict:
        occurrence = None
        if isinstance(index, tuple):
            epoch, index = index
            occurrence = f"epoch:{epoch}"
        if hasattr(self.dataset, "get_metadata"):
            sample = self.dataset.get_metadata(index)
            if not isinstance(sample.get("id"), str) or not isinstance(sample.get("caption"), str):
                raise TypeError("Dataset metadata id and caption must be strings")
        else:
            sample = self.dataset[index]
            validate_sample(sample)
        key = cache_fingerprint(
            self.config.model.pretrained_model, self.config.model.revision, self.config.data.resolution,
            sample["id"], sample["caption"], occurrence,
        )
        if self.wait_for_online:
            deadline = time.monotonic() + 1800
            while True:
                if self.cache.contains(key):
                    try:
                        value = self.cache.load(key)
                    except CacheCorruptionError:
                        pass
                    else:
                        if self.discard_after_load:
                            self.cache.discard(key)
                        return value
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Online producer did not create a valid cache entry for {sample['id']}")
                time.sleep(0.1)
        if not self.cache.contains(key):
            raise FileNotFoundError(f"Missing cache entry for {sample['id']}; run `fm-train precompute` first")
        value = self.cache.load(key)
        if self.discard_after_load:
            self.cache.discard(key)
        return value


def collate_cached(samples: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "ids": [sample["id"] for sample in samples],
        "latents": torch.stack([sample["latent"] for sample in samples]),
        "prompt_embeds": pad_sequence(
            [sample["prompt_embeds"] for sample in samples], batch_first=True
        ),
        "prompt_masks": pad_sequence(
            [sample["prompt_mask"] for sample in samples], batch_first=True
        ),
    }
