import threading
import time
from types import SimpleNamespace

import torch

from fm_train.cache import TensorCache, cache_fingerprint
from fm_train.cached_dataset import CachedTrainingDataset, collate_cached


class MetadataDataset:
    def __len__(self):
        return 1

    def get_metadata(self, index):
        return {"id": "sample", "caption": "caption"}


def test_cache_fingerprint_can_distinguish_online_occurrences():
    first = cache_fingerprint("model", None, 64, "sample", "caption", "epoch:0")
    second = cache_fingerprint("model", None, 64, "sample", "caption", "epoch:1")

    assert first != second


def test_online_dataset_waits_for_valid_cache_entry(monkeypatch, tmp_path):
    config = SimpleNamespace(
        model=SimpleNamespace(pretrained_model="model", revision=None),
        data=SimpleNamespace(factory="unused", resolution=64),
        cache=SimpleNamespace(directory=str(tmp_path)),
    )
    monkeypatch.setattr("fm_train.cached_dataset.build_dataset", lambda *_: MetadataDataset())
    dataset = CachedTrainingDataset(config, {}, wait_for_online=True)
    key = cache_fingerprint("model", None, 64, "sample", "caption")
    path = dataset.cache.path(key)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")

    def replace_entry():
        time.sleep(0.15)
        TensorCache(tmp_path).store(key, {"id": "sample", "latent": torch.ones(1)}, overwrite=True)

    writer = threading.Thread(target=replace_entry)
    writer.start()
    value = dataset[0]
    writer.join()

    assert value["id"] == "sample"


def test_collate_pads_variable_length_conditioning():
    batch = collate_cached([
        {
            "id": "short", "latent": torch.ones(2, 2, 2),
            "prompt_embeds": torch.ones(2, 3), "prompt_mask": torch.ones(2),
        },
        {
            "id": "long", "latent": torch.zeros(2, 2, 2),
            "prompt_embeds": torch.full((4, 3), 2.0), "prompt_mask": torch.ones(4),
        },
    ])

    assert batch["prompt_embeds"].shape == (2, 4, 3)
    assert batch["prompt_masks"].shape == (2, 4)
    assert not batch["prompt_masks"][0, 2:].any()


def test_online_dataset_discards_consumed_entry(monkeypatch, tmp_path):
    config = SimpleNamespace(
        model=SimpleNamespace(pretrained_model="model", revision=None),
        data=SimpleNamespace(factory="unused", resolution=64),
        cache=SimpleNamespace(directory=str(tmp_path)),
    )
    monkeypatch.setattr("fm_train.cached_dataset.build_dataset", lambda *_: MetadataDataset())
    dataset = CachedTrainingDataset(
        config, {}, wait_for_online=True, discard_after_load=True
    )
    key = cache_fingerprint("model", None, 64, "sample", "caption")
    dataset.cache.store(key, {"id": "sample", "latent": torch.ones(1)})

    assert dataset[0]["id"] == "sample"
    assert not dataset.cache.contains(key)


def test_validation_entry_remains_available_for_repeated_metrics(monkeypatch, tmp_path):
    config = SimpleNamespace(
        model=SimpleNamespace(pretrained_model="model", revision=None),
        data=SimpleNamespace(factory="unused", resolution=64),
        cache=SimpleNamespace(directory=str(tmp_path)),
    )
    monkeypatch.setattr("fm_train.cached_dataset.build_dataset", lambda *_: MetadataDataset())
    dataset = CachedTrainingDataset(config, {}, wait_for_online=True)
    key = cache_fingerprint("model", None, 64, "sample", "caption")
    dataset.cache.store(key, {"id": "sample", "latent": torch.ones(1)})

    assert dataset[0]["id"] == "sample"
    assert dataset[0]["id"] == "sample"
    assert dataset.cache.contains(key)


def test_online_dataset_uses_epoch_specific_cache_entry(monkeypatch, tmp_path):
    config = SimpleNamespace(
        model=SimpleNamespace(pretrained_model="model", revision=None),
        data=SimpleNamespace(factory="unused", resolution=64),
        cache=SimpleNamespace(directory=str(tmp_path)),
    )
    monkeypatch.setattr("fm_train.cached_dataset.build_dataset", lambda *_: MetadataDataset())
    dataset = CachedTrainingDataset(
        config, {}, wait_for_online=True, discard_after_load=True
    )
    epoch_zero = cache_fingerprint("model", None, 64, "sample", "caption", "epoch:0")
    epoch_one = cache_fingerprint("model", None, 64, "sample", "caption", "epoch:1")
    dataset.cache.store(epoch_zero, {"id": "epoch-zero", "latent": torch.ones(1)})
    dataset.cache.store(epoch_one, {"id": "epoch-one", "latent": torch.ones(1)})

    assert dataset[(1, 0)]["id"] == "epoch-one"
    assert dataset.cache.contains(epoch_zero)
    assert not dataset.cache.contains(epoch_one)
