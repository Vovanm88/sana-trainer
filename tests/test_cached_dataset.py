import threading
import time
from types import SimpleNamespace

import torch

from fm_train.cache import TensorCache, cache_fingerprint
from fm_train.cached_dataset import CachedTrainingDataset


class MetadataDataset:
    def __len__(self):
        return 1

    def get_metadata(self, index):
        return {"id": "sample", "caption": "caption"}


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
