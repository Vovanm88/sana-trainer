import torch
import pytest

from fm_train.cache import CacheCorruptionError, TensorCache, cache_fingerprint


def test_cache_roundtrip_and_fingerprint(tmp_path):
    cache = TensorCache(tmp_path)
    first = cache_fingerprint("model", None, 1024, "id", "caption")
    second = cache_fingerprint("model", None, 1024, "id", "other")
    assert first != second
    cache.store(first, {"latent": torch.arange(3), "id": "id"})
    assert torch.equal(cache.load(first)["latent"], torch.arange(3))


def test_cache_reports_truncated_entry(tmp_path):
    cache = TensorCache(tmp_path)
    key = "ab" * 32
    path = cache.path(key)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")

    with pytest.raises(CacheCorruptionError):
        cache.load(key)

    cache.discard(key)
    assert not cache.contains(key)


def test_cache_rejects_unexpected_payload(tmp_path):
    cache = TensorCache(tmp_path)
    key = "cd" * 32
    path = cache.path(key)
    path.parent.mkdir(parents=True)
    torch.save(torch.ones(1), path)

    with pytest.raises(CacheCorruptionError):
        cache.load(key)
