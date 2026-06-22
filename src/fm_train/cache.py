from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import torch


CACHE_VERSION = 1


class CacheCorruptionError(RuntimeError):
    pass


def cache_fingerprint(model: str, revision: str | None, resolution: int, sample_id: str, caption: str) -> str:
    value = json.dumps(
        {"v": CACHE_VERSION, "model": model, "revision": revision, "resolution": resolution,
         "id": sample_id, "caption": caption},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class TensorCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.pt"

    def contains(self, key: str) -> bool:
        return self.path(key).is_file()

    def load(self, key: str) -> dict[str, Any]:
        try:
            value = torch.load(self.path(key), map_location="cpu", weights_only=True)
        except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
            raise CacheCorruptionError(f"Invalid cache entry for {key}") from error
        if not isinstance(value, dict):
            raise CacheCorruptionError(f"Invalid cache entry for {key}")
        if value.get("cache_version") != CACHE_VERSION:
            raise CacheCorruptionError(f"Unsupported cache version for {key}")
        return value

    def discard(self, key: str) -> None:
        self.path(key).unlink(missing_ok=True)

    def store(self, key: str, tensors: dict[str, Any], overwrite: bool = False) -> Path:
        target = self.path(key)
        if target.exists() and not overwrite:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        value = {"cache_version": CACHE_VERSION, **tensors}
        torch.save(value, temporary)
        os.replace(temporary, target)
        return target

    def store_empty(self, tensors: dict[str, Any]) -> Path:
        return self.store("empty-conditioning", tensors, overwrite=True)

    def load_empty(self) -> dict[str, Any]:
        return self.load("empty-conditioning")
