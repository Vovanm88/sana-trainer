from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop, pil_to_tensor, resize


def import_object(path: str) -> Any:
    try:
        module_name, name = path.split(":", 1)
    except ValueError as error:
        raise ValueError(f"Expected import path 'module:object', got {path!r}") from error
    return getattr(importlib.import_module(module_name), name)


def build_dataset(factory_path: str, args: dict[str, Any]) -> Dataset:
    dataset = import_object(factory_path)(args)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Dataset factory {factory_path} did not return torch.utils.data.Dataset")
    return dataset


def prepare_image(image: Image.Image | torch.Tensor, resolution: int = 1024) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image
        if tensor.ndim != 3:
            raise ValueError("Dataset image tensor must have shape [C,H,W]")
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div(255)
    else:
        image = image.convert("RGB")
        width, height = image.size
        scale = resolution / min(width, height)
        image = resize(
            image,
            [round(height * scale), round(width * scale)],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        tensor = pil_to_tensor(image).float().div(255)
    tensor = center_crop(tensor, [resolution, resolution])
    tensor = tensor.mul(2).sub(1).contiguous()
    expected_shape = (3, resolution, resolution)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Prepared image must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    return tensor


def validate_sample(sample: dict[str, Any]) -> None:
    missing = {"id", "image", "caption"} - set(sample)
    if missing:
        raise ValueError(f"Dataset sample is missing: {', '.join(sorted(missing))}")
    if not isinstance(sample["id"], str) or not isinstance(sample["caption"], str):
        raise TypeError("Dataset sample id and caption must be strings")


def apply_prompt_dropout(
    embeds: torch.Tensor,
    masks: torch.Tensor,
    empty_embed: torch.Tensor,
    empty_mask: torch.Tensor,
    probability: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dropped = torch.rand(embeds.shape[0], generator=generator, device="cpu") < probability
    if dropped.any():
        embeds = embeds.clone()
        masks = masks.clone()
        sequence_length = embeds.shape[1]
        replacement_embed = empty_embed[:, :sequence_length].to(embeds)
        replacement_mask = empty_mask[:, :sequence_length].to(masks.device)
        if replacement_embed.shape[1] < sequence_length:
            replacement_embed = torch.nn.functional.pad(
                replacement_embed, (0, 0, 0, sequence_length - replacement_embed.shape[1])
            )
            replacement_mask = torch.nn.functional.pad(
                replacement_mask, (0, sequence_length - replacement_mask.shape[1])
            )
        replacement_embed = replacement_embed.expand(dropped.sum(), -1, -1)
        replacement_mask = replacement_mask.expand(dropped.sum(), -1)
        embeds[dropped.to(embeds.device)] = replacement_embed
        masks[dropped.to(masks.device)] = replacement_mask
    return embeds, masks, dropped


class EpochRandomSampler(Sampler[int]):
    """A permutation sampler whose complete state is just seed + epoch."""

    def __init__(self, data_source: Dataset, seed: int, include_epoch: bool = False):
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0
        self.include_epoch = include_epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        if self.include_epoch:
            yield from ((self.epoch, index) for index in indices)
        else:
            yield from indices

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
