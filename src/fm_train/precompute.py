from __future__ import annotations

import time

import torch
from tqdm.auto import tqdm

from .cache import CacheCorruptionError, TensorCache, cache_fingerprint
from .config import Config
from .data import build_dataset, prepare_image, validate_sample
from .producer import BoundedProducer


def run_precompute(
    config: Config,
    online: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
) -> None:
    from diffusers import SanaPipeline

    device = torch.device(config.cache.producer_device)
    dtype = torch.bfloat16
    pipeline = SanaPipeline.from_pretrained(
        config.model.pretrained_model, revision=config.model.revision, variant=config.model.variant,
        transformer=None, torch_dtype=dtype,
    )
    pipeline.vae.requires_grad_(False).to(device, dtype=dtype)
    pipeline.text_encoder.requires_grad_(False).to(device, dtype=dtype)
    pipeline.set_progress_bar_config(disable=True)
    cache = TensorCache(config.cache.directory)
    with torch.inference_mode():
        empty_embed, empty_mask, _, _ = pipeline.encode_prompt(
            "", do_classifier_free_guidance=False, device=device,
            max_sequence_length=config.model.max_sequence_length,
        )
        cache.store_empty({"prompt_embeds": empty_embed.cpu(), "prompt_mask": empty_mask.cpu()})
        if num_shards < 1 or not 0 <= shard_index < num_shards:
            raise ValueError("Precompute shard_index must be in [0, num_shards)")
        datasets = [(config.data.factory_args, True)]
        if config.validation.enabled:
            validation = (config.data.validation_factory_args, False)
            datasets = [validation, *datasets] if online else [*datasets, validation]
        for factory_args, is_training in datasets:
            dataset = build_dataset(config.data.factory, factory_args)

            def prepare(index: int):
                sample = dataset[index]
                validate_sample(sample)
                key = cache_fingerprint(
                    config.model.pretrained_model, config.model.revision, config.data.resolution,
                    sample["id"], sample["caption"],
                )
                if cache.contains(key) and not config.cache.overwrite:
                    try:
                        cache.load(key)
                    except CacheCorruptionError:
                        cache.discard(key)
                    else:
                        return key, sample, None
                image = prepare_image(sample["image"], config.data.resolution)
                if torch.cuda.is_available():
                    image = image.pin_memory()
                return key, sample, image

            indices = range(shard_index, len(dataset), num_shards)
            if online and is_training:
                generator = torch.Generator().manual_seed(config.training.seed)
                indices = torch.randperm(len(dataset), generator=generator).tolist()[shard_index::num_shards]
            prepared = BoundedProducer(indices, prepare, maxsize=config.cache.queue_size)

            def encode_batch(items: list[tuple]) -> None:
                images = torch.stack([item[2] for item in items]).to(
                    device, dtype=dtype, non_blocking=True
                )
                latents = pipeline.vae.encode(images).latent * pipeline.vae.config.scaling_factor
                captions = [item[1]["caption"] for item in items]
                embeds, masks, _, _ = pipeline.encode_prompt(
                    captions, do_classifier_free_guidance=False, device=device,
                    max_sequence_length=config.model.max_sequence_length,
                )
                for item_index, (key, sample, _) in enumerate(items):
                    cache.store(
                        key,
                        {"id": sample["id"], "caption": sample["caption"],
                         "latent": latents[item_index].cpu(),
                         "prompt_embeds": embeds[item_index].cpu(),
                         "prompt_mask": masks[item_index].cpu()},
                        overwrite=config.cache.overwrite,
                    )

            pending = []
            cached = 0
            label = "train" if is_training else "validation"
            with tqdm(
                total=len(indices), desc=f"Cache {label}", unit="sample",
                dynamic_ncols=True, position=1 if online else 0,
            ) as progress:
                for key, sample, image in prepared:
                    if image is None:
                        cached += 1
                        progress.set_postfix(cached=cached)
                        progress.update()
                        continue
                    pending.append((key, sample, image))
                    if len(pending) == config.cache.batch_size:
                        encode_batch(pending)
                        progress.update(len(pending))
                        pending.clear()
                        if online:
                            time.sleep(0)
                if pending:
                    encode_batch(pending)
                    progress.update(len(pending))
