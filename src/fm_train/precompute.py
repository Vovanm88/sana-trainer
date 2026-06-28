from __future__ import annotations

from collections import deque
import time

import torch
from tqdm.auto import tqdm

from .cache import CacheCorruptionError, TensorCache, cache_fingerprint
from .config import Config
from .data import build_dataset, prepare_image, validate_sample
from .producer import BoundedProducer


def prune_consumed_keys(cache: TensorCache, pending_keys: deque[str]) -> None:
    survivors: deque[str] = deque()
    while pending_keys:
        key = pending_keys.popleft()
        if cache.contains(key):
            survivors.append(key)
    pending_keys.extend(survivors)


def online_indices(
    length: int,
    seed: int,
    epoch: int,
    shard_index: int,
    num_shards: int,
    consumer_batch_size: int = 1,
    consumer_processes: int = 1,
) -> list[int]:
    if consumer_batch_size < 1 or consumer_processes < 1:
        raise ValueError("Online consumer batch size and process count must be positive")
    generator = torch.Generator().manual_seed(seed + epoch)
    consumed = length - (length % (consumer_batch_size * consumer_processes))
    return torch.randperm(length, generator=generator).tolist()[:consumed][shard_index::num_shards]


def run_precompute(
    config: Config,
    online: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    online_consumer_processes: int = 1,
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
            pending_keys: deque[str] = deque()

            def wait_for_capacity(incoming: int) -> None:
                if not online or not is_training:
                    return
                while len(pending_keys) + incoming > config.cache.online_max_entries:
                    prune_consumed_keys(cache, pending_keys)
                    if len(pending_keys) + incoming <= config.cache.online_max_entries:
                        break
                    time.sleep(0.1)

            def prepare(index: int):
                sample = dataset[index]
                validate_sample(sample)
                key = cache_fingerprint(
                    config.model.pretrained_model, config.model.revision, config.data.resolution,
                    sample["id"], sample["caption"],
                    f"epoch:{epoch}" if online and is_training else None,
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
                    prompt_mask = masks[item_index].cpu()
                    prompt_length = max(int(prompt_mask.sum().item()), 1)
                    cache.store(
                        key,
                        {"id": sample["id"], "caption": sample["caption"],
                         "latent": latents[item_index].cpu(),
                         "prompt_embeds": embeds[item_index, :prompt_length].cpu(),
                         "prompt_mask": prompt_mask[:prompt_length]},
                        overwrite=config.cache.overwrite,
                    )

            epoch = 0
            while True:
                if online and is_training:
                    set_epoch = getattr(dataset, "set_epoch", None)
                    if set_epoch is not None:
                        set_epoch(epoch)
                    indices = online_indices(
                        len(dataset), config.training.seed, epoch, shard_index, num_shards,
                        config.data.batch_size, online_consumer_processes,
                    )
                    label = f"Online train producer epoch {epoch + 1}"
                else:
                    indices = range(shard_index, len(dataset), num_shards)
                    split = "train" if is_training else "validation"
                    label = f"Cache {split} (persistent)"
                prepared = BoundedProducer(
                    indices,
                    prepare,
                    maxsize=config.cache.queue_size,
                    workers=config.cache.prepare_workers,
                )
                pending = []
                cached = 0
                with tqdm(
                    total=len(indices), desc=label, unit="sample",
                    dynamic_ncols=True, position=1 if online else 0,
                ) as progress:
                    for key, sample, image in prepared:
                        if image is None:
                            wait_for_capacity(1)
                            if online and is_training:
                                pending_keys.append(key)
                            cached += 1
                            postfix = {"cached": cached}
                            if online and is_training:
                                postfix["window"] = f"{len(pending_keys)}/{config.cache.online_max_entries}"
                            progress.set_postfix(postfix)
                            progress.update()
                            continue
                        pending.append((key, sample, image))
                        if len(pending) == config.cache.batch_size:
                            wait_for_capacity(len(pending))
                            encode_batch(pending)
                            if online and is_training:
                                pending_keys.extend(item[0] for item in pending)
                                progress.set_postfix(
                                    window=f"{len(pending_keys)}/{config.cache.online_max_entries}"
                                )
                            progress.update(len(pending))
                            pending.clear()
                            if online:
                                time.sleep(0)
                    if pending:
                        wait_for_capacity(len(pending))
                        encode_batch(pending)
                        if online and is_training:
                            pending_keys.extend(item[0] for item in pending)
                            progress.set_postfix(
                                window=f"{len(pending_keys)}/{config.cache.online_max_entries}"
                            )
                        progress.update(len(pending))
                if not online or not is_training:
                    break
                epoch += 1
