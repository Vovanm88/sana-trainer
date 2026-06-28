from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .cache import TensorCache
from .cached_dataset import CachedTrainingDataset, collate_cached
from .checkpoints import (
    latest_checkpoint,
    load_export_step,
    rotate_checkpoints,
    save_weight_checkpoint,
)
from .config import Config
from .data import EpochRandomSampler, apply_prompt_dropout
from .objective import (
    FlowSchedule,
    flow_mse_per_sample,
    flow_target,
    interpolate_flow,
    loss_weight,
    validate_source_prediction_type,
)
from .optim import WarmupPlateauCooldown, build_optimizer, unscale_gradients


logger = logging.getLogger(__name__)


def _prediction(model, noisy, timesteps, embeds, masks):
    timestep_scale = getattr(model.config, "timestep_scale", 1.0)
    prediction = model(
        noisy,
        encoder_hidden_states=embeds,
        encoder_attention_mask=masks,
        timestep=timesteps * timestep_scale,
        return_dict=False,
    )[0]
    if prediction.shape[1] == noisy.shape[1] * 2:
        prediction = prediction.chunk(2, dim=1)[0]
    return prediction


def _batch_loss(model, batch, schedule, objective, device, generator=None):
    clean = batch["latents"].to(device, dtype=torch.bfloat16, non_blocking=True)
    embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16, non_blocking=True)
    masks = batch["prompt_masks"].to(device, non_blocking=True)
    noise = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
    sigma, timesteps, indices = schedule.sample(
        clean.shape[0], device, clean.dtype, generator, objective.t_sampling_bias
    )
    while sigma.ndim < clean.ndim:
        sigma = sigma.unsqueeze(-1)
    noisy = interpolate_flow(clean, noise, sigma)
    target = flow_target(clean, noise)
    weight = loss_weight(sigma, objective.weighting)
    per_sample_loss = flow_mse_per_sample(_prediction(model, noisy, timesteps, embeds, masks), target, weight)
    t = timesteps.float() / schedule.timesteps.float().max().clamp_min(1).to(device=device)
    return per_sample_loss.mean(), sigma, indices, per_sample_loss, t


def _t_bucket_label(index: int, bucket_count: int) -> str:
    start = index / bucket_count
    end = (index + 1) / bucket_count
    return f"{index:02d}_{start:.1f}_{end:.1f}".replace(".", "p")


def _bucketed_t_metrics(accelerator, per_sample_loss, t, grad_norm, bucket_count: int = 10) -> dict[str, float]:
    per_sample_loss = accelerator.gather(per_sample_loss.detach().float())
    t = accelerator.gather(t.detach().float()).clamp(0, 1)
    bucket_indices = torch.clamp((t * bucket_count).long(), max=bucket_count - 1)
    counts = torch.bincount(bucket_indices, minlength=bucket_count).float()
    loss_sums = torch.zeros(bucket_count, device=per_sample_loss.device, dtype=torch.float32)
    loss_sums.scatter_add_(0, bucket_indices, per_sample_loss)

    metrics = {}
    for index in range(bucket_count):
        label = _t_bucket_label(index, bucket_count)
        count = counts[index].item()
        metrics[f"train/t_bucket/count/{label}"] = count
        metrics[f"train/t_bucket/loss/{label}"] = (
            (loss_sums[index] / counts[index]).item() if count > 0 else float("nan")
        )

    mean_t = t.mean()
    grad_bucket = min(int((mean_t * bucket_count).item()), bucket_count - 1)
    metrics[f"train/t_bucket/grad_norm/{_t_bucket_label(grad_bucket, bucket_count)}"] = float(grad_norm)
    return metrics


@torch.no_grad()
def validate(accelerator, model, loader, schedule, config: Config) -> float:
    model.eval()
    losses = []
    generator = torch.Generator(device=accelerator.device).manual_seed(config.validation.seed)
    for index, batch in enumerate(loader):
        if index >= config.validation.batches:
            break
        loss, _, _, _, _ = _batch_loss(
            model, batch, schedule, config.objective, accelerator.device, generator
        )
        losses.append(accelerator.gather(loss.reshape(1)).mean())
    model.train()
    if not losses:
        raise ValueError("Validation dataset is empty")
    return torch.stack(losses).mean().item()


@torch.no_grad()
def sample_previews(accelerator, model, config: Config, step: int) -> None:
    """Synchronous state consolidation; all ranks must enter this function."""
    from diffusers import FlowMatchEulerDiscreteScheduler, SanaPipeline, SanaTransformer2DModel

    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        transformer = SanaTransformer2DModel.from_config(unwrapped.config).to(dtype=torch.bfloat16)
        transformer.load_state_dict(state)
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=config.objective.num_train_timesteps,
            use_dynamic_shifting=False,
            shift=config.objective.shift,
        )
        pipeline = SanaPipeline.from_pretrained(
            config.model.pretrained_model, revision=config.model.revision, variant=config.model.variant,
            transformer=transformer, scheduler=scheduler, torch_dtype=torch.bfloat16,
        )
        pipeline.enable_model_cpu_offload(accelerator.device.index or 0)
        images = []
        for index, prompt in enumerate(config.sampling.prompts):
            generator = torch.Generator(device="cpu").manual_seed(config.sampling.seed + index)
            image = pipeline(
                prompt=prompt, height=config.data.resolution, width=config.data.resolution,
                num_inference_steps=config.sampling.num_inference_steps,
                guidance_scale=config.sampling.guidance_scale, generator=generator,
            ).images[0]
            images.append(image)
        try:
            import trackio

            accelerator.log(
                {f"samples/{index}": trackio.Image(image) for index, image in enumerate(images)},
                step=step,
            )
        except (ImportError, AttributeError):
            sample_dir = Path(config.training.output_dir) / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(images):
                image.save(sample_dir / f"step-{step}-{index}.png")
        del pipeline, transformer, state
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()


def run_training(config: Config) -> None:
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
    from diffusers import DPMSolverMultistepScheduler, SanaTransformer2DModel

    project = ProjectConfiguration(project_dir=config.training.output_dir, automatic_checkpoint_naming=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="trackio" if config.trackio.enabled else None,
        project_config=project,
    )
    set_seed(config.training.seed, device_specific=True)
    source_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        config.model.pretrained_model, subfolder="scheduler", revision=config.model.revision
    )
    validate_source_prediction_type(source_scheduler.config)
    resume = config.training.resume_from
    if resume == "latest":
        resume = latest_checkpoint(config.training.output_dir)
    resume = Path(resume) if resume else None
    if resume:
        model = SanaTransformer2DModel.from_pretrained(
            resume / "transformer", torch_dtype=torch.bfloat16
        )
        global_step = load_export_step(resume) if config.training.resume_global_step else 0
    else:
        model = SanaTransformer2DModel.from_pretrained(
            config.model.pretrained_model, subfolder="transformer", revision=config.model.revision,
            variant=config.model.variant, torch_dtype=torch.bfloat16,
        )
        global_step = 0
    model.train().requires_grad_(True)
    if config.model.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    online_cache = config.cache.mode == "online"
    train_dataset = CachedTrainingDataset(
        config, config.data.factory_args, wait_for_online=online_cache,
        discard_after_load=online_cache,
    )
    train_sampler = EpochRandomSampler(train_dataset, config.training.seed, include_epoch=online_cache)
    train_loader = DataLoader(
        train_dataset, batch_size=config.data.batch_size, sampler=train_sampler, num_workers=config.data.num_workers,
        pin_memory=True, collate_fn=collate_cached, drop_last=True,
    )
    validation_loader = None
    if config.validation.enabled:
        validation_dataset = CachedTrainingDataset(
            config, config.data.validation_factory_args, config.cache.mode == "online"
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=config.data.batch_size, shuffle=False,
            num_workers=config.data.num_workers, pin_memory=True, collate_fn=collate_cached,
        )
    optimizer = build_optimizer(model.parameters(), config.optimizer)
    lr_scheduler = WarmupPlateauCooldown(optimizer, config.training.max_steps, config.lr)
    for _ in range(global_step):
        lr_scheduler.step()
    if validation_loader is None:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    else:
        model, optimizer, train_loader, validation_loader = accelerator.prepare(
            model, optimizer, train_loader, validation_loader
        )
    schedule = FlowSchedule.create(config.objective.shift, config.objective.num_train_timesteps)
    empty = TensorCache(config.cache.directory).load_empty()
    dropout_generator = torch.Generator(device="cpu").manual_seed(config.training.seed + accelerator.process_index)
    epoch = 0
    batch_in_epoch = 0

    if config.trackio.enabled:
        trackio_kwargs = {}
        if config.trackio.run_name:
            trackio_kwargs["name"] = config.trackio.run_name
        if config.trackio.space_id:
            trackio_kwargs["space_id"] = config.trackio.space_id
        init_kwargs = {"trackio": trackio_kwargs} if trackio_kwargs else {}
        accelerator.init_trackers(config.trackio.project, config={}, init_kwargs=init_kwargs)
    train_dataset.set_epoch(epoch)
    train_sampler.set_epoch(epoch)
    iterator = iter(train_loader)
    for _ in range(batch_in_epoch):
        try:
            next(iterator)
        except StopIteration:
            epoch += 1
            batch_in_epoch = 0
            train_dataset.set_epoch(epoch)
            train_sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            break
    started = time.perf_counter()
    progress = tqdm(
        total=config.training.max_steps, initial=global_step, desc="Training", unit="step",
        dynamic_ncols=True, position=0, disable=not accelerator.is_local_main_process,
    )
    accumulated_losses = []
    accumulated_t = []
    accumulated_sigma = []
    accumulated_dropped = []
    accumulated_data_seconds = 0.0
    accumulated_compute_seconds = 0.0
    accumulated_microbatches = 0
    optimizer_step_started = time.perf_counter()
    while global_step < config.training.max_steps:
        if accumulated_microbatches == 0:
            optimizer_step_started = time.perf_counter()
        data_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            batch_in_epoch = 0
            train_dataset.set_epoch(epoch)
            train_sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        batch_in_epoch += 1
        data_time = time.perf_counter() - data_started
        accumulated_data_seconds += data_time
        embeds, masks, dropped = apply_prompt_dropout(
            batch["prompt_embeds"], batch["prompt_masks"], empty["prompt_embeds"], empty["prompt_mask"],
            config.data.prompt_dropout, dropout_generator,
        )
        batch["prompt_embeds"], batch["prompt_masks"] = embeds, masks
        step_started = time.perf_counter()
        with accelerator.accumulate(model):
            loss, sigma, _, per_sample_loss, t = _batch_loss(
                model, batch, schedule, config.objective, accelerator.device
            )
            accumulated_losses.append(per_sample_loss.detach())
            accumulated_t.append(t.detach())
            accumulated_sigma.append(sigma.detach().reshape(sigma.shape[0], -1)[:, 0])
            accumulated_dropped.append(dropped.detach())
            accelerator.backward(loss * config.training.loss_scale)
            grad_norm = torch.tensor(0.0, device=accelerator.device)
            if accelerator.sync_gradients:
                unscale_gradients(model.parameters(), config.training.loss_scale)
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        accumulated_compute_seconds += time.perf_counter() - step_started
        accumulated_microbatches += 1
        if not accelerator.sync_gradients:
            continue
        global_step += 1
        lr_scheduler.step()
        elapsed = time.perf_counter() - step_started
        optimizer_step_wall = time.perf_counter() - optimizer_step_started
        step_losses = torch.cat(accumulated_losses)
        step_t = torch.cat(accumulated_t)
        step_sigma = torch.cat(accumulated_sigma)
        step_dropped = torch.cat(accumulated_dropped)
        samples_per_optimizer_step = (
            config.data.batch_size * accumulated_microbatches * accelerator.num_processes
        )
        metrics = {
            "train/loss": accelerator.gather(step_losses.float()).mean().item(),
            "train/lr": lr_scheduler.get_last_lr()[0], "train/grad_norm": float(grad_norm),
            "train/sigma_mean": accelerator.gather(step_sigma.float()).mean().item(),
            "data/dropout_rate": step_dropped.float().mean().item(),
            "time/data_seconds": data_time, "time/step_seconds": elapsed,
            "time/samples_per_second": config.data.batch_size * accelerator.num_processes / max(elapsed, 1e-9),
            "time/data_seconds_sum": accumulated_data_seconds,
            "time/compute_seconds_sum": accumulated_compute_seconds,
            "time/optimizer_step_wall_seconds": optimizer_step_wall,
            "time/effective_samples_per_second": (
                samples_per_optimizer_step / max(optimizer_step_wall, 1e-9)
            ),
            "time/microbatches": accumulated_microbatches,
            "time/data_fraction": accumulated_data_seconds / max(optimizer_step_wall, 1e-9),
            "time/compute_fraction": accumulated_compute_seconds / max(optimizer_step_wall, 1e-9),
        }
        metrics.update(_bucketed_t_metrics(accelerator, step_losses, step_t, grad_norm))
        accumulated_losses.clear()
        accumulated_t.clear()
        accumulated_sigma.clear()
        accumulated_dropped.clear()
        accumulated_data_seconds = 0.0
        accumulated_compute_seconds = 0.0
        accumulated_microbatches = 0
        if torch.cuda.is_available():
            metrics["memory/max_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
        accelerator.log(metrics, step=global_step)
        progress.set_postfix(
            loss=f"{metrics['train/loss']:.4f}", lr=f"{metrics['train/lr']:.2e}",
            data=f"{data_time:.2f}s",
        )
        progress.update()

        if config.validation.enabled and global_step % config.validation.every_steps == 0:
            validation_started = time.perf_counter()
            validation_loss = validate(accelerator, model, validation_loader, schedule, config)
            validation_seconds = time.perf_counter() - validation_started
            lr_scheduler.step_metric(validation_loss)
            accelerator.log(
                {"validation/loss": validation_loss, "time/validation_seconds": validation_seconds},
                step=global_step,
            )
            progress.set_postfix(
                loss=f"{metrics['train/loss']:.4f}", validation=f"{validation_loss:.4f}",
                lr=f"{metrics['train/lr']:.2e}", data=f"{data_time:.2f}s",
            )
        if config.sampling.enabled and global_step % config.sampling.every_steps == 0:
            sampling_started = time.perf_counter()
            sample_previews(accelerator, model, config, global_step)
            accelerator.log(
                {"time/sampling_seconds": time.perf_counter() - sampling_started},
                step=global_step,
            )
        preserved_step = config.checkpointing.enabled and (
            global_step in config.checkpointing.milestone_steps
            or global_step >= config.training.max_steps
        )
        if (
            config.checkpointing.enabled
            and global_step % config.checkpointing.every_steps == 0
            and not preserved_step
        ):
            checkpoint_started = time.perf_counter()
            save_weight_checkpoint(accelerator, model, config, global_step)
            if accelerator.is_main_process:
                rotate_checkpoints(config.training.output_dir, config.checkpointing.keep_last)
            accelerator.log(
                {"time/checkpoint_seconds": time.perf_counter() - checkpoint_started},
                step=global_step,
            )
        if preserved_step:
            checkpoint_started = time.perf_counter()
            save_weight_checkpoint(accelerator, model, config, global_step, milestone=True)
            accelerator.log(
                {"time/checkpoint_seconds": time.perf_counter() - checkpoint_started},
                step=global_step,
            )

    progress.close()
    accelerator.log({"time/total_seconds": time.perf_counter() - started}, step=global_step)
    accelerator.end_training()
