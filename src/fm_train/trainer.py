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
from .objective import FlowSchedule, flow_mse, flow_target, interpolate_flow, loss_weight, validate_source_prediction_type
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


def _batch_loss(model, batch, schedule, weighting, device, generator=None):
    clean = batch["latents"].to(device, dtype=torch.bfloat16, non_blocking=True)
    embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16, non_blocking=True)
    masks = batch["prompt_masks"].to(device, non_blocking=True)
    noise = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
    sigma, timesteps, indices = schedule.sample(clean.shape[0], device, clean.dtype, generator)
    while sigma.ndim < clean.ndim:
        sigma = sigma.unsqueeze(-1)
    noisy = interpolate_flow(clean, noise, sigma)
    target = flow_target(clean, noise)
    weight = loss_weight(sigma, weighting)
    return flow_mse(_prediction(model, noisy, timesteps, embeds, masks), target, weight), sigma, indices


@torch.no_grad()
def validate(accelerator, model, loader, schedule, config: Config) -> float:
    model.eval()
    losses = []
    generator = torch.Generator(device=accelerator.device).manual_seed(config.validation.seed)
    for index, batch in enumerate(loader):
        if index >= config.validation.batches:
            break
        loss, _, _ = _batch_loss(
            model, batch, schedule, config.objective.weighting, accelerator.device, generator
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
        global_step = load_export_step(resume)
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
    train_sampler.set_epoch(epoch)
    iterator = iter(train_loader)
    for _ in range(batch_in_epoch):
        try:
            next(iterator)
        except StopIteration:
            epoch += 1
            batch_in_epoch = 0
            train_sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            break
    started = time.perf_counter()
    progress = tqdm(
        total=config.training.max_steps, initial=global_step, desc="Training", unit="step",
        dynamic_ncols=True, position=0, disable=not accelerator.is_local_main_process,
    )
    while global_step < config.training.max_steps:
        data_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            batch_in_epoch = 0
            train_sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        batch_in_epoch += 1
        data_time = time.perf_counter() - data_started
        embeds, masks, dropped = apply_prompt_dropout(
            batch["prompt_embeds"], batch["prompt_masks"], empty["prompt_embeds"], empty["prompt_mask"],
            config.data.prompt_dropout, dropout_generator,
        )
        batch["prompt_embeds"], batch["prompt_masks"] = embeds, masks
        step_started = time.perf_counter()
        with accelerator.accumulate(model):
            loss, sigma, _ = _batch_loss(model, batch, schedule, config.objective.weighting, accelerator.device)
            accelerator.backward(loss * config.training.loss_scale)
            grad_norm = torch.tensor(0.0, device=accelerator.device)
            if accelerator.sync_gradients:
                unscale_gradients(model.parameters(), config.training.loss_scale)
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if not accelerator.sync_gradients:
            continue
        global_step += 1
        lr_scheduler.step()
        elapsed = time.perf_counter() - step_started
        metrics = {
            "train/loss": accelerator.gather(loss.detach().reshape(1)).mean().item(),
            "train/lr": lr_scheduler.get_last_lr()[0], "train/grad_norm": float(grad_norm),
            "train/sigma_mean": sigma.float().mean().item(), "data/dropout_rate": dropped.float().mean().item(),
            "time/data_seconds": data_time, "time/step_seconds": elapsed,
            "time/samples_per_second": config.data.batch_size * accelerator.num_processes / max(elapsed, 1e-9),
        }
        if torch.cuda.is_available():
            metrics["memory/max_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
        accelerator.log(metrics, step=global_step)
        progress.set_postfix(
            loss=f"{metrics['train/loss']:.4f}", lr=f"{metrics['train/lr']:.2e}",
            data=f"{data_time:.2f}s",
        )
        progress.update()

        if config.validation.enabled and global_step % config.validation.every_steps == 0:
            validation_loss = validate(accelerator, model, validation_loader, schedule, config)
            lr_scheduler.step_metric(validation_loss)
            accelerator.log({"validation/loss": validation_loss}, step=global_step)
            progress.set_postfix(
                loss=f"{metrics['train/loss']:.4f}", validation=f"{validation_loss:.4f}",
                lr=f"{metrics['train/lr']:.2e}", data=f"{data_time:.2f}s",
            )
        if config.sampling.enabled and global_step % config.sampling.every_steps == 0:
            sample_previews(accelerator, model, config, global_step)
        preserved_step = config.checkpointing.enabled and (
            global_step in config.checkpointing.milestone_steps
            or global_step >= config.training.max_steps
        )
        if (
            config.checkpointing.enabled
            and global_step % config.checkpointing.every_steps == 0
            and not preserved_step
        ):
            save_weight_checkpoint(accelerator, model, config, global_step)
            if accelerator.is_main_process:
                rotate_checkpoints(config.training.output_dir, config.checkpointing.keep_last)
        if preserved_step:
            save_weight_checkpoint(accelerator, model, config, global_step, milestone=True)

    progress.close()
    accelerator.log({"time/total_seconds": time.perf_counter() - started}, step=global_step)
    accelerator.end_training()
