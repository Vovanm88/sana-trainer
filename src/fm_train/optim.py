from __future__ import annotations

import math
from typing import Iterable

import torch

from .config import LRConfig, OptimizerConfig


def build_optimizer(parameters: Iterable[torch.nn.Parameter], config: OptimizerConfig):
    parameters = list(parameters)
    if config.name == "sgd":
        return torch.optim.SGD(
            parameters, lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay
        )
    kwargs = dict(
        lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    if config.name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    if config.name == "adamw8bit":
        try:
            from bitsandbytes.optim import AdamW8bit
        except ImportError as error:
            raise RuntimeError("optimizer.name=adamw8bit requires bitsandbytes on Linux/CUDA") from error
        return AdamW8bit(parameters, **kwargs)
    raise ValueError(f"Unknown optimizer: {config.name}")


class WarmupPlateauCooldown:
    """Linear warmup, metric-driven middle phase, and guaranteed terminal decay."""

    def __init__(self, optimizer, max_steps: int, config: LRConfig):
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = round(max_steps * config.warmup_ratio)
        self.cooldown_start = max_steps - round(max_steps * config.cooldown_ratio)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.plateau_factor, patience=config.plateau_patience,
            threshold=config.plateau_threshold,
        )

    def step(self) -> None:
        self.step_count += 1
        if self.warmup_steps and self.step_count <= self.warmup_steps:
            factor = self.step_count / self.warmup_steps
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
                group["lr"] = base_lr * factor
        elif self.step_count >= self.cooldown_start:
            denominator = max(1, self.max_steps - self.cooldown_start)
            factor = max(0.0, (self.max_steps - self.step_count) / denominator)
            if self.step_count == self.cooldown_start:
                self.cooldown_lrs = [group["lr"] for group in self.optimizer.param_groups]
            for group, start_lr in zip(self.optimizer.param_groups, self.cooldown_lrs, strict=True):
                group["lr"] = start_lr * factor

    def step_metric(self, metric: float) -> None:
        if self.warmup_steps < self.step_count < self.cooldown_start:
            self.plateau.step(metric)

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"step_count": self.step_count, "plateau": self.plateau.state_dict(),
                "cooldown_lrs": getattr(self, "cooldown_lrs", None)}

    def load_state_dict(self, state: dict) -> None:
        self.step_count = state["step_count"]
        self.plateau.load_state_dict(state["plateau"])
        if state.get("cooldown_lrs") is not None:
            self.cooldown_lrs = state["cooldown_lrs"]


def unscale_gradients(parameters: Iterable[torch.nn.Parameter], scale: float) -> None:
    if scale == 1:
        return
    inverse = 1.0 / scale
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(inverse)
