from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import yaml


@dataclass
class ModelConfig:
    pretrained_model: str = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
    revision: str | None = None
    variant: str | None = "bf16"
    max_sequence_length: int = 300
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    factory: str = "fm_train.example_dataset:build_dataset"
    factory_args: dict[str, Any] = field(default_factory=dict)
    validation_factory_args: dict[str, Any] = field(default_factory=dict)
    resolution: int = 1024
    batch_size: int = 1
    num_workers: int = 4
    prompt_dropout: float = 0.1


@dataclass
class CacheConfig:
    mode: Literal["offline", "online"] = "offline"
    directory: str = "cache"
    producer_device: str = "cuda:0"
    queue_size: int = 8
    batch_size: int = 8
    online_max_entries: int = 256
    overwrite: bool = False


@dataclass
class TrainingConfig:
    profile: Literal["cpt", "sft"] = "cpt"
    output_dir: str = "outputs/cpt"
    max_steps: int = 10000
    gradient_accumulation_steps: int = 1
    mixed_precision: Literal["bf16"] = "bf16"
    seed: int = 42
    max_grad_norm: float = 1.0
    loss_scale: float = 1024.0
    resume_from: str | None = None


@dataclass
class ObjectiveConfig:
    name: Literal["flow"] = "flow"
    scheduler: Literal["z_image", "flux1"] = "z_image"
    shift: float = 3.0
    num_train_timesteps: int = 1000
    weighting: Literal["none", "sigma_inverse"] = "none"


@dataclass
class OptimizerConfig:
    name: Literal["sgd", "adamw", "adamw8bit"] = "adamw"
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    momentum: float = 0.9


@dataclass
class LRConfig:
    warmup_ratio: float = 0.05
    cooldown_ratio: float = 0.1
    plateau_factor: float = 0.5
    plateau_patience: int = 3
    plateau_threshold: float = 1e-4


@dataclass
class DeepSpeedConfig:
    stage: Literal[0, 2, 3] = 2
    config_file: str = "accelerate/zero2.yaml"


@dataclass
class ValidationConfig:
    enabled: bool = True
    every_steps: int = 500
    batches: int = 8
    seed: int = 1234


@dataclass
class SamplingConfig:
    enabled: bool = True
    every_steps: int = 1000
    prompts: list[str] = field(default_factory=lambda: ["a detailed photograph of a mountain lake"])
    num_inference_steps: int = 30
    guidance_scale: float = 4.5
    seed: int = 1234


@dataclass
class CheckpointConfig:
    enabled: bool = True
    every_steps: int = 1000
    keep_last: int = 5
    milestone_steps: list[int] = field(default_factory=list)


@dataclass
class TrackioConfig:
    enabled: bool = True
    project: str = "sana-fm"
    run_name: str | None = None
    space_id: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr: LRConfig = field(default_factory=LRConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    trackio: TrackioConfig = field(default_factory=TrackioConfig)

    def validate(self) -> None:
        if self.data.resolution != 1024:
            raise ValueError("The Sana v1 backend requires data.resolution=1024")
        if not 0 <= self.data.prompt_dropout <= 1:
            raise ValueError("data.prompt_dropout must be in [0, 1]")
        if self.training.loss_scale < 1:
            raise ValueError("training.loss_scale must be >= 1")
        if self.training.max_steps < 1 or self.training.gradient_accumulation_steps < 1:
            raise ValueError("Training steps and gradient accumulation must be positive")
        if self.lr.warmup_ratio < 0 or self.lr.cooldown_ratio < 0:
            raise ValueError("LR phase ratios cannot be negative")
        if self.lr.warmup_ratio + self.lr.cooldown_ratio >= 1:
            raise ValueError("Warmup and cooldown must leave a plateau phase")
        if self.deepspeed.stage not in (0, 2, 3):
            raise ValueError("deepspeed.stage must be 0, 2, or 3")
        if self.cache.mode == "online" and not self.cache.producer_device.startswith("cuda:"):
            raise ValueError("Online preprocessing requires a dedicated CUDA producer_device")
        if self.cache.batch_size < 1:
            raise ValueError("cache.batch_size must be positive")
        if self.cache.online_max_entries < self.cache.batch_size:
            raise ValueError("cache.online_max_entries must be >= cache.batch_size")
        if self.objective.shift <= 0:
            raise ValueError("objective.shift must be positive")
        if self.optimizer.name not in ("sgd", "adamw", "adamw8bit"):
            raise ValueError(f"Unsupported optimizer: {self.optimizer.name}")
        if self.objective.scheduler not in ("z_image", "flux1"):
            raise ValueError(f"Unsupported flow scheduler preset: {self.objective.scheduler}")
        if self.validation.enabled and not self.data.validation_factory_args:
            raise ValueError("Validation is enabled but data.validation_factory_args is empty")
        expected_stage = f"zero{self.deepspeed.stage}"
        if self.deepspeed.stage and expected_stage not in Path(self.deepspeed.config_file).stem.lower():
            raise ValueError("deepspeed.stage does not match deepspeed.config_file")


def _construct(cls: type, raw: dict[str, Any], path: str = "config"):
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {', '.join(sorted(unknown))}")
    hints = get_type_hints(cls)
    values = {}
    for name, value in raw.items():
        typ = hints[name]
        if is_dataclass(typ):
            if not isinstance(value, dict):
                raise ValueError(f"{path}.{name} must be a mapping")
            value = _construct(typ, value, f"{path}.{name}")
        values[name] = value
    return cls(**values)


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    config = _construct(Config, raw)
    config.validate()
    return config
