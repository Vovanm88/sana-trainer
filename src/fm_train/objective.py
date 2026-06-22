from __future__ import annotations

from dataclasses import dataclass

import torch


def interpolate_flow(clean: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return (1 - sigma) * clean + sigma * noise


def flow_target(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return noise - clean


def loss_weight(sigma: torch.Tensor, scheme: str) -> torch.Tensor:
    if scheme == "none":
        return torch.ones_like(sigma)
    if scheme == "sigma_inverse":
        return (sigma.square() + 1e-4).reciprocal()
    raise ValueError(f"Unknown weighting scheme: {scheme}")


def flow_mse(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    per_element = (prediction.float() - target.float()).square() * weight.float()
    return per_element.reshape(per_element.shape[0], -1).mean(dim=1).mean()


@dataclass
class FlowSchedule:
    sigmas: torch.Tensor
    timesteps: torch.Tensor

    @classmethod
    def create(cls, shift: float = 3.0, num_train_timesteps: int = 1000) -> "FlowSchedule":
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps, use_dynamic_shifting=False, shift=shift
        )
        return cls(scheduler.sigmas[:-1].cpu(), scheduler.timesteps.cpu())

    def sample(
        self, batch_size: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator_device = getattr(generator, "device", torch.device("cpu"))
        indices = torch.randint(
            0, len(self.timesteps)-1, (batch_size,), generator=generator, device=generator_device
        ).cpu()
        timesteps = self.timesteps[indices].to(device=device)
        sigma = self.sigmas[indices].to(device=device, dtype=dtype)
        return sigma, timesteps, indices


def validate_source_prediction_type(config: object) -> None:
    prediction_type = getattr(config, "prediction_type", "flow_prediction")
    use_flow_sigmas = getattr(config, "use_flow_sigmas", True)
    if prediction_type != "flow_prediction" or not use_flow_sigmas:
        raise ValueError(
            "The Sana backend only accepts a source scheduler configured for flow_prediction/use_flow_sigmas"
        )
