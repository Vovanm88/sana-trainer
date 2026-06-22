"""Small CUDA/Accelerate smoke test that does not download model weights."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from accelerate import Accelerator

from fm_train.config import OptimizerConfig
from fm_train.optim import build_optimizer, unscale_gradients


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(256, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 256),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=("adamw", "adamw8bit"), default="adamw")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=2, mixed_precision="bf16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    model = TinyModel()
    optimizer = build_optimizer(model.parameters(), OptimizerConfig(name=args.optimizer, learning_rate=1e-3))
    dataset = torch.utils.data.TensorDataset(torch.randn(args.steps * 16, 256))
    loader = torch.utils.data.DataLoader(dataset, batch_size=8)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    optimizer.zero_grad(set_to_none=True)
    loss = None
    for (cpu_inputs,) in loader:
        with accelerator.accumulate(model):
            inputs = cpu_inputs.to(accelerator.device, dtype=torch.bfloat16)
            loss = model(inputs).float().square().mean()
            accelerator.backward(loss * 1024)
            if accelerator.sync_gradients:
                unscale_gradients(model.parameters(), 1024)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    assert loss is not None
    reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
    state_dir = Path(tempfile.gettempdir()) / f"fmtrain-smoke-{args.optimizer}"
    accelerator.save_state(state_dir)
    accelerator.load_state(state_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(
            f"ok optimizer={args.optimizer} processes={accelerator.num_processes} "
            f"gpu={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()} "
            f"loss={reduced_loss.item():.6f} state={state_dir}"
        )


if __name__ == "__main__":
    main()
