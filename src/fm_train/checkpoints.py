from __future__ import annotations

import json
import shutil
from pathlib import Path


def checkpoint_step(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[-1])


def rotate_checkpoints(output_dir: str | Path, keep_last: int) -> list[Path]:
    root = Path(output_dir)
    checkpoints = sorted(root.glob("checkpoint-*"), key=checkpoint_step)
    removed = checkpoints[:-keep_last] if keep_last >= 0 else []
    for path in removed:
        shutil.rmtree(path)
    return removed


def save_weight_checkpoint(accelerator, model, config, step: int, milestone: bool = False) -> Path:
    """Export only consolidated DiT weights and flow scheduler configuration."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    prefix = "milestone" if milestone else "checkpoint"
    path = Path(config.training.output_dir) / f"{prefix}-{step}"
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        path.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(
            path / "transformer", state_dict=state_dict, safe_serialization=True
        )
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=config.objective.num_train_timesteps,
            use_dynamic_shifting=False,
            shift=config.objective.shift,
        )
        scheduler.save_pretrained(path / "scheduler")
        (path / "export_manifest.json").write_text(
            json.dumps(
                {
                    "global_step": step,
                    "base_model": config.model.pretrained_model,
                    "variant": config.model.variant,
                    "contents": ["transformer", "scheduler"],
                    "restartable_weights_only": True,
                    "optimizer_state": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    return path


def load_export_step(path: str | Path) -> int:
    manifest = json.loads((Path(path) / "export_manifest.json").read_text(encoding="utf-8"))
    return int(manifest["global_step"])


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    paths = sorted([*root.glob("checkpoint-*"), *root.glob("milestone-*")], key=checkpoint_step)
    return paths[-1] if paths else None
