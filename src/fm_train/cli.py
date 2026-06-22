from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .config import load_config


def _launch(config_path: str) -> int:
    config = load_config(config_path)
    environment = os.environ.copy()
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(index) for index in range(_cuda_count())]
    producer = None
    if config.cache.mode == "online":
        producer_index = int(config.cache.producer_device.split(":", 1)[1])
        if producer_index >= len(devices) or len(devices) < 2:
            raise ValueError("Online mode needs at least one producer GPU and one training GPU")
        producer_env = environment.copy()
        producer_env["CUDA_VISIBLE_DEVICES"] = devices[producer_index]
        producer = subprocess.Popen(
            [sys.executable, "-m", "fm_train.cli", "precompute", config_path, "--online"], env=producer_env
        )
        devices = [device for index, device in enumerate(devices) if index != producer_index]
    training_env = environment.copy()
    training_env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    command = [
        "accelerate", "launch", "--config_file", config.deepspeed.config_file,
        "--num_processes", str(len(devices)),
        "--gradient_accumulation_steps", str(config.training.gradient_accumulation_steps),
        "-m", "fm_train.cli", "train", config_path,
    ]
    try:
        return subprocess.call(command, env=training_env)
    finally:
        if producer is not None and producer.poll() is None:
            producer.terminate()
            producer.wait(timeout=30)


def _cuda_count() -> int:
    import torch

    return torch.cuda.device_count()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fm-train")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "launch"):
        child = subparsers.add_parser(command)
        child.add_argument("config")
    precompute = subparsers.add_parser("precompute")
    precompute.add_argument("config")
    precompute.add_argument("--online", action="store_true")
    precompute.add_argument("--shard-index", type=int, default=0)
    precompute.add_argument("--num-shards", type=int, default=1)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    config = load_config(args.config)
    if args.command == "validate-config":
        print(f"Valid configuration: {Path(args.config).resolve()}")
    elif args.command == "precompute":
        from .precompute import run_precompute

        run_precompute(config, args.online, args.shard_index, args.num_shards)
    elif args.command == "train":
        from .trainer import run_training

        run_training(config)
    elif args.command == "launch":
        raise SystemExit(_launch(args.config))


if __name__ == "__main__":
    main()
