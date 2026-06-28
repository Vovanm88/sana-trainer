"""Benchmark real dataset preprocessing with different encoder batch sizes."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from fm_train.config import load_config
from fm_train.precompute import run_precompute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--prepare-workers", type=int, nargs="+", default=[1])
    parser.add_argument("--output-root", default="cache/precompute-benchmark")
    args = parser.parse_args()
    for prepare_workers in args.prepare_workers:
        for batch_size in args.batch_sizes:
            config = load_config(args.config)
            config.validation.enabled = False
            config.data.factory_args["max_samples"] = args.samples
            config.cache.batch_size = batch_size
            config.cache.prepare_workers = prepare_workers
            config.cache.overwrite = True
            config.cache.directory = f"{args.output_root}-w{prepare_workers}-bs{batch_size}"
            target = Path(config.cache.directory).resolve()
            expected = Path(args.output_root + f"-w{prepare_workers}-bs{batch_size}").resolve()
            if target != expected:
                raise RuntimeError(f"Unexpected benchmark cache path: {target}")
            shutil.rmtree(target, ignore_errors=True)
            started = time.perf_counter()
            run_precompute(config)
            elapsed = time.perf_counter() - started
            print(
                f"prepare_workers={prepare_workers} batch_size={batch_size} "
                f"samples={args.samples} seconds={elapsed:.3f} "
                f"images_per_second={args.samples / elapsed:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
