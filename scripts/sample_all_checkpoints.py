"""Render the same prompts and seed for every exported training checkpoint.

One persistent worker is started per GPU. Checkpoints are distributed between
workers, while all image saving and progress reporting happens independently.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import re
import sys
import textwrap
import traceback
from pathlib import Path
from queue import Empty
from typing import Any

from tqdm.auto import tqdm

from fm_train.checkpoints import checkpoint_step
from fm_train.config import Config, load_config
from fm_train.data import build_dataset


def _checkpoint_paths(root: Path) -> list[Path]:
    paths = [*root.glob("checkpoint-*"), *root.glob("milestone-*")]
    paths = [
        path
        for path in paths
        if path.is_dir()
        and (path / "transformer").is_dir()
        and (path / "scheduler").is_dir()
    ]
    return sorted(paths, key=lambda path: (checkpoint_step(path), path.name))


def _dataset_long_prompts(config: Config, count: int, seed: int) -> list[dict[str, str]]:
    if count <= 0:
        return []
    dataset = build_dataset(config.data.factory, dict(config.data.factory_args))
    get_metadata = getattr(dataset, "get_metadata", None)
    if get_metadata is None:
        raise TypeError(
            f"{type(dataset).__name__} has no get_metadata(); "
            "cannot read long captions without decoding images"
        )
    if len(dataset) == 0:
        raise ValueError("Cannot sample long prompts from an empty dataset")

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    seen_indices: set[int] = set()
    seen_prompts: set[str] = set()
    # Random attempts avoid walking a very large parquet dataset in the common case.
    max_attempts = min(max(count * 100, 1000), max(len(dataset) * 2, 1000))
    def consider(index: int) -> bool:
        if index in seen_indices:
            return False
        seen_indices.add(index)
        sample = get_metadata(index)
        prompt = sample.get("caption_long")
        if not isinstance(prompt, str) or not prompt.strip() or prompt in seen_prompts:
            return False
        prompt = prompt.strip()
        seen_prompts.add(prompt)
        selected.append(
            {"prompt": prompt, "source": "dataset_long", "sample_id": str(sample.get("id", ""))}
        )
        return len(selected) == count

    for _ in range(max_attempts):
        if consider(rng.randrange(len(dataset))):
            return selected
    # Guarantee success when enough valid captions exist, even for a tiny or
    # unusually sparse dataset where random draws happened to miss them.
    for index in range(len(dataset)):
        if consider(index):
            return selected

    raise ValueError(
        f"Found only {len(selected)} distinct non-empty long captions; requested {count}"
    )


def _slug(prompt: str, max_length: int = 64) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", prompt).strip("-").lower()
    return (value[:max_length].rstrip("-") or "prompt")


def _image_path(output_root: Path, checkpoint_name: str, index: int, prompt: str) -> Path:
    return output_root / checkpoint_name / f"{index:03d}-{_slug(prompt)}.png"


def _make_comparison_pdf(
    output_root: Path,
    checkpoints: list[Path],
    prompt_records: list[dict[str, str]],
    pdf_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            return ImageFont.load_default()

    title_font = font(28, bold=True)
    body_font = font(20)
    label_font = font(22, bold=True)
    columns = min(3, max(1, round(len(checkpoints) ** 0.5)))
    rows = (len(checkpoints) + columns - 1) // columns
    image_size = 440
    cell_width = image_size + 32
    cell_height = image_size + 62
    page_width = columns * cell_width + 48
    prompt_lines = 1
    pages: list[Image.Image] = []

    for prompt_index, record in enumerate(prompt_records):
        wrapped = textwrap.wrap(
            f"{prompt_index:03d} [{record['source']}] {record['prompt']}",
            width=max(60, page_width // 13),
        )
        prompt_lines = max(len(wrapped), 1)
        header_height = 36 + prompt_lines * 28
        page = Image.new(
            "RGB",
            (page_width, header_height + rows * cell_height + 24),
            "white",
        )
        draw = ImageDraw.Draw(page)
        for line_index, line in enumerate(wrapped):
            draw.text((24, 18 + line_index * 28), line, fill="black", font=body_font)

        for checkpoint_index, checkpoint in enumerate(checkpoints):
            row, column = divmod(checkpoint_index, columns)
            x = 24 + column * cell_width
            y = header_height + row * cell_height
            path = _image_path(output_root, checkpoint.name, prompt_index, record["prompt"])
            if not path.exists():
                raise FileNotFoundError(f"Cannot build PDF; missing image: {path}")
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
                paste_x = x + (image_size - image.width) // 2
                paste_y = y + 38 + (image_size - image.height) // 2
                page.paste(image, (paste_x, paste_y))
            draw.text((x, y), checkpoint.name, fill="black", font=label_font)
        pages.append(page)

    if not pages:
        raise ValueError("Cannot build a PDF without prompts")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(
        pdf_path,
        "PDF",
        resolution=150.0,
        save_all=True,
        append_images=pages[1:],
        title="Checkpoint sampling comparison",
    )
    for page in pages:
        page.close()


def _render_checkpoint(
    checkpoint: Path,
    output_root: Path,
    prompts: list[str],
    config: Config,
    gpu: int,
    batch_size: int,
    cpu_offload: bool,
    overwrite: bool,
    event_queue: Any,
) -> None:
    checkpoint_output = output_root / checkpoint.name
    checkpoint_output.mkdir(parents=True, exist_ok=True)
    if not overwrite and all(
        _image_path(output_root, checkpoint.name, index, prompt).exists()
        for index, prompt in enumerate(prompts)
    ):
        event_queue.put(("progress", checkpoint.name, len(prompts)))
        return

    import torch
    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        SanaPipeline,
        SanaTransformer2DModel,
    )

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    transformer = SanaTransformer2DModel.from_pretrained(
        checkpoint / "transformer", torch_dtype=dtype
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint / "scheduler")
    pipeline = SanaPipeline.from_pretrained(
        config.model.pretrained_model,
        revision=config.model.revision,
        variant=config.model.variant,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=dtype,
    )
    pipeline.set_progress_bar_config(disable=True)
    if cpu_offload:
        pipeline.enable_model_cpu_offload(gpu)
    else:
        pipeline.to(device)

    for start in range(0, len(prompts), batch_size):
        indices = list(range(start, min(start + batch_size, len(prompts))))
        pending = [
            index
            for index in indices
            if overwrite or not _image_path(output_root, checkpoint.name, index, prompts[index]).exists()
        ]
        skipped = len(indices) - len(pending)
        if skipped:
            event_queue.put(("progress", checkpoint.name, skipped))
        if not pending:
            continue

        batch_prompts = [prompts[index] for index in pending]
        # A fresh generator per prompt makes both the seed and initial noise
        # independent of checkpoint assignment, batching, and worker GPU.
        generators = [
            torch.Generator(device="cpu").manual_seed(config.sampling.seed)
            for _ in pending
        ]
        result = pipeline(
            prompt=batch_prompts,
            height=config.data.resolution,
            width=config.data.resolution,
            num_inference_steps=config.sampling.num_inference_steps,
            guidance_scale=config.sampling.guidance_scale,
            generator=generators,
        )
        for index, image in zip(pending, result.images, strict=True):
            image.save(_image_path(output_root, checkpoint.name, index, prompts[index]))
        event_queue.put(("progress", checkpoint.name, len(pending)))

    del pipeline, transformer, scheduler
    torch.cuda.empty_cache()


def _worker(
    gpu: int,
    task_queue: Any,
    event_queue: Any,
    config_path: str,
    output_root: str,
    prompts: list[str],
    batch_size: int,
    cpu_offload: bool,
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    try:
        while True:
            checkpoint_value = task_queue.get()
            if checkpoint_value is None:
                break
            checkpoint = Path(checkpoint_value)
            error = None
            try:
                _render_checkpoint(
                    checkpoint,
                    Path(output_root),
                    prompts,
                    config,
                    gpu,
                    batch_size,
                    cpu_offload,
                    overwrite,
                    event_queue,
                )
            except Exception:
                error = traceback.format_exc()
            event_queue.put(("checkpoint_done", checkpoint.name, error))
    except Exception:
        event_queue.put(("worker_error", str(gpu), traceback.format_exc()))
    finally:
        event_queue.put(("worker_done", str(gpu), None))


def _parse_gpus(value: str | None) -> list[int]:
    if value:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        import torch

        devices = list(range(min(torch.cuda.device_count(), 4)))
    if not devices:
        raise ValueError("No CUDA GPUs selected")
    if len(devices) != len(set(devices)):
        raise ValueError("GPU list contains duplicates")
    return devices


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample every checkpoint with identical prompts and seed on multiple GPUs."
    )
    parser.add_argument("config", help="Training YAML config")
    parser.add_argument(
        "--checkpoint-dir",
        help="Directory containing checkpoint-* and milestone-* (default: training.output_dir)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: <checkpoint-dir>/samples-all-checkpoints)",
    )
    parser.add_argument(
        "--dataset-prompts", type=int, default=2, help="Number of random caption_long prompts"
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        help="Seed used only to choose dataset prompts (default: sampling.seed)",
    )
    parser.add_argument(
        "--gpus", help="Comma-separated local CUDA indices (default: up to four visible GPUs)"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Prompts per inference call")
    parser.add_argument(
        "--cpu-offload", action="store_true", help="Use Diffusers model CPU offload in each worker"
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing PNG files")
    parser.add_argument(
        "--pdf",
        help="Comparison PDF path (default: <output-dir>/comparison.pdf)",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Do not create a comparison PDF")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.dataset_prompts < 0:
        raise ValueError("--dataset-prompts cannot be negative")

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    checkpoint_root = Path(args.checkpoint_dir or config.training.output_dir)
    output_root = Path(args.output_dir or checkpoint_root / "samples-all-checkpoints")
    checkpoints = _checkpoint_paths(checkpoint_root)
    if not checkpoints:
        raise FileNotFoundError(f"No complete checkpoints found in {checkpoint_root}")

    prompt_records = [
        {"prompt": prompt, "source": "config", "sample_id": ""}
        for prompt in config.sampling.prompts
    ]
    dataset_seed = config.sampling.seed if args.dataset_seed is None else args.dataset_seed
    prompt_records.extend(_dataset_long_prompts(config, args.dataset_prompts, dataset_seed))
    prompts = [record["prompt"] for record in prompt_records]
    if not prompts:
        raise ValueError("No prompts were selected")

    gpus = _parse_gpus(args.gpus)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prompts.json").write_text(
        json.dumps(
            {
                "config": str(config_path),
                "seed": config.sampling.seed,
                "num_inference_steps": config.sampling.num_inference_steps,
                "guidance_scale": config.sampling.guidance_scale,
                "prompts": prompt_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    context = mp.get_context("spawn")
    task_queue = context.Queue()
    event_queue = context.Queue()
    for checkpoint in checkpoints:
        task_queue.put(str(checkpoint))
    workers = []
    for gpu in gpus:
        task_queue.put(None)
        process = context.Process(
            target=_worker,
            args=(
                gpu,
                task_queue,
                event_queue,
                str(config_path),
                str(output_root),
                prompts,
                args.batch_size,
                args.cpu_offload,
                args.overwrite,
            ),
            name=f"sample-gpu-{gpu}",
        )
        process.start()
        workers.append(process)

    completed = 0
    worker_done = 0
    progress_by_checkpoint = {checkpoint.name: 0 for checkpoint in checkpoints}
    errors: list[str] = []
    total = len(checkpoints) * len(prompts)
    try:
        with tqdm(total=total, desc="Sampling checkpoints", unit="image") as progress:
            while worker_done < len(workers):
                try:
                    kind, name, value = event_queue.get(timeout=1)
                except Empty:
                    crashed = [
                        process
                        for process in workers
                        if not process.is_alive() and process.exitcode not in (None, 0)
                    ]
                    if crashed:
                        errors.extend(
                            f"{process.name} exited with code {process.exitcode}" for process in crashed
                        )
                        break
                    continue
                if kind == "progress":
                    progress_by_checkpoint[name] += int(value)
                    progress.update(int(value))
                elif kind == "checkpoint_done":
                    completed += 1
                    if value:
                        missing = len(prompts) - progress_by_checkpoint[name]
                        progress.update(max(missing, 0))
                        errors.append(f"{name}:\n{value}")
                    progress.set_postfix(checkpoints=f"{completed}/{len(checkpoints)}")
                elif kind == "worker_error":
                    errors.append(f"GPU {name} worker:\n{value}")
                elif kind == "worker_done":
                    worker_done += 1
    except KeyboardInterrupt:
        for process in workers:
            process.terminate()
        raise
    finally:
        for process in workers:
            process.join()

    if completed != len(checkpoints):
        errors.append(f"Only {completed}/{len(checkpoints)} checkpoints completed")
    if errors:
        print("\n\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"Saved {total} images to {output_root}")
    if not args.no_pdf:
        pdf_path = Path(args.pdf) if args.pdf else output_root / "comparison.pdf"
        _make_comparison_pdf(output_root, checkpoints, prompt_records, pdf_path)
        print(f"Saved comparison PDF to {pdf_path}")


if __name__ == "__main__":
    main()
