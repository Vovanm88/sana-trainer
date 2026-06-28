# Sana FM Trainer

Full-parameter CPT/SFT for Sana 1.6B using the Z-Image/FLUX.1 flow Euler schedule.
Sana is already a flow-prediction model; this project adapts its sigma schedule rather
than treating it as an epsilon/DDIM checkpoint.

```bash
uv sync --extra dev
uv run fm-train validate-config configs/cpt.yaml
uv run fm-train precompute configs/cpt.yaml
uv run fm-train launch configs/cpt.yaml
```

To fill one shared offline cache on four GPUs, start one disjoint shard per GPU:

```bash
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu uv run fm-train precompute configs/commoncatalog_sft.yaml \
    --num-shards 4 --shard-index $gpu &
done
wait
```

For online preprocessing set `cache.mode: online`. `launch` reserves
`cache.producer_device`, starts the cache producer there, and exposes the remaining
GPUs to Accelerate. The launcher overrides `num_processes` from the visible training
GPU count.

Dataset factories use `module:callable`; the callable receives `factory_args` and
returns a PyTorch `Dataset`. Every sample must contain `id`, `image`, and `caption`.
The included parquet adapter is only an example and can be replaced without changing
the trainer.

SupUps training splits can be deterministically downsampled by metadata tag:

```yaml
data:
  factory_args:
    tag_keep_probabilities:
      illustration: 0.5
      anime: 0.5
      photo: 1.0
    untagged_keep_probability: 1.0
    tag_filter_seed: 7319
```

For a multi-label sample the lowest configured probability is used. Unknown tag
names and probabilities outside `[0, 1]` are rejected. Leave these fields out of
`validation_factory_args` to validate against the original distribution. Inspect
the effective distribution with:

```bash
uv run python scripts/report_supups_tags.py configs/supups_v3_cpt.yaml \
  --json outputs/supups-v3-cpt/tag-report.json
```

For exact balance between overlapping tags, define ordered, mutually exclusive
primary classes. Each sample is assigned to the first matching class:

```yaml
data:
  factory_args:
    tag_balance_classes:
      anime: [anime]
      art: [illustration, sketch, pixel_art]
      nsfw: [nsfw]
      portrait: [portrait]
      landscape: [landscape]
      photo: [photo]
    tag_balance_target: smallest
    tag_balance_seed: 7319
    tag_balance_keep_unmatched: true
    tag_balance_rotate_each_epoch: true
```

`smallest` uses the smallest primary class size for every class; a positive integer
requests an explicit per-class count. Unmatched samples can be kept or dropped.
With `tag_balance_rotate_each_epoch`, every epoch deterministically selects a new
subset from each full class pool while preserving exactly the same class counts.
This mode is intended for the online cache, whose producer and consumers share the
same epoch number. Use a target below the smallest pool size if the smallest class
must rotate too. Normally `smallest` is preferable: it uses every image from the
bottleneck class while rotating only the larger pools.

Both rolling `checkpoint-*` and preserved `milestone-*` directories contain only the
BF16 `transformer/`, `scheduler/`, and a manifest. They never contain VAE, Gemma,
optimizer moments, or FP32 master weights. `resume_from` can restart from these weights
and step number, but the optimizer, plateau history, RNG, and data order start fresh.

To render every rolling and milestone checkpoint with the config prompts plus two
random `caption_long` dataset prompts, using the same sampling seed everywhere:

```bash
uv run python scripts/sample_all_checkpoints.py configs/supups_v2_cpt.yaml \
  --gpus 0,1,2,3
```

The script starts one persistent process per GPU, distributes checkpoints between
them, shows aggregate `tqdm` progress, and writes images plus `prompts.json` under
`<training.output_dir>/samples-all-checkpoints`. It also creates `comparison.pdf`
with one prompt per page and checkpoints arranged as a labeled grid. Existing images
are resumed; pass `--overwrite` to regenerate them, `--no-pdf` to skip the report, or
`--cpu-offload` if a full pipeline does not fit.
