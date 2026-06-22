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

Both rolling `checkpoint-*` and preserved `milestone-*` directories contain only the
BF16 `transformer/`, `scheduler/`, and a manifest. They never contain VAE, Gemma,
optimizer moments, or FP32 master weights. `resume_from` can restart from these weights
and step number, but the optimizer, plateau history, RNG, and data order start fresh.
