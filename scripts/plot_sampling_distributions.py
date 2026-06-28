from __future__ import annotations

from pathlib import Path

import torch

from fm_train.objective import FlowSchedule


N = 10_000
BINS = 50
DECILES = 10
SEED = 123


def hist(values: torch.Tensor, bins: int) -> list[int]:
    indices = torch.clamp((values.clamp(0, 1) * bins).long(), max=bins - 1)
    return torch.bincount(indices, minlength=bins).tolist()


def deciles(values: torch.Tensor) -> list[int]:
    indices = torch.clamp((values.clamp(0, 1) * DECILES).long(), max=DECILES - 1)
    return torch.bincount(indices, minlength=DECILES).tolist()


def old_uniform_index(
    schedule: FlowSchedule, valid_count: int, t_grid: torch.Tensor, valid_sigmas: torch.Tensor, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, valid_count, (n,), generator=generator)
    return t_grid[indices], valid_sigmas[indices]


def stratified_bucket(
    valid_count: int,
    t_grid: torch.Tensor,
    valid_sigmas: torch.Tensor,
    n: int,
    seed: int,
    bucket_count: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    buckets = torch.clamp((t_grid * bucket_count).long(), max=bucket_count - 1)
    chunks = []
    remaining = n
    while remaining > 0:
        bucket_order = torch.randperm(bucket_count, generator=generator)
        take = min(remaining, bucket_count)
        chunks.append(bucket_order[:take])
        remaining -= take
    target_buckets = torch.cat(chunks)
    indices = torch.empty(n, dtype=torch.long)
    all_indices = torch.arange(valid_count)
    for bucket in range(bucket_count):
        mask = target_buckets == bucket
        count = int(mask.sum().item())
        if count == 0:
            continue
        candidates = all_indices[buckets == bucket]
        offsets = torch.randint(0, candidates.numel(), (count,), generator=generator)
        indices[mask] = candidates[offsets]
    return t_grid[indices], valid_sigmas[indices]


def continuous_uniform(
    valid_count: int, t_grid: torch.Tensor, valid_sigmas: torch.Tensor, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.argsort(t_grid)
    x = t_grid[order]
    y = valid_sigmas[order]
    t = torch.rand(n, generator=generator)
    right = torch.searchsorted(x, t, right=True).clamp(1, valid_count - 1)
    left = right - 1
    span = (x[right] - x[left]).clamp_min(1e-12)
    blend = (t - x[left]) / span
    sigma = y[left] + blend * (y[right] - y[left])
    return t, sigma


def add_histogram(
    svg: list[str],
    x0: int,
    y0: int,
    plot_w: int,
    plot_h: int,
    label: str,
    values: torch.Tensor,
    color: str,
) -> None:
    counts = hist(values, BINS)
    ymax = max(counts) * 1.10
    svg.append(f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#d1d5db"/>')
    svg.append(f'<text x="{x0}" y="{y0 - 10}" font-family="Arial" font-size="13" font-weight="700" fill="{color}">{label}</text>')
    for tick in range(6):
        yy = y0 + plot_h - tick * plot_h / 5
        value = int(ymax * tick / 5)
        svg.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{x0 - 8}" y="{yy + 4:.1f}" text-anchor="end" font-family="Arial" font-size="10" fill="#6b7280">{value}</text>')
    bar_w = plot_w / BINS
    for index, count in enumerate(counts):
        bar_h = 0 if ymax == 0 else count / ymax * plot_h
        bar_x = x0 + index * bar_w
        bar_y = y0 + plot_h - bar_h
        svg.append(
            f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{max(bar_w - 1, 1):.2f}" '
            f'height="{bar_h:.2f}" fill="{color}" opacity="0.82"/>'
        )
    for tick in range(6):
        xx = x0 + tick * plot_w / 5
        svg.append(f'<line x1="{xx:.1f}" y1="{y0 + plot_h}" x2="{xx:.1f}" y2="{y0 + plot_h + 5}" stroke="#6b7280"/>')
        svg.append(f'<text x="{xx:.1f}" y="{y0 + plot_h + 19}" text-anchor="middle" font-family="Arial" font-size="10" fill="#6b7280">{tick / 5:.1f}</text>')
    decile_text = ", ".join(str(value) for value in deciles(values))
    svg.append(f'<text x="{x0}" y="{y0 + plot_h + 39}" font-family="Arial" font-size="10" fill="#374151">deciles: {decile_text}</text>')


def main() -> None:
    output = Path("outputs/t_sigma_sampling_comparison.svg")
    output.parent.mkdir(parents=True, exist_ok=True)

    schedule = FlowSchedule.create(shift=3.0, num_train_timesteps=1000)
    valid_count = min(len(schedule.sigmas), len(schedule.timesteps))
    valid_timesteps = schedule.timesteps[:valid_count].float()
    valid_sigmas = schedule.sigmas[:valid_count].float()
    t_grid = valid_timesteps / valid_timesteps.max().clamp_min(1)

    samples = [
        ("old: uniform schedule index",) + old_uniform_index(schedule, valid_count, t_grid, valid_sigmas, N, SEED),
        ("after: stratified t buckets",) + stratified_bucket(valid_count, t_grid, valid_sigmas, N, SEED),
        ("after: continuous uniform t",) + continuous_uniform(valid_count, t_grid, valid_sigmas, N, SEED),
    ]

    for name, t, sigma in samples:
        print(name)
        print("  t deciles    ", deciles(t))
        print("  sigma deciles", deciles(sigma))

    width, height = 1260, 760
    margin_x, margin_top = 70, 70
    plot_w, plot_h = 340, 235
    col_gap, row_gap = 45, 95
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="70" y="34" font-family="Arial" font-size="24" font-weight="700" fill="#111827">t / sigma sampling distributions, n=10000</text>',
        '<text x="70" y="58" font-family="Arial" font-size="13" fill="#4b5563">Top row: normalized t. Bottom row: sigma.</text>',
    ]
    for col, (name, t, sigma) in enumerate(samples):
        x0 = margin_x + col * (plot_w + col_gap)
        svg.append(f'<text x="{x0}" y="{margin_top - 12}" font-family="Arial" font-size="15" font-weight="700" fill="#111827">{name}</text>')
        add_histogram(svg, x0, margin_top, plot_w, plot_h, "t", t, "#2563eb")
        add_histogram(svg, x0, margin_top + plot_h + row_gap, plot_w, plot_h, "sigma", sigma, "#dc2626")
    svg.append("</svg>")
    output.write_text("\n".join(svg), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
