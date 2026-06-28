from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from fm_train.config import load_config
from fm_train.objective import FlowSchedule


def main() -> None:
    config_path = Path("configs/supups_v1_sft.yaml")
    output_dir = Path("outputs/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "current_t_sigma_distribution.png"
    svg_path = output_dir / "current_t_sigma_distribution.svg"

    config = load_config(config_path)
    schedule = FlowSchedule.create(
        shift=config.objective.shift,
        num_train_timesteps=config.objective.num_train_timesteps,
    )
    bias = config.objective.t_sampling_bias
    sample_count = 500_000
    generator = torch.Generator(device="cpu").manual_seed(config.training.seed)
    sigma, timesteps, _ = schedule.sample(
        sample_count,
        torch.device("cpu"),
        torch.float32,
        generator=generator,
        t_sampling_bias=bias,
    )

    valid_count = min(len(schedule.sigmas), len(schedule.timesteps))
    valid_timesteps = schedule.timesteps[:valid_count].float()
    max_timestep = valid_timesteps.max().clamp_min(1)
    discrete_t = (valid_timesteps / max_timestep).numpy()
    discrete_sigma = schedule.sigmas[:valid_count].float().numpy()
    order = np.argsort(discrete_t)
    discrete_t = discrete_t[order]
    discrete_sigma = discrete_sigma[order]

    t = (timesteps.float() / max_timestep).numpy()
    sigma_np = sigma.float().numpy()
    t_grid = np.linspace(1e-5, 1.0, 2000)
    t_pdf = (1.0 / bias) * np.power(t_grid, 1.0 / bias - 1.0)
    sigma_quantiles = np.quantile(sigma_np, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    t_quantiles = np.quantile(t, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(16, 11), dpi=150)
    gs = fig.add_gridspec(2, 2)
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_sigma = fig.add_subplot(gs[1, 0])
    ax_joint = fig.add_subplot(gs[1, 1])

    ax_curve.plot(discrete_t, discrete_sigma, color="#1f77b4", linewidth=2.2, label="z-image scheduler sigma(t)")
    ax_curve.scatter(discrete_t[::20], discrete_sigma[::20], color="#1f77b4", s=10, alpha=0.35, label="scheduler grid")
    for q in t_quantiles[[2, 4, 6]]:
        ax_curve.axvline(q, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_curve.set_title("Original z-image sigma curve")
    ax_curve.set_xlabel("normalized timestep t")
    ax_curve.set_ylabel("sigma")
    ax_curve.set_xlim(0, 1)
    ax_curve.set_ylim(0, 1)
    ax_curve.legend(loc="upper left", fontsize=10)

    sns.histplot(t, bins=120, stat="density", ax=ax_t, color="#2ca02c", alpha=0.55, edgecolor=None)
    ax_t.plot(t_grid, t_pdf, color="#111111", linewidth=2, label=f"analytic pdf, rand() ** {bias:g}")
    ax_t.set_title(f"Actual sampled t distribution, n={sample_count:,}")
    ax_t.set_xlabel("normalized timestep t")
    ax_t.set_ylabel("density")
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, np.quantile(t_pdf, 0.985) * 1.15)
    ax_t.legend(loc="upper right", fontsize=10)

    sns.histplot(sigma_np, bins=120, stat="density", ax=ax_sigma, color="#ff7f0e", alpha=0.62, edgecolor=None)
    for q in sigma_quantiles[[2, 4, 6]]:
        ax_sigma.axvline(q, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.75)
    ax_sigma.set_title("Actual sampled sigma distribution")
    ax_sigma.set_xlabel("sigma")
    ax_sigma.set_ylabel("density")
    ax_sigma.set_xlim(0, 1)

    ax_joint.plot(discrete_t, discrete_sigma, color="#1f77b4", linewidth=1.6, alpha=0.9, label="z-image sigma(t)")
    hb = ax_joint.hexbin(t, sigma_np, gridsize=80, cmap="magma", bins="log", mincnt=1, linewidths=0)
    ax_joint.plot(discrete_t, discrete_sigma, color="#7dd3fc", linewidth=1.0, alpha=0.85)
    ax_joint.set_title("Sample density over original sigma curve")
    ax_joint.set_xlabel("sampled t")
    ax_joint.set_ylabel("sampled sigma")
    ax_joint.set_xlim(0, 1)
    ax_joint.set_ylim(0, 1)
    fig.colorbar(hb, ax=ax_joint, label="log sample count")

    fig.suptitle(
        "Current SFT t/sigma sampling: "
        f"shift={config.objective.shift:g}, steps={config.objective.num_train_timesteps}, "
        f"t_sampling_bias={bias:g}",
        y=0.99,
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_path)
    fig.savefig(svg_path)

    print(f"config={config_path}")
    print(f"shift={config.objective.shift:g}")
    print(f"num_train_timesteps={config.objective.num_train_timesteps}")
    print(f"t_sampling_bias={bias:g}")
    print(f"scheduler_grid_points={valid_count}")
    print(f"samples={sample_count}")
    print("t_quantiles_1_5_10_25_50_75_90_95_99=" + ",".join(f"{x:.6f}" for x in t_quantiles))
    print("sigma_quantiles_1_5_10_25_50_75_90_95_99=" + ",".join(f"{x:.6f}" for x in sigma_quantiles))
    print(f"saved_png={png_path.resolve()}")
    print(f"saved_svg={svg_path.resolve()}")


if __name__ == "__main__":
    main()
